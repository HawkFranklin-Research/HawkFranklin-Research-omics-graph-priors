"""Patient metadata and train-fitted clinical covariates."""
import numpy as np
import pandas as pd


def parse_stage(value):
    if pd.isna(value):
        return np.nan
    text = str(value).upper().strip().removeprefix("STAGE ")
    for roman, number in (("IV", 4), ("III", 3), ("II", 2), ("I", 1)):
        if text in {roman, roman + "A", roman + "B", roman + "C"}:
            return number
    return np.nan


def build_patient_table(processed_root, cancer, samples, endpoint_spec):
    root = processed_root / "clinical"
    patients = pd.read_parquet(root / "patient_master.parquet")
    targets = pd.read_parquet(root / "targets.parquet")
    patients = patients.loc[patients["cancer"].eq(cancer)].drop_duplicates("patient_id")
    targets = targets.drop_duplicates("patient_id")
    data = samples.merge(patients, on="patient_id", how="left", suffixes=("", "_clinical"), indicator=True)
    join_rate = float(data["_merge"].eq("both").mean())
    if join_rate < .98:
        raise ValueError("Clinical patient join rate below 98%")
    data = data.drop(columns="_merge").merge(targets, on="patient_id", how="left", suffixes=("", "_target"))
    data["age"] = pd.to_numeric(data.get("AGE", data.get("Age")), errors="coerce")
    data["male"] = data.get("SEX", data.get("Sex")).astype(str).str.upper().map({"MALE": 1, "FEMALE": 0})
    data["stage_num"] = data.get("AJCC_PATHOLOGIC_TUMOR_STAGE", data.get("Stage")).map(parse_stage)
    for name, endpoint in endpoint_spec.get("binary", {}).items():
        raw = data[endpoint["column"]]
        data[name] = raw.map({**{v: 1 for v in endpoint["positive"]}, **{v: 0 for v in endpoint["negative"]}})
    for name, endpoint in endpoint_spec.get("multiclass", {}).items():
        data[name] = data[endpoint["column"]].where(data[endpoint["column"]].isin(endpoint["classes"]))
    data.attrs["join_rate"] = join_rate
    return data


def encode_covariates(frame, columns, train_idx):
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        train = values.iloc[train_idx]
        if col == "hpv":
            out[col + "_pos"] = values.eq(1).astype(float)
            out[col + "_unknown"] = values.isna().astype(float)
        else:
            median = train.median()
            if pd.isna(median):
                raise ValueError(f"No training values for clinical covariate {col}")
            if col == "stage_num":
                out["stage_missing"] = values.isna().astype(float)
            values = values.fillna(median)
            if col == "age":
                mean, sd = train.mean(), train.std()
                values = (values - mean) / (sd if sd > 0 else 1)
            out[col] = values
    return out
