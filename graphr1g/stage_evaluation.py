"""Graph-free baselines and independent endpoint evaluation."""
import json
import traceback
import numpy as np
import pandas as pd
from .baselines import fit_pca, LowRankImputer
from .clinical import encode_covariates
from .evaluate.downstream import binary_endpoint, survival_endpoint
from .evaluate.recon import reconstruction_metrics
from .normalize import ChannelNormalizer
from .config import config_hash
from .runtime import successful_run, run_guard
from .protocol import cancer_protocol, baseline_names, single_gene_name, check_single_gene


def baseline_stage(cfg, cancer, paths, force=False):
    cube = np.load(paths.stage("01_cube") / "cube.npz")
    ids, universe = cube["patient_ids"].astype(str), cube["genes"].astype(str).tolist()
    genes = pd.read_csv(paths.stage("03_nodes") / "nodes.csv").gene.tolist()
    indices = [universe.index(g) for g in genes]
    values, observed = cube["values"][:, indices, :], cube["observed"][:, indices, :]
    id_map = {pid: i for i, pid in enumerate(ids)}
    folds = json.loads((paths.stage("02_folds") / "folds.json").read_text())
    for fold in folds:
        r, f = fold["repeat"], fold["fold"]
        if r >= cfg.get("execution_max_seeds", cfg["cv"]["n_repeats"]) or f >= cfg.get("execution_max_folds", cfg["cv"]["n_folds"]): continue
        train = np.array([id_map[x] for x in fold["train_ids"]])
        test = np.array([id_map[x] for x in fold["test_ids"]])
        normalizer = ChannelNormalizer([c["type"] for c in cfg["channels"]]).fit(values, observed, train, ids)
        x, mask = normalizer.transform(values, observed)
        fixed = np.load(paths.eval_mask(r, f))
        if not np.array_equal(fixed["patient_ids"], ids[test]):
            raise ValueError("Baseline evaluation mask patient order mismatch")
        split = np.full(len(ids), "test", dtype=object)
        split[train] = "train"
        split[[id_map[x] for x in fold["val_ids"]]] = "val"
        protocol = cancer_protocol(cfg, cancer)
        names = baseline_names(cfg, cancer)
        for name in names:
            directory = paths.baseline(name, r, f)
            directory.mkdir(parents=True, exist_ok=True)
            if not force and successful_run(directory, cfg): continue
            try:
                rows = []
                if name == "mofa":
                    from .baselines import fit_mofa
                    emb = fit_mofa(x, mask, train, cfg["model"]["latent"], r)
                    frame = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
                    frame.insert(0, "split", split); frame.insert(0, "patient_id", ids)
                    frame.to_parquet(directory / "embeddings_gene_latent_pca.parquet", index=False)
                elif name.startswith("pca_"):
                    channel = None if name == "pca_concat" else 0
                    model, emb = fit_pca(x, train, cfg["model"]["latent"], channel)
                    frame = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])]); frame.insert(0, "split", split); frame.insert(0, "patient_id", ids)
                    frame.to_parquet(directory / "embeddings_gene_latent_pca.parquet", index=False)
                elif name == single_gene_name(protocol):
                    gene = protocol["single_gene"]["gene"]
                    if gene not in universe: raise ValueError(f"{gene} absent from input cube; control not testable")
                    gene_index = universe.index(gene)
                    reference = cube["values"][:, gene_index:gene_index+1, :]
                    reference_mask = cube["observed"][:, gene_index:gene_index+1, :]
                    reference_normalizer = ChannelNormalizer([c["type"] for c in cfg["channels"]]).fit(reference, reference_mask, train, ids)
                    reference_values, _ = reference_normalizer.transform(reference, reference_mask)
                    pd.DataFrame(dict(patient_id=ids, split=split, emb_0=reference_values[:, 0, 0])).to_parquet(directory / "embeddings_gene_latent_pca.parquet", index=False)
                else:
                    model = LowRankImputer(rank=cfg["model"]["latent"]).fit(x[train], mask[train]) if name == "lowrank_impute" else None
                if name in ("pca_concat", "lowrank_impute", "gene_mean"):
                    for mode in ("entry", "node"):
                        hidden = fixed[mode]
                        visible = mask[test].astype(bool) & ~hidden
                        test_input = np.where(visible, x[test], 0)
                        if name == "pca_concat":
                            pred = model.inverse_transform(model.transform(test_input.reshape(len(test), -1))).reshape(test_input.shape)
                        elif name == "lowrank_impute": pred = model.transform(test_input, visible)
                        else: pred = np.zeros_like(test_input)
                        rows.extend(dict(condition=name, repeat=r, fold=f, task=mode, **item) for item in reconstruction_metrics(x[test], pred, hidden, cfg["channels"]))
                    pd.DataFrame(rows).to_csv(directory / "recon_metrics.csv", index=False)
                (directory / "status.json").write_text(json.dumps(dict(status="done", config_hash=config_hash(cfg)), indent=2))
            except Exception as exc:
                (directory / "status.json").write_text(json.dumps(dict(status="failed", config_hash=config_hash(cfg), error=str(exc), traceback=traceback.format_exc()), indent=2))


def evaluate_stage(cfg, cancer, paths):
    from .audit import audit_runs, expected_runs
    audit_runs(cfg, paths, cancer=cancer)
    protocol = cancer_protocol(cfg, cancer)
    patients = pd.read_csv(paths.stage("01_cube") / "patients.csv")
    endpoints = cfg["endpoints"][cancer]
    metrics, failures = [], []
    expected = {(c, r, f) for _, c, r, f in expected_runs(cfg, paths, cancer)}
    for root, condition, repeat, fold in expected_runs(cfg, paths, cancer):
        directory = paths.run(condition, repeat, fold) if root == "05_runs" else paths.baseline(condition, repeat, fold)
        if not successful_run(directory, cfg): continue
        with run_guard(paths.stage("07_eval") / "runs" / condition / directory.name, cfg):
            condition = directory.parent.name
            repeat, fold = map(int, directory.name.replace("r", "").split("_f"))
            if (condition, repeat, fold) not in expected or not successful_run(directory, cfg): continue
            if (directory / "recon_metrics.csv").exists():
                for row in pd.read_csv(directory / "recon_metrics.csv").to_dict("records"):
                    metrics.append(dict(condition=condition, repeat=repeat, fold=fold, metric=f"{row['task']}_{row['channel']}_{row['metric']}", value=row["value"]))
            for embedding_file in directory.glob("embeddings_*.parquet"):
                readout = embedding_file.stem.removeprefix("embeddings_")
                frame = pd.read_parquet(embedding_file).merge(patients, on="patient_id", how="left")
                cols = [c for c in frame if c.startswith("emb_")]
                emb = frame[cols].to_numpy()
                train = np.flatnonzero(frame.split.eq("train")); test = np.flatnonzero(frame.split.eq("test"))
                for name in endpoints.get("binary", {}):
                    try:
                        score, predictions, _ = binary_endpoint(emb, frame[name], train, test)
                        predictions["patient_id"] = frame.patient_id.iloc[predictions["index"]].to_numpy()
                        predictions.to_parquet(directory / f"predictions_{name}_{readout}.parquet", index=False)
                        metrics.extend(dict(condition=condition, repeat=repeat, fold=fold, metric=f"{name}_{key}_{readout}", value=value) for key, value in score.items())
                    except Exception as exc: failures.append(dict(condition=condition, repeat=repeat, fold=fold, metric=name, reason=str(exc), traceback=traceback.format_exc()))
                for survival_name, name in endpoints.get("survival", {}).items():
                    try:
                        clinical = encode_covariates(frame, endpoints["clinical_covariates"], train).to_numpy()
                        results = {}
                        for label, features in (("clinical", clinical), ("embedding", emb), ("joint", np.column_stack([clinical, emb]))):
                            results[label], fitted = survival_endpoint(features, frame[name["time"]], frame[name["event"]], train, test)
                            fitted.v2_predictions_.to_csv(directory / f"survival_{survival_name}_{label}_{readout}.csv", index=False)
                            for metric in ("harrell", "uno"):
                                metrics.append(dict(condition=condition, repeat=repeat, fold=fold, metric=f"{survival_name}_{label}_{metric}_{readout}", value=results[label][metric]))
                        metrics.append(dict(condition=condition, repeat=repeat, fold=fold, metric=f"{survival_name}_delta_c_{readout}", value=results["joint"]["harrell"]-results["clinical"]["harrell"]))
                    except Exception as exc: failures.append(dict(condition=condition, repeat=repeat, fold=fold, metric=survival_name, reason=str(exc), traceback=traceback.format_exc(), fit_failed=1))
                for endpoint in endpoints.get("multiclass", {}):
                    try:
                        from .evaluate.downstream import multiclass_endpoint
                        result = multiclass_endpoint(emb, frame[endpoint], train, test)
                        metrics.extend(dict(condition=condition, repeat=repeat, fold=fold, metric=f"{endpoint}_{key}_{readout}", value=value) for key, value in result.items())
                    except Exception as exc:
                        failures.append(dict(condition=condition, repeat=repeat, fold=fold, metric=endpoint, reason=str(exc), traceback=traceback.format_exc()))
    out = paths.stage("07_eval")
    table = pd.DataFrame(metrics, columns=["condition", "repeat", "fold", "metric", "value"])
    table.to_csv(out / "per_run_metrics.csv", index=False)
    audit_runs(cfg, paths, table, cancer)
    pd.DataFrame(failures).to_csv(out / "endpoint_failures.csv", index=False)
    if table.empty: raise ValueError("No evaluation metrics were generated")
    from .checks import check_evaluation
    check_evaluation(table)
    from .evaluate.aggregate import aggregate_metrics, compare_metrics
    aggregate_metrics(table).to_csv(out / "summary_by_condition.csv", index=False)
    n = len(patients); compare_metrics(table, n * (1-1/cfg["cv"]["n_folds"]) * (1-cfg["cv"]["validation_fraction"]), n / cfg["cv"]["n_folds"], primary_metrics=protocol["primary_metrics"]).to_csv(out / "comparisons.csv", index=False)
    exploratory = compare_metrics(table, n * (1-1/cfg["cv"]["n_folds"]) * (1-cfg["cv"]["validation_fraction"]), n / cfg["cv"]["n_folds"], exploratory=True, primary_metrics=protocol["primary_metrics"])
    exploratory.to_csv(out / "comparisons_exploratory.csv", index=False)
    from sklearn.metrics import roc_auc_score
    from .evaluate.stats import bootstrap_ci
    pooled_rows = []
    predictions = {}
    for root in (paths.stage("05_runs"), paths.stage("06_baselines")):
        for file in root.glob("*/r*_f*/predictions_*_gene_latent_pca.parquet"):
            repeat = int(file.parent.name.split("_")[0][1:])
            fold = int(file.parent.name.split("_f")[1])
            if (file.parent.parent.name, repeat, fold) not in expected or not successful_run(file.parent, cfg): continue
            key = (file.parent.parent.name, repeat, file.name)
            predictions.setdefault(key, []).append(pd.read_parquet(file))
    for (condition, repeat, filename), frames in predictions.items():
        frame = pd.concat(frames, ignore_index=True)
        if frame.patient_id.duplicated().any() or frame.label.nunique() < 2:
            failures.append(dict(condition=condition, repeat=repeat, metric=filename, reason="pooled predictions have duplicate IDs or one class"))
            continue
        point = roc_auc_score(frame.label, frame.score)
        try:
            interval = bootstrap_ci(lambda sample: roc_auc_score(sample[:, 0], sample[:, 1]), frame[["label", "score"]].to_numpy(), cfg["evaluation"]["bootstrap_n"], repeat)
            pooled_rows.append(dict(condition=condition, repeat=repeat, endpoint=filename, auroc=point, ci_low=interval[0], ci_high=interval[1], n=len(frame)))
        except ValueError as exc:
            failures.append(dict(condition=condition, repeat=repeat, metric=filename, reason=f"bootstrap: {exc}"))
    pd.DataFrame(pooled_rows).to_csv(out / "pooled_classification.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "endpoint_failures.csv", index=False)
    from .evaluate.figures import required_figures
    required_figures(table, paths, protocol["primary_metrics"])
    if not cfg["checks"]["relaxed"]:
        check_single_gene(table, protocol)
        mean = table[(table.condition == "gene_mean") & table.metric.str.endswith("_r2")]
        assert (mean.value.abs() < .02).all(), mean.to_dict("records")
