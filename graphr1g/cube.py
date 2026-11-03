"""Build raw patient-by-gene-by-channel matrices from harmonized tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .clinical import build_patient_table
from .symbols import HGNCHarmonizer, source_symbols


def subsample_patients(patients, limit, label, event, seed):
    """Keep the original joint split when viable, otherwise use the fold fallback order."""
    import logging
    from sklearn.model_selection import train_test_split
    labels = patients[label].fillna("NA").astype(str)
    events = patients[event].fillna("NA").astype(str)
    strata, chosen = None, "unstratified"
    for name, candidate in (("label×event", labels + ":" + events), ("label", labels), ("event", events)):
        counts = candidate.value_counts()
        if counts.min() >= 2 and len(counts) <= min(limit, len(patients)-limit):
            strata, chosen = candidate, name
            break
    logging.getLogger(__name__).info("Cube subsample strata: %s", chosen)
    selected, _ = train_test_split(patients, train_size=limit, random_state=seed, stratify=strata)
    return selected


def build_cube(cfg, cancer, hgnc, destination):
    root = Path(cfg["data"]["processed_root"])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    harmonizer = HGNCHarmonizer(hgnc, cfg["symbols"]["locus_groups"])
    frames, maps, sample_frames, duplicates = {}, [], [], []
    for cohort in cfg["cohorts"]:
        for channel in cfg["channels"]:
            path = root / "harmonized" / cohort.lower() / cancer / (channel["modality"] + ".parquet")
            raw = pd.read_parquet(path)
            meta = pd.read_parquet(path.with_name(path.stem + ".samples.parquet"))
            meta = meta[meta["tissue"].astype(str).str.lower().eq("tumor")].copy()
            meta = meta[meta["sample_id"].astype(str).str.contains(r"-01(?:[A-Z]|$)", regex=True)]
            meta["priority"] = (~meta["sample_id"].astype(str).str.contains("-01A", regex=False)).astype(int)
            meta = meta.sort_values(["priority", "sample_id"])
            dropped = meta[meta.duplicated("patient_id")].copy()
            dropped["modality"] = channel["modality"]
            duplicates.append(dropped)
            meta = meta.drop_duplicates("patient_id")
            sample_frames.append(meta[["sample_id", "patient_id", "cohort", "cancer"]])
            raw = meta[["sample_id", "patient_id"]].merge(raw, on="sample_id", how="inner").set_index("patient_id")
            source_cols = [c for c in raw.columns if c.startswith("GENE:")]
            mapping_rows = source_symbols(source_cols)
            mapped = harmonizer.map([s for _, s in mapping_rows])
            mapped["source_column"] = [c for c, _ in mapping_rows]
            mapped["cohort"], mapped["modality"] = cohort, channel["modality"]
            maps.append(mapped)
            mapped = mapped[mapped["approved"].notna()]
            mapped = mapped.drop_duplicates(["approved", "source_column"])
            columns = mapped["source_column"].tolist()
            source_values = raw[columns].apply(pd.to_numeric, errors="coerce")
            source_values.columns = mapped["approved"].tolist()
            grouped = source_values.T.groupby(level=0, sort=False)
            data = (grouped.max() if channel["type"] == "binary" else grouped.mean()).T
            transform = channel.get("transform")
            if transform == "rna_auto_log":
                if np.nanpercentile(data.to_numpy(), 99) > 50:
                    data = np.log2(data.clip(lower=0) + 1)
            elif transform == "methylation_m":
                data = data.clip(.001, .999)
                data = np.log2(data / (1 - data))
            elif transform == "cnv_clip":
                data = data.clip(-2, 2)
            frames[(cohort, channel["name"])] = data
    combined_samples = pd.concat(sample_frames).drop_duplicates("patient_id")
    required = cfg["cube"]["required_channels"]
    eligible = set(combined_samples.patient_id)
    for channel in required:
        eligible &= set().union(*(set(frame.index) for (cohort, name), frame in frames.items() if name == channel))
    samples = combined_samples[combined_samples.patient_id.isin(eligible)].copy()
    patients = build_patient_table(root, cancer, samples, cfg["endpoints"][cancer])
    (destination / "patient_join.json").write_text(json.dumps(dict(join_rate=patients.attrs["join_rate"]), indent=2))
    limit = cfg["cube"].get("max_patients")
    if limit and len(patients) > limit:
        label = cfg["cv"]["stratify_label"]
        event = cfg["cv"]["stratify_event"]
        patients = subsample_patients(patients, limit, label, event, cfg["cube"]["seed"])
    patients = patients.sort_values("patient_id").reset_index(drop=True)
    ids = patients.patient_id.astype(str).tolist()
    genes = sorted(set().union(*(set(f.columns) for f in frames.values())))
    values = np.full((len(ids), len(genes), len(cfg["channels"])), np.nan, dtype=np.float32)
    index = {g: i for i, g in enumerate(genes)}
    for c, channel in enumerate(cfg["channels"]):
        for cohort in cfg["cohorts"]:
            frame = frames[(cohort, channel["name"])].reindex(ids)
            positions = [index[g] for g in frame.columns]
            arr = frame.to_numpy(dtype=np.float32)
            existing = values[:, positions, c]
            values[:, positions, c] = np.where(np.isfinite(arr), arr, existing)
    observed = np.isfinite(values)
    rna = next(i for i, c in enumerate(cfg["channels"]) if c["name"] == cfg["cube"]["coverage_channel"])
    keep = observed[:, :, rna].mean(axis=0) >= cfg["cube"]["min_rna_coverage"]
    values, observed = values[:, keep, :], observed[:, keep, :]
    genes = np.array(genes)[keep]
    np.savez_compressed(destination / "cube.npz", values=values, observed=observed.astype(np.uint8), patient_ids=np.array(ids), genes=genes)
    patients.to_csv(destination / "patients.csv", index=False)
    meta = hgnc.drop_duplicates("symbol").set_index("symbol").reindex(genes)
    gene_table = pd.DataFrame({"gene": genes, "hgnc_id": meta["hgnc_id"].to_numpy(), "locus_group": meta["locus_group"].to_numpy()})
    for c, channel in enumerate(cfg["channels"]):
        gene_table[channel["name"] + "_coverage"] = observed[:, :, c].mean(axis=0)
    gene_table.to_csv(destination / "genes.csv", index=False)
    pd.concat(duplicates).to_csv(destination / "duplicate_samples.csv", index=False)
    mapping = pd.concat(maps, ignore_index=True)
    mapping.groupby(["cohort", "modality", "method"]).size().rename("count").reset_index().to_csv(destination / "harmonization_summary.csv", index=False)
    mapping["n_sources"] = mapping.groupby(["cohort", "modality", "approved"])["source_column"].transform("nunique")
    mapping.rename(columns={"source": "source_symbol", "approved": "approved_symbol"}).to_csv(destination / "harmonization_map.csv", index=False)
    coverage = []
    rna_modality = next(c["modality"] for c in cfg["channels"] if c["name"] == cfg["cube"]["coverage_channel"])
    rna_mapping = mapping[mapping.modality.eq(rna_modality)]
    rna_before = set(rna_mapping.source.str.upper())
    rna_after = set(rna_mapping.approved.dropna())
    for c, channel in enumerate(cfg["channels"]):
        channel_mapping = mapping[mapping.modality.eq(channel["modality"])]
        before = set(channel_mapping.source.str.upper())
        after = set(channel_mapping.approved.dropna())
        coverage.append(dict(channel=channel["name"], observed_fraction=float(observed[:, :, c].mean()), genes_with_90pct_coverage=float((observed[:, :, c].mean(axis=0) >= .9).mean()), rna_gene_overlap_before=len(rna_before & before)/max(len(rna_before),1), rna_gene_overlap_after=len(rna_after & after)/max(len(rna_after),1)))
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(destination / "qc_coverage.csv", index=False)
    import matplotlib.pyplot as plt
    ax = coverage_frame.set_index("channel")[["rna_gene_overlap_before", "rna_gene_overlap_after"]].plot.bar(color=["#bd6f58", "#167d7f"], ylim=(0, 1), ylabel="Fraction of RNA gene names shared")
    ax.figure.tight_layout()
    ax.figure.savefig(destination / "qc_coverage.png", dpi=180)
    plt.close(ax.figure)
    (destination / "channels.json").write_text(json.dumps(cfg["channels"], indent=2))
    return values, observed, patients, gene_table
