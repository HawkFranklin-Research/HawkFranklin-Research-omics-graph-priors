"""Audit all requested grid units, including missing and failed runs."""
import json
import pandas as pd
from .config import config_hash
from .protocol import baseline_names


def expected_runs(cfg, paths, cancer=None):
    folds = json.loads((paths.stage("02_folds") / "folds.json").read_text())
    units = [("05_runs", c["name"]) for c in cfg["conditions"]]
    units += [("06_baselines", n) for n in baseline_names(cfg, cancer)]
    expected = []
    for fold in folds:
        r, f = fold["repeat"], fold["fold"]
        if r >= cfg.get("execution_max_seeds", cfg["cv"]["n_repeats"]) or f >= cfg.get("execution_max_folds", cfg["cv"]["n_folds"]):
            continue
        for root, condition in units:
            expected.append((root, condition, r, f))
    return expected


def audit_runs(cfg, paths, metrics=None, cancer=None):
    rows = []
    present = set() if metrics is None or metrics.empty else set(metrics[["condition", "repeat", "fold"]].itertuples(index=False, name=None))
    for root, condition, r, f in expected_runs(cfg, paths, cancer):
        file = paths.stage(root) / condition / f"r{r}_f{f}" / "status.json"
        record = json.loads(file.read_text()) if file.exists() else {"status": "missing"}
        if record.get("status") == "done" and record.get("config_hash") != config_hash(cfg):
            record = dict(status="stale", reason="Configuration hash mismatch")
        if record.get("status") != "done" or (metrics is not None and (condition, r, f) not in present):
            if record.get("status") == "done": record = dict(status="no_metrics", reason="No evaluable metrics; see endpoint_failures.csv")
            rows.append(dict(condition=condition, repeat=r, fold=f, **record))
    pd.DataFrame(rows, columns=list(dict.fromkeys(["condition", "repeat", "fold", "status"] + [k for row in rows for k in row]))).to_csv(paths.stage("07_eval") / "missing_runs.csv", index=False)
    if metrics is not None:
        missing = {(row["condition"], row["repeat"], row["fold"]) for row in rows}
        assert all((c, r, f) in present | missing for _, c, r, f in expected_runs(cfg, paths, cancer))
