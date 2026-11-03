#!/usr/bin/env python3
"""Read-only equivalence check of frozen and generalized pipeline outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["condition", "repeat", "fold", "metric"]
NEW_MANIFEST_KEYS = {"git_dirty_tracked", "git_untracked_count", "protocol_file", "protocol_sha256", "hf_repo", "hf_revision", "local_processed_root", "numerics_policy", "torch_deterministic_algorithms", "cublas_workspace_config"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def expected_addition(path):
    return (path in {"protocol.md", "source_diff.patch"}
            or (path.startswith("10_attribution/") and "_rank_percentile_" in Path(path).name and path.endswith(".csv")))


def output_files(root):
    return {str(file.relative_to(root)) for file in root.rglob("*") if file.is_file()}


def compare_metric_rows(reference, candidate, baseline_conditions, model_tol):
    """Require identical keys/columns; baseline values exact, model values absolute-tolerant."""
    if set(reference.columns) != set(candidate.columns) or not set(KEYS + ["value"]) <= set(reference.columns):
        raise ValueError("Metric columns differ or required columns are missing")
    if reference.duplicated(KEYS).any() or candidate.duplicated(KEYS).any():
        raise ValueError("Duplicate metric keys")
    a, b = reference.set_index(KEYS).sort_index(), candidate.set_index(KEYS).sort_index()
    if not a.index.equals(b.index): raise ValueError("Missing or extra metric rows")
    b = b[a.columns]
    baseline = a.index.get_level_values("condition").isin(baseline_conditions)
    pd.testing.assert_frame_equal(a[baseline], b[baseline], check_exact=True)
    model_a, model_b = a[~baseline], b[~baseline]
    av, bv = model_a.value.to_numpy(dtype=float), model_b.value.to_numpy(dtype=float)
    close = np.isclose(av, bv, rtol=0, atol=model_tol, equal_nan=True)
    if not close.all():
        differences = pd.DataFrame(dict(reference=av[~close], candidate=bv[~close],
                                        absdiff=np.abs(av[~close]-bv[~close])), index=model_a.index[~close])
        raise ValueError(f"Model metrics differ beyond {model_tol:g}; {len(differences)} failing rows:\n"
                         + differences.to_string(float_format=lambda x: f"{x:.12g}"))
    pd.testing.assert_frame_equal(model_a.drop(columns="value"), model_b.drop(columns="value"), check_exact=True)


def compare_runs(reference_dir, candidate_dir, model_tol=1e-4):
    if model_tol < 0: raise ValueError("model tolerance must be nonnegative")
    reference, candidate = Path(reference_dir), Path(candidate_dir)
    rows = []
    def check(name, operation):
        try:
            detail = operation()
            rows.append(dict(check=name, status="PASS", detail=str(detail or "")))
        except (AssertionError, ValueError, OSError, KeyError, TypeError) as exc:
            rows.append(dict(check=name, status="FAIL", detail=str(exc)))
    def exact_file(path):
        if not (reference/path).is_file() or not (candidate/path).is_file():
            raise ValueError(f"Missing {path}")
        if sha256(reference/path) != sha256(candidate/path): raise ValueError(f"SHA256 differs: {path}")
        return path
    check("fold SHA256", lambda: exact_file("02_folds/folds.json"))
    def masks():
        files_a = {p.name for p in (reference/"05_runs/_eval_masks").glob("*.npz")}
        files_b = {p.name for p in (candidate/"05_runs/_eval_masks").glob("*.npz")}
        if not files_a or files_a != files_b: raise ValueError("Missing or extra evaluation masks")
        for name in sorted(files_a): exact_file("05_runs/_eval_masks/" + name)
        return f"{len(files_a)} exact archives"
    check("evaluation-mask SHA256", masks)
    check("node CSV exact", lambda: exact_file("03_nodes/nodes.csv"))
    def metrics():
        baseline_conditions = {p.name for root in (reference, candidate) for p in (root/"06_baselines").iterdir() if p.is_dir()}
        a = pd.read_csv(reference/"07_eval/per_run_metrics.csv")
        b = pd.read_csv(candidate/"07_eval/per_run_metrics.csv")
        compare_metric_rows(a, b, baseline_conditions, model_tol)
        return f"{len(a)} matched rows; baselines exact, models atol={model_tol:g}"
    check("metric values", metrics)
    if reference.is_dir() and candidate.is_dir():
        a, b = output_files(reference), output_files(candidate)
        stages = sorted({p.split("/")[0] for p in a | b if "/" in p})
        for stage in stages:
            def names(stage=stage):
                left = {p for p in a if p.startswith(stage + "/")}
                right = {p for p in b if p.startswith(stage + "/")}
                missing = left-right
                extra = {p for p in right-left if not expected_addition(p)}
                if missing or extra: raise ValueError(f"missing={sorted(missing)}, extra={sorted(extra)}")
                return f"{len(left)} reference files"
            check("file names: " + stage, names)
        def root_names():
            missing = {p for p in a-b if "/" not in p}
            extra = {p for p in b-a if "/" not in p and not expected_addition(p)}
            if missing or extra: raise ValueError(f"missing={sorted(missing)}, extra={sorted(extra)}")
        check("root file names", root_names)
        additions = sorted(p for p in b-a if expected_addition(p))
        rows.append(dict(check="expected additions", status="PASS", detail=", ".join(additions) or "none"))
    else:
        rows.append(dict(check="output directories", status="FAIL", detail="Both directories must exist"))
    def manifest_keys():
        a = json.loads((reference/"run_manifest.json").read_text())
        b = json.loads((candidate/"run_manifest.json").read_text())
        if set(a)-set(b) or set(b)-set(a)-NEW_MANIFEST_KEYS:
            raise ValueError("Unexpected missing or added manifest keys")
        return "expected new keys: " + ", ".join(sorted(set(b)-set(a)))
    check("manifest keys", manifest_keys)
    return rows


def rank_diagnostics(labels, reference_scores, candidate_scores):
    """Quantify rank sensitivity without rounding, re-scoring stored metrics or changing acceptance."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels)
    a, b = np.asarray(reference_scores, dtype=float), np.asarray(candidate_scores, dtype=float)
    if labels.ndim != 1 or a.shape != labels.shape or b.shape != labels.shape:
        raise ValueError("Rank diagnostic arrays must have the same one-dimensional shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Rank diagnostics require finite scores and both binary classes")
    positive, negative = labels == 1, labels == 0
    ap, an, bp, bn = a[positive], a[negative], b[positive], b[negative]
    changed, margin = 0, 0.
    # Bound temporary allocations even if this audit is used on a larger fold.
    chunk = max(1, 1000000 // len(an))
    for start in range(0, len(ap), chunk):
        da = ap[start:start+chunk, None] - an[None, :]
        db = bp[start:start+chunk, None] - bn[None, :]
        different = np.sign(da) != np.sign(db)
        changed += int(different.sum())
        if different.any(): margin = max(margin, float(np.maximum(np.abs(da[different]), np.abs(db[different])).max()))
    return dict(n_entries=len(labels), n_positive=int(positive.sum()), n_negative=int(negative.sum()),
                max_score_absdiff=float(np.max(np.abs(a-b))), positive_negative_pairs=int(len(ap)*len(an)),
                changed_pair_orders=changed, max_changed_pair_score_margin=margin,
                replay_reference_auroc=float(roc_auc_score(labels, a)), replay_candidate_auroc=float(roc_auc_score(labels, b)),
                replay_reference_auprc=float(average_precision_score(labels, a)), replay_candidate_auprc=float(average_precision_score(labels, b)))


def load_prediction_context(root, condition, repeat, fold):
    """Read saved artifacts only; never call stages, fit transforms, or construct output paths."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import yaml
    from graphr1g.data import FoldTensorData
    from graphr1g.graphs.io import load_graph
    from graphr1g.models import build_model
    from graphr1g.normalize import ChannelNormalizer
    cfg = yaml.safe_load((root/"config.resolved.yaml").read_text())
    split = next(row for row in json.loads((root/"02_folds/folds.json").read_text())
                 if row["repeat"] == repeat and row["fold"] == fold)
    genes = pd.read_csv(root/"03_nodes/nodes.csv").gene.astype(str).tolist()
    with np.load(root/"01_cube/cube.npz", allow_pickle=False) as cube:
        ids, universe = cube["patient_ids"].astype(str), cube["genes"].astype(str).tolist()
        positions = [universe.index(gene) for gene in genes]
        values, observed = cube["values"][:, positions, :], cube["observed"][:, positions, :]
    normalizer = ChannelNormalizer.load(root/"04_graphs/normalization"/f"r{repeat}_f{fold}")
    data = FoldTensorData(values, observed, normalizer, ids)
    lookup = {patient: i for i, patient in enumerate(ids)}
    test = np.array([lookup[patient] for patient in split["test_ids"]])
    with np.load(root/"05_runs/_eval_masks"/f"r{repeat}_f{fold}.npz", allow_pickle=False) as source:
        masks = {name: source[name].copy() for name in ("entry", "node")}
        if not np.array_equal(source["patient_ids"], ids[test]): raise ValueError("Mask patient order differs")
    checkpoint = torch.load(root/"05_runs"/condition/f"r{repeat}_f{fold}"/"model.pt", map_location="cpu", weights_only=True)
    spec = checkpoint["spec"]
    model = build_model(spec, len(genes), len(cfg["channels"]), checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to("cpu").eval()
    name = "string" if spec["graph"] == "none" else spec["graph"]
    graph = load_graph(root/"04_graphs"/name/f"r{repeat}_f{fold}")
    return dict(cfg=cfg, genes=genes, ids=ids[test], data=data, test=test, masks=masks, graph=graph, model=model)


def replay_scores(context, task, channel):
    import torch
    data, graph, test = context["data"], context["graph"], context["test"]
    mask = context["masks"][task]
    batch_size = context["cfg"]["resources"]["batch_size"]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            idx, hidden = test[start:start+batch_size], mask[start:start+batch_size]
            batch = data.make_batch(idx, graph, hidden).to("cpu")
            pred = context["model"](batch)["recon"].cpu().numpy().reshape(len(idx), len(context["genes"]), -1)
            predictions.append(pred[:, :, channel])
    selected = mask[:, :, channel].astype(bool)
    return data.x[test, :, channel][selected], np.concatenate(predictions)[selected]


def audit_prediction_case(reference, candidate, condition, repeat, fold, metric):
    """Replay two saved checkpoints under one CPU execution policy; diagnostic evidence only."""
    import torch
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        with torch.random.fork_rng(devices=[]):
            a = load_prediction_context(reference, condition, repeat, fold)
            b = load_prediction_context(candidate, condition, repeat, fold)
            settings = [(task, i, spec["name"], score) for task in ("entry", "node")
                        for i, spec in enumerate(a["cfg"]["channels"]) if spec["type"] == "binary"
                        for score in ("auroc", "auprc") if metric == f"{task}_{spec['name']}_{score}"]
            if len(settings) != 1: raise ValueError("CPU rank replay applies only to binary reconstruction AUROC/AUPRC")
            task, channel, _, _ = settings[0]
            same_inputs = (a["genes"] == b["genes"] and a["cfg"]["channels"] == b["cfg"]["channels"]
                           and np.array_equal(a["ids"], b["ids"])
                           and np.array_equal(a["data"].x[a["test"]], b["data"].x[b["test"]])
                           and np.array_equal(a["data"].m[a["test"]], b["data"].m[b["test"]])
                           and np.array_equal(a["masks"][task], b["masks"][task])
                           and torch.equal(a["graph"].edge_index, b["graph"].edge_index)
                           and torch.equal(a["graph"].edge_weight, b["graph"].edge_weight))
            if not same_inputs: raise ValueError("Replay inputs, masks or graph differ; rank-noise diagnosis is not established")
            sa, sb = a["model"].state_dict(), b["model"].state_dict()
            if set(sa) != set(sb) or any(sa[key].shape != sb[key].shape for key in sa):
                raise ValueError("Checkpoint parameter keys or shapes differ")
            weights_exact = all(torch.equal(sa[key], sb[key]) for key in sa)
            max_weight_delta = max((float((sa[key].double()-sb[key].double()).abs().max()) for key in sa if sa[key].numel()), default=0.)
            ya, pa = replay_scores(a, task, channel)
            yb, pb = replay_scores(b, task, channel)
            if not np.array_equal(ya, yb): raise ValueError("Replay labels differ")
            return dict(device="cpu", inputs_exact=True, checkpoint_parameters_exact=weights_exact,
                        max_checkpoint_absdiff=max_weight_delta, **rank_diagnostics(ya, pa, pb))
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def audit_failed_predictions(reference, candidate, model_tol):
    """Audit only failing rank rows; never use replay values to replace the strict CSV verdict."""
    reference, candidate = Path(reference), Path(candidate)
    a = pd.read_csv(reference/"07_eval/per_run_metrics.csv").set_index(KEYS).sort_index()
    b = pd.read_csv(candidate/"07_eval/per_run_metrics.csv").set_index(KEYS).sort_index()
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index):
        raise ValueError("Prediction audit requires unique, matched metric rows")
    baseline = {p.name for root in (reference, candidate) for p in (root/"06_baselines").iterdir() if p.is_dir()}
    different = ~np.isclose(a.value.to_numpy(float), b.value.to_numpy(float), rtol=0, atol=model_tol, equal_nan=True)
    result = []
    for condition, repeat, fold, metric in a.index[different]:
        if condition in baseline or not metric.endswith(("_auroc", "_auprc")): continue
        row = dict(condition=condition, repeat=int(repeat), fold=int(fold), metric=metric,
                   stored_reference=float(a.loc[(condition, repeat, fold, metric), "value"]),
                   stored_candidate=float(b.loc[(condition, repeat, fold, metric), "value"]))
        try:
            row.update(audit_prediction_case(reference, candidate, condition, int(repeat), int(fold), metric))
        except Exception as exc:
            row["audit_error"] = f"{type(exc).__name__}: {exc}"
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--model-tol", type=float, default=1e-4)
    parser.add_argument("--audit-predictions", action="store_true",
                        help="Replay saved checkpoints for failing binary reconstruction rank rows on CPU; never alters PASS/FAIL")
    args = parser.parse_args()
    if args.model_tol < 0: parser.error("--model-tol must be nonnegative")
    rows = compare_runs(args.reference_dir, args.candidate_dir, args.model_tol)
    print("RESULT | CHECK | DETAIL")
    for row in rows: print(f"{row['status']} | {row['check']} | {row['detail']}")
    if args.audit_predictions:
        print("CPU CHECKPOINT AUDIT (diagnostic only; strict CSV verdict unchanged)")
        try:
            print(json.dumps(audit_failed_predictions(args.reference_dir, args.candidate_dir, args.model_tol), indent=2))
        except Exception as exc:
            print(f"AUDIT ERROR | {type(exc).__name__}: {exc}")
    raise SystemExit(1 if any(row["status"] == "FAIL" for row in rows) else 0)


if __name__ == "__main__":
    main()
