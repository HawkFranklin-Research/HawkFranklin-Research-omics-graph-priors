"""A2 (protocol_brca §12.4, exploratory): node-masked reconstruction by STRING degree bin, HNSCC and BRCA.

Same method as manuscript/figure_code/analysis_degree_recon.py (HNSCC): reload each split's saved normalizer, fixed test
node mask and trained models; refit pca_concat and lowrank_impute exactly as the pipeline does; pool node-masked
entries per degree bin; R^2 = 1 - MSE / Var(y) per bin. Before binning, every recomputed overall node R^2 must match
the saved recon_metrics.csv (|diff| < 1e-3). HNSCC uses the frozen graphr1 package, BRCA the graphr1g copy (read-only).
Adds the mutation channel: observed mutation rate and node-masked mutation AUROC per bin (B6 folded in here).

Writes outputs/v2/followup/degree_recon_{hnscc,brca}.csv (+ _per_split.csv, _check.csv). CPU inference.
Usage: python3 scripts/analysis/followup_degree_recon.py {hnscc|brca}
"""
import importlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

COHORTS = {"hnscc": ("graphr1", ROOT / "outputs/v2/hnscc_v2_e1"), "brca": ("graphr1g", ROOT / "outputs/v2/brca_v2_e5")}
NEURAL = ["gatv2_string", "gcn_string", "sage_string", "gatv2_string_rewired", "gatv2_coexpr", "gcn_coexpr", "dense_ae"]
BINNINGS = {
    "protocol": [(0, 0, "0 (isolated)"), (1, 5, "1–5"), (6, 25, "6–25"), (26, 10**6, ">25")],
    "fine": [(0, 0, "0 (isolated)"), (1, 2, "1–2"), (3, 5, "3–5"), (6, 10, "6–10"), (11, 25, "11–25"), (26, 10**6, ">25")],
}
CONT = ["rna", "meth", "cnv"]
MUT = 3
BATCH = 8


def degree(graph, n):
    return np.bincount(graph.edge_index[0].numpy(), minlength=n)


def bin_of(deg, bins):
    out = np.empty(len(deg), dtype=object)
    for lo, hi, label in bins:
        out[(deg >= lo) & (deg <= hi)] = label
    return out


def r2(y, p):
    return 1 - np.mean((p - y) ** 2) / np.var(y) if len(y) > 1 and np.var(y) > 0 else np.nan


def auroc(y, p):
    return roc_auc_score(y, p) if (y == 1).sum() >= 5 and len(np.unique(y)) == 2 else np.nan


def main(cohort):
    pkg, run = COHORTS[cohort]
    baselines = importlib.import_module(f"{pkg}.baselines")
    FoldTensorData = importlib.import_module(f"{pkg}.data").FoldTensorData
    load_graph = importlib.import_module(f"{pkg}.graphs.io").load_graph
    build_model = importlib.import_module(f"{pkg}.models").build_model
    ChannelNormalizer = importlib.import_module(f"{pkg}.normalize").ChannelNormalizer
    started = time.time()
    torch.set_num_threads(6)
    cube = np.load(run / "01_cube/cube.npz")
    ids, universe = cube["patient_ids"].astype(str), cube["genes"].astype(str).tolist()
    genes = pd.read_csv(run / "03_nodes/nodes.csv").gene.astype(str).tolist()
    pos = [universe.index(g) for g in genes]
    values, observed = cube["values"][:, pos, :], cube["observed"][:, pos, :]
    del cube
    lookup = {p: i for i, p in enumerate(ids)}
    folds = json.loads((run / "02_folds/folds.json").read_text())
    string_deg = degree(load_graph(run / "04_graphs/string"), len(genes))
    rows, checks = [], []
    for fold in folds:
        r, f = fold["repeat"], fold["fold"]
        train = np.array([lookup[x] for x in fold["train_ids"]])
        test = np.array([lookup[x] for x in fold["test_ids"]])
        data = FoldTensorData(values, observed, ChannelNormalizer.load(run / f"04_graphs/normalization/r{r}_f{f}"), ids)
        masks = np.load(run / f"05_runs/_eval_masks/r{r}_f{f}.npz")
        assert np.array_equal(masks["patient_ids"], ids[test]), "mask patient order mismatch"
        node = masks["node"]
        preds = {}
        for cond in NEURAL:
            d = run / f"05_runs/{cond}/r{r}_f{f}"
            if json.loads((d / "status.json").read_text()).get("status") != "done":
                continue
            ckpt = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
            spec = ckpt["spec"]
            model = build_model(spec, len(genes), 4, ckpt["model_config"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            graph = load_graph(run / f"04_graphs/{'string' if spec['graph'] == 'none' else spec['graph']}/r{r}_f{f}")
            out = []
            with torch.no_grad():
                for s in range(0, len(test), BATCH):
                    idx = test[s:s + BATCH]
                    batch = data.make_batch(idx, graph, node[s:s + len(idx)])
                    out.append(model(batch)["recon"].numpy().reshape(len(idx), len(genes), -1))
            preds[cond] = np.concatenate(out)
        x, m = data.x, data.m
        visible = m[test].astype(bool) & ~node
        test_input = np.where(visible, x[test], 0)
        pca, _ = baselines.fit_pca(x, train, 16)
        preds["pca_concat"] = pca.inverse_transform(pca.transform(test_input.reshape(len(test), -1))).reshape(test_input.shape)
        preds["lowrank_impute"] = baselines.LowRankImputer(rank=16).fit(x[train], m[train]).transform(test_input, visible)
        target = x[test]
        node_obs = node & m[test].astype(bool)
        for cond, pred in preds.items():
            saved = pd.read_csv(run / ("05_runs" if cond in NEURAL else "06_baselines") / cond / f"r{r}_f{f}/recon_metrics.csv")
            for c, ch in enumerate(CONT):
                sel = node[..., c]
                ref = saved.query("task == 'node' and channel == @ch and metric == 'r2'").value.iloc[0]
                ours = r2(target[..., c][sel], pred[..., c][sel])
                checks.append(dict(condition=cond, repeat=r, fold=f, channel=ch, recomputed=ours, saved=ref, abs_diff=abs(ours - ref)))
            for binning, bins in BINNINGS.items():
                labels = bin_of(string_deg, bins)
                for _, _, lab in bins:
                    gmask = (labels == lab)[None, :]
                    base = dict(condition=cond, repeat=r, fold=f, binning=binning, degree_bin=lab, n_genes=int(gmask.sum()))
                    for c, ch in enumerate(CONT):
                        sel = node[..., c] & gmask
                        rows.append(dict(base, channel=ch, metric="r2", n_entries=int(sel.sum()),
                                         value=r2(target[..., c][sel], pred[..., c][sel])))
                    sel = node_obs[..., MUT] & gmask
                    y = target[..., MUT][sel]
                    rows.append(dict(base, channel="mut", metric="auroc", n_entries=int(sel.sum()), value=auroc(y, pred[..., MUT][sel])))
                    rows.append(dict(base, channel="mut", metric="positive_rate", n_entries=int(sel.sum()),
                                     value=float(y.mean()) if len(y) else np.nan))
        print(f"{cohort} r{r}_f{f} done ({time.time() - started:.0f}s)", flush=True)
    per_split, check = pd.DataFrame(rows), pd.DataFrame(checks)
    print("max |recomputed - saved| node R^2:", check.abs_diff.max())
    assert check.abs_diff.max() < 1e-3, check.sort_values("abs_diff").tail()
    order = {lab: i for bins in BINNINGS.values() for i, (_, _, lab) in enumerate(bins)}
    summary = (per_split.groupby(["condition", "binning", "channel", "metric", "degree_bin"], as_index=False)
               .agg(mean=("value", "mean"), sd=("value", "std"), n_splits=("value", "count"),
                    n_genes=("n_genes", "mean"), n_entries_per_split=("n_entries", "mean")))
    summary["bin_order"] = summary.degree_bin.map(order)
    summary = summary.sort_values(["binning", "channel", "metric", "condition", "bin_order"])
    out = ROOT / "outputs/v2/followup"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f"degree_recon_{cohort}.csv", index=False)
    per_split.to_csv(out / f"degree_recon_{cohort}_per_split.csv", index=False)
    check.to_csv(out / f"degree_recon_{cohort}_check.csv", index=False)
    print(f"finished in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
