"""Fail-fast checks for core artifact invariants."""
import numpy as np
import logging


def check_cube(values, observed, directory=None, patients=None, cohorts=None):
    if np.any(~np.isfinite(values[np.asarray(observed, dtype=bool)])):
        raise ValueError("Non-finite observed cube value")
    if directory is None: return
    import json
    import pandas as pd
    if not (directory / "harmonization_map.csv").is_file():
        raise ValueError("Missing harmonization map")
    coverage = pd.read_csv(directory / "qc_coverage.csv").set_index("channel")
    for channel in ("cnv", "mut"):
        if channel not in coverage.index or coverage.loc[channel, "rna_gene_overlap_after"] < .85:
            raise ValueError(f"Gene harmonization coverage gate failed for {channel}")
    if json.loads((directory / "patient_join.json").read_text())["join_rate"] < .98:
        raise ValueError("Clinical patient join rate below 98%")
    if {str(c).lower() for c in cohorts} != {"tcga"} or not patients.cohort.str.lower().eq("tcga").all():
        raise ValueError("V2 primary analysis requires a TCGA-only cohort")


def check_graph(graph, n_nodes):
    edges = graph.edge_index.cpu().numpy().T
    pairs = [tuple(pair) for pair in edges]
    if len(set(pairs)) != len(pairs) or any(a == b or a < 0 or b < 0 or a >= n_nodes or b >= n_nodes for a, b in pairs):
        raise ValueError("Graph contains duplicate, self, or out-of-range edges")
    pair_set = set(pairs)
    if any((b, a) not in pair_set for a, b in pairs):
        raise ValueError("Graph is not symmetric")
    diagnostics = graph.diagnostics or {}
    if "edges_target" in diagnostics:
        target, final = diagnostics["edges_target"], diagnostics["edges_final"]
        if final != len(pairs) // 2:
            raise ValueError("Graph edge diagnostics disagree with artifact")
        if final < target:
            logging.getLogger(__name__).warning("Coexpression density shortfall: %s / %s edges", final, target)
        elif abs(final-target) > .1 * target:
            raise ValueError("Coexpression density exceeds target tolerance")


def check_attention(edge_index, alpha, tolerance=1e-4):
    target = edge_index[1].detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy().reshape(len(target), -1)
    sums = np.zeros((target.max() + 1, alpha.shape[1]))
    np.add.at(sums, target, alpha)
    active = np.bincount(target, minlength=len(sums)) > 0
    if not np.allclose(sums[active], 1, atol=tolerance):
        raise ValueError("Attention does not sum to one per target")


def check_node_coverage(table, channel_thresholds=None):
    channel_thresholds = channel_thresholds or {"rna": .95, "cnv": .90, "mut": .90}
    for name, required in channel_thresholds.items():
        col = name + "_coverage"
        if col not in table or (table[col] >= .9).mean() < required:
            raise ValueError(f"Node coverage gate failed: {name}")


def check_attribution(scores, genes):
    if len(scores) != len(genes) or set(scores.gene) != set(genes):
        raise ValueError("Attribution must contain every gene exactly once")
    if not np.isfinite(scores.score).all() or np.allclose(scores.score, 0):
        raise ValueError("Attribution scores are nonfinite or all zero")


def check_training(history, trained=True):
    for column in ("train_loss", "val_loss_EM", "val_loss_NM"):
        if column not in history or not np.isfinite(history[column]).all():
            raise ValueError(f"Invalid training history: {column}")
    if history.empty or (trained and not history.val_loss_EM.iloc[-1] < history.val_loss_EM.iloc[0]):
        raise ValueError("Final val_loss_EM must improve over first val_loss_EM")


def check_normalization(normalizer, x, observed, train_idx, test_idx, ids):
    import hashlib
    train_ids = sorted(str(ids[i]) for i in train_idx)
    assert not set(train_ids) & {str(ids[i]) for i in test_idx}
    assert normalizer.train_patient_hash == hashlib.sha1("\n".join(train_ids).encode()).hexdigest()
    for c, kind in enumerate(normalizer.channel_types):
        if kind != "continuous": continue
        # Constant genes have zero variance by definition and cannot be standardized.
        active = normalizer.std[:, c] > normalizer.sd_floor
        mask = observed[train_idx, :, c].astype(bool) & active
        values = x[train_idx, :, c][mask]
        if not len(values): continue
        assert abs(values.mean()) < .05, (c, values.mean())
        assert .9 < values.std() < 1.1, (c, values.std())


def check_embeddings(array):
    assert np.isfinite(array).all()
    assert (array.std(axis=0) > 1e-6).sum() >= min(8, array.shape[1])


def check_evaluation(table):
    auc = table[table.metric.str.contains("auroc")].value.dropna()
    assert auc.between(0, 1).all()
    cindex = table[table.metric.str.contains("_harrell_|_uno_")].value.dropna()
    assert ((cindex > 0) & (cindex < 1)).all()
    assert table.groupby(["metric", "condition"]).value.mean().groupby("metric").nunique().max() > 1
