"""Train-fold correlation graphs matched to a reference edge count."""
import numpy as np
import torch
from . import GraphArtifact


def build_coexpression_graph(rna_train, target_edges, k_min=1, k_max=50, tolerance=.1, fold=None):
    if k_min != 1 or k_max != 50:
        raise ValueError("Revised density matching requires k in 1–50")
    x = np.asarray(rna_train, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or target_edges <= 0:
        raise ValueError("Expected patients by genes and a positive reference edge count")
    corr = np.nan_to_num(np.corrcoef(x, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 0)
    abs_corr = np.abs(corr)
    # Stable ranking excludes self even when a gene has zero correlations.
    ranking = abs_corr.copy()
    np.fill_diagonal(ranking, -np.inf)
    neighbours_order = np.argsort(-ranking, axis=1, kind="stable")
    pairs = set()
    for k in range(1, min(50, x.shape[1] - 1) + 1):
        neighbours = neighbours_order[:, :k]
        pairs = {(min(i, int(j)), max(i, int(j))) for i, row in enumerate(neighbours) for j in row if i != j}
        if len(pairs) >= target_edges:
            break
    before = len(pairs)
    drop = sorted(pairs, key=lambda pair: (abs_corr[pair], *pair))[:max(0, before-target_edges)]
    pairs.difference_update(drop)
    pairs = sorted(pairs)
    directed = pairs + [(b, a) for a, b in pairs]
    weights = [abs_corr[a, b] for a, b in directed]
    return GraphArtifact(torch.tensor(directed, dtype=torch.long).T.contiguous(), torch.tensor(weights, dtype=torch.float32), "coexpr", fold=fold, diagnostics=dict(k=k, edges_before_trim=before, edges_target=target_edges, edges_final=len(pairs), trimming_applied=before >= target_edges, shortfall=max(0, target_edges-len(pairs)), undirected_edges=len(pairs), target_edges=target_edges))
