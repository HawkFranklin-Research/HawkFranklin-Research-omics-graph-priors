"""Build a STRING prior graph from cached, versioned reference files."""
import numpy as np
import pandas as pd
import torch
from . import GraphArtifact


def build_string_graph(links_path, info_path, genes, harmonizer, min_score=700):
    info = pd.read_csv(info_path, sep="\t", dtype=str)
    id_col = "#string_protein_id" if "#string_protein_id" in info else info.columns[0]
    mapped = harmonizer.map(info["preferred_name"].fillna(""))
    info["gene"] = mapped.approved.to_numpy()
    node = {gene: i for i, gene in enumerate(genes)}
    proteins = dict(zip(info[id_col], info["gene"].map(node)))
    pairs = {}
    for chunk in pd.read_csv(links_path, sep=r"\s+", chunksize=500000):
        chunk = chunk[chunk["combined_score"] >= min_score]
        for a, b, score in chunk[["protein1", "protein2", "combined_score"]].itertuples(index=False, name=None):
            a, b = proteins.get(a), proteins.get(b)
            if pd.isna(a) or pd.isna(b) or a == b:
                continue
            key = tuple(sorted((int(a), int(b))))
            pairs[key] = max(pairs.get(key, 0), float(score)/1000)
    if not pairs:
        raise ValueError("No STRING edges match the node universe")
    edges = sorted(pairs)
    directed = edges + [(b, a) for a, b in edges]
    degree = np.bincount(np.array(edges).ravel(), minlength=len(genes))
    return GraphArtifact(torch.tensor(directed, dtype=torch.long).T.contiguous(), torch.tensor([pairs[e] for e in edges] * 2, dtype=torch.float32), "string", diagnostics=dict(undirected_edges=len(edges), isolated=int((degree == 0).sum()), isolated_fraction=float((degree == 0).mean()), degree_quantiles=np.quantile(degree, [0,.25,.5,.75,1]).tolist()))
