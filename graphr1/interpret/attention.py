"""Direction-aware attention statistics using PyG's returned edge order."""
import numpy as np
import pandas as pd
from graphr1.checks import check_attention


def summarize_attention(returned_indices, coefficients, genes, labels=None, threshold=1.5):
    if len(returned_indices) != len(coefficients) or not returned_indices:
        raise ValueError("Attention tensors and returned edge indices are required per patient")
    sums = {}
    for patient, (edge, alpha) in enumerate(zip(returned_indices, coefficients)):
        check_attention(edge, alpha)
        endpoints = edge.detach().cpu().numpy()
        values = alpha.detach().cpu().numpy().reshape(endpoints.shape[1], -1).mean(axis=1)
        degree = np.bincount(endpoints[1], minlength=len(genes))
        for source, target, value in zip(*endpoints, values):
            if source == target:
                continue
            key = (int(source), int(target))
            sums.setdefault(key, []).append((float(value), float(value * degree[target]), None if labels is None else labels[patient]))
    rows = []
    for (source, target), items in sums.items():
        arr = np.asarray([x[1] for x in items])
        row = dict(src_node=source, dst_node=target, src_gene=genes[source], dst_gene=genes[target], mean_alpha=np.mean([x[0] for x in items]), mean_enrichment=arr.mean(), prevalence=(arr >= threshold).mean(), n_patients=len(items))
        if labels is not None:
            positive = [x[1] for x in items if x[2] == 1]
            negative = [x[1] for x in items if x[2] == 0]
            row["group_difference"] = np.mean(positive)-np.mean(negative) if positive and negative else None
        rows.append(row)
    edges = pd.DataFrame(rows)
    genescores = edges.groupby("src_gene", as_index=False).mean(numeric_only=True).rename(columns={"src_gene": "gene", "mean_enrichment": "attention_gene_score"})
    return edges, genescores
