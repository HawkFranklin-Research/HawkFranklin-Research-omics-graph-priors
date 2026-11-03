"""Label-blind gene-universe selection and coverage reporting."""
import numpy as np


def select_nodes(values, observed, genes, gene_table, hgnc, hallmark, cfg, train_idx=None):
    if cfg["selection_scope"] == "fold_train":
        if train_idx is None:
            raise ValueError("Fold-train node selection requires training indices")
        selected_values = values[train_idx]
    elif cfg["selection_scope"] == "all_patients":
        selected_values = values
    else:
        raise ValueError("Unknown node selection scope")
    coding = set(hgnc.loc[hgnc["locus_group"].eq("protein-coding gene"), "symbol"])
    membership = {}
    for symbols in hallmark.values():
        for symbol in set(symbols):
            membership[symbol] = membership.get(symbol, 0) + 1
    universe = coding if cfg["universe"] == "protein_coding" else coding & set(membership)
    table = gene_table.copy()
    table["rna_var"] = np.nanvar(selected_values[:, :, cfg.get("rna_channel_index", 0)], axis=0)
    table["hallmark_memberships"] = table.gene.map(membership).fillna(0).astype(int)
    table = table[table.gene.isin(universe)].sort_values(["rna_var", "gene"], ascending=[False, True])
    if cfg.get("top_n"):
        table = table.head(cfg["top_n"])
    table = table.reset_index(drop=True)
    table.insert(0, "node_index", np.arange(len(table)))
    if table.empty:
        raise ValueError("No nodes in the selected universe")
    return table
