"""B3 (post hoc diagnostic): why does the ESR1 basal control fail for STRING-GATv2 but pass for co-expression GATv2?

Tests the "hub dilution" explanation: in STRING >= 700 an ER-regulon gene's neighbours should be less co-expressed with
it than its neighbours in the fold's co-expression graph. For ESR1, PGR, GATA3, AR, XBP1 (ER regulon) and MKI67
(proliferation control), on BRCA fold r0_f0 (the attribution units are repeat 0):
  degree and degree percentile in STRING and in co-expression (r0_f0, then median over the 5 repeat-0 folds),
  every neighbour with its edge weight, its training-fold RNA Pearson r with the gene, and its knockout percentile
  for the basal logit in gatv2_string; summary: mean |r| and fraction |r| > 0.3 over the neighbours.
RNA correlations use only r0_f0 training patients, normalized by the saved r0_f0 normalizer (no test patients).
FOXA1 is not in the 2,000-gene node set. Writes outputs/v2/followup/esr1_diagnostic.json and esr1_neighbours.csv.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from graphr1g.normalize import ChannelNormalizer  # noqa: E402

RUN = ROOT / "outputs/v2/brca_v2_e5"
GENES = ["ESR1", "PGR", "GATA3", "AR", "XBP1", "MKI67"]
OUT = ROOT / "outputs/v2/followup"


def edges(path):
    e = pd.read_csv(path / "edges.csv")
    both = pd.concat([e, e.rename(columns={"src_gene": "dst_gene", "dst_gene": "src_gene"})])
    return both.drop_duplicates(["src_gene", "dst_gene"])


def main():
    nodes = pd.read_csv(RUN / "03_nodes/nodes.csv").gene.astype(str).tolist()
    cube = np.load(RUN / "01_cube/cube.npz")
    ids, universe = cube["patient_ids"].astype(str), cube["genes"].astype(str).tolist()
    pos = [universe.index(g) for g in nodes]
    values, observed = cube["values"][:, pos, :], cube["observed"][:, pos, :]
    fold = next(f for f in json.loads((RUN / "02_folds/folds.json").read_text()) if f["repeat"] == 0 and f["fold"] == 0)
    lookup = {p: i for i, p in enumerate(ids)}
    train = np.array([lookup[x] for x in fold["train_ids"]])
    x, m = ChannelNormalizer.load(RUN / "04_graphs/normalization/r0_f0").transform(values, observed)
    rna = pd.DataFrame(np.where(m[train, :, 0] > 0, x[train, :, 0], np.nan), columns=nodes)
    ko = pd.read_csv(RUN / "10_attribution/gatv2_string/r0_f0/knockout_basal_logit.csv")
    ko_pct = ko.set_index("gene").score.rank(pct=True)
    graphs = {"string": edges(RUN / "04_graphs/string"), "coexpr": edges(RUN / "04_graphs/coexpr/r0_f0")}
    degs = {k: g.groupby("src_gene").size().reindex(nodes, fill_value=0) for k, g in graphs.items()}
    coexpr_deg_folds = pd.concat([edges(RUN / f"04_graphs/coexpr/r0_f{f}").groupby("src_gene").size().reindex(nodes, fill_value=0)
                                  for f in range(5)], axis=1)
    rows, report = [], {"fold": "r0_f0", "n_train_patients": int(len(train)), "foxa1_in_node_set": "FOXA1" in nodes, "genes": {}}
    for gene in GENES:
        info = {}
        for name, g in graphs.items():
            nb = g[g.src_gene.eq(gene)].sort_values("weight", ascending=False)
            r = [rna[gene].corr(rna[n]) for n in nb.dst_gene]
            for (_, e), rv in zip(nb.iterrows(), r):
                rows.append(dict(gene=gene, graph=name, neighbour=e.dst_gene, edge_weight=e.weight, rna_pearson_r=rv,
                                 neighbour_string_degree=int(degs["string"][e.dst_gene]),
                                 neighbour_knockout_pct_basal_gatv2_string=ko_pct.get(e.dst_gene, np.nan)))
            r = np.asarray(r, float)
            info[name] = dict(degree=int(degs[name][gene]), degree_percentile=float((degs[name] <= degs[name][gene]).mean()),
                              n_neighbours=len(nb), mean_abs_r=float(np.nanmean(np.abs(r))) if len(r) else None,
                              frac_abs_r_gt_0_3=float(np.mean(np.abs(r) > .3)) if len(r) else None,
                              top15=[dict(gene=n, weight=float(w), r=float(c)) for n, w, c in
                                     list(zip(nb.dst_gene, nb.weight, r))[:15]])
        info["coexpr_degree_repeat0_folds"] = coexpr_deg_folds.loc[gene].astype(int).tolist()
        info["knockout_pct_basal_gatv2_string_r0_f0"] = float(ko_pct.get(gene, np.nan))
        report["genes"][gene] = info
        print(gene, {k: (v["degree"], v["mean_abs_r"]) for k, v in info.items() if isinstance(v, dict)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "esr1_neighbours.csv", index=False)
    (OUT / "esr1_diagnostic.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
