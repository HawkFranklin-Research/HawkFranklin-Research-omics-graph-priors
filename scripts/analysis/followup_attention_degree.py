"""A3 (protocol_brca §11-12.5, exploratory): is gene/edge attention stability a degree effect? HNSCC and BRCA.

Extends manuscript/figure_code/analysis_attention_alpha.py (HNSCC) to both cohorts and adds degree residualization:
  scores     degree_weighted = mean_enrichment (alpha * d_in, as the pipeline), raw_alpha = mean_alpha,
             residualized    = gene score (degree_weighted) minus its OLS fit on log(1 + own-graph degree)
  within     all 105 run pairs of one arm (as the pipeline's gate), gene RBO / Jaccard@100, edge RBO / Jaccard@100
  cross      same split (r, f), gatv2_string vs gatv2_string_rewired and vs gatv2_string_untrained, gene RBO
  spearman   gene score vs STRING degree and vs own-graph degree, per run
The degree_weighted within-arm pairs must reproduce 09_attention/stability.csv.
Writes outputs/v2/followup/attention_degree_{cohort}.csv (summary) and _pairs.csv, _spearman.csv.
Usage: python3 scripts/analysis/followup_attention_degree.py {hnscc|brca}
"""
import importlib
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

COHORTS = {"hnscc": ("graphr1", ROOT / "outputs/v2/hnscc_v2_e1"), "brca": ("graphr1g", ROOT / "outputs/v2/brca_v2_e5")}
ARMS = ["gatv2_string", "gatv2_string_untrained", "gatv2_string_rewired", "gatv2_coexpr", "gatv2_coexpr_rewired"]
OWN_GRAPH = {"gatv2_string": "string", "gatv2_string_untrained": "string", "gatv2_string_rewired": "string_rewired",
             "gatv2_coexpr": "coexpr", "gatv2_coexpr_rewired": "coexpr_rewired"}
CROSS = [("gatv2_string", "gatv2_string_rewired"), ("gatv2_string", "gatv2_string_untrained")]


def main(cohort):
    pkg, run = COHORTS[cohort]
    load_graph = importlib.import_module(f"{pkg}.graphs.io").load_graph
    stab = importlib.import_module(f"{pkg}.interpret.stability")
    genes = pd.read_csv(run / "03_nodes/nodes.csv").gene.astype(str).tolist()

    def degree(path):
        g = load_graph(path)
        return pd.Series(np.bincount(g.edge_index[0].numpy(), minlength=len(genes)), index=genes)

    string_deg = degree(run / "04_graphs/string")
    own_deg = {}

    def own(arm, r, f):
        key = (OWN_GRAPH[arm], r, f)
        if key not in own_deg:
            p = run / f"04_graphs/{OWN_GRAPH[arm]}"
            own_deg[key] = degree(p / f"r{r}_f{f}" if (p / f"r{r}_f{f}").exists() else p)
        return own_deg[key]

    def gene_scores(frame, arm, r, f):
        weighted = frame.groupby("src_gene").mean_enrichment.mean()  # pipeline: mean over outgoing edges
        raw = frame.groupby("src_gene").mean_alpha.mean()
        d = np.log1p(own(arm, r, f).loc[weighted.index].to_numpy(float))
        X = np.column_stack([np.ones_like(d), d])
        beta, *_ = np.linalg.lstsq(X, weighted.to_numpy(), rcond=None)
        resid = pd.Series(weighted.to_numpy() - X @ beta, index=weighted.index)
        return {"degree_weighted": weighted, "raw_alpha": raw, "residualized": resid}

    def ranking(s):
        return s.sort_values(ascending=False).index.tolist()

    def edges_ranked(frame, column):
        o = frame.sort_values(column, ascending=False)
        return list(zip(o.src_gene, o.dst_gene))

    edges, scores = {}, {}
    for arm, layer in itertools.product(ARMS, (1, 2)):
        for r, f in itertools.product(range(3), range(5)):
            e = pd.read_parquet(run / f"09_attention/{arm}/r{r}_f{f}/edge_stats_L{layer}.parquet")
            edges[(arm, layer, r, f)] = e
            scores[(arm, layer, r, f)] = gene_scores(e, arm, r, f)
        print(cohort, arm, layer, flush=True)

    pairs, spear = [], []
    for arm, layer in itertools.product(ARMS, (1, 2)):
        for (ra, fa), (rb, fb) in itertools.combinations(itertools.product(range(3), range(5)), 2):  # all 105 run pairs, as the gate
                a, b = (arm, layer, ra, fa), (arm, layer, rb, fb)
                for label in ("degree_weighted", "raw_alpha", "residualized"):
                    ga, gb = ranking(scores[a][label]), ranking(scores[b][label])
                    row = dict(comparison="within_arm", condition=arm, layer=layer, run_a=f"r{ra}_f{fa}", run_b=f"r{rb}_f{fb}",
                               score=label, gene_rbo=stab.rbo(ga, gb), gene_jaccard_100=stab.jaccard_at_k(ga, gb, 100))
                    if label != "residualized":
                        col = "mean_enrichment" if label == "degree_weighted" else "mean_alpha"
                        ea, eb = edges_ranked(edges[a], col), edges_ranked(edges[b], col)
                        row.update(edge_rbo=stab.rbo(ea, eb), edge_jaccard_100=stab.jaccard_at_k(ea, eb, 100))
                    pairs.append(row)
        for r, f in itertools.product(range(3), range(5)):
            for label, s in scores[(arm, layer, r, f)].items():
                for gname, deg in (("string", string_deg), ("own_graph", own(arm, r, f))):
                    spear.append(dict(condition=arm, layer=layer, repeat=r, fold=f, score=label, degree_graph=gname,
                                      spearman_rho=spearmanr(s.values, deg.loc[s.index].values).statistic))
    for (x, y), layer in itertools.product(CROSS, (1, 2)):
        for r, f in itertools.product(range(3), range(5)):
            for label in ("degree_weighted", "raw_alpha", "residualized"):
                ga, gb = ranking(scores[(x, layer, r, f)][label]), ranking(scores[(y, layer, r, f)][label])
                pairs.append(dict(comparison="cross_arm", condition=f"{x}_vs_{y}", layer=layer, run_a=f"r{r}_f{f}",
                                  run_b=f"r{r}_f{f}", score=label, gene_rbo=stab.rbo(ga, gb),
                                  gene_jaccard_100=stab.jaccard_at_k(ga, gb, 100)))
    pairs, spear = pd.DataFrame(pairs), pd.DataFrame(spear)

    pipe = pd.read_csv(run / "09_attention/stability.csv")
    mine = pairs[(pairs.score == "degree_weighted")]
    merged = mine.merge(pipe, on=["condition", "layer", "run_a", "run_b"], suffixes=("", "_pipe"))
    assert len(merged) == len(mine), f"{len(merged)} of {len(mine)} pairs found in stability.csv"
    for col in ("gene_rbo",) + (("edge_rbo",) if "edge_rbo_pipe" in merged else ()):
        diff = (merged[col] - merged[col + "_pipe"]).abs().max()
        print(f"max |ours - pipeline| {col}: {diff:.2e}")
        assert diff < 1e-6, col

    summary = []
    for keys, g in pairs.groupby(["comparison", "condition", "layer", "score"]):
        for metric in ("gene_rbo", "gene_jaccard_100", "edge_rbo", "edge_jaccard_100"):
            if metric in g and g[metric].notna().any():
                summary.append(dict(analysis=keys[0], condition=keys[1], layer=keys[2], score=keys[3], degree_graph="",
                                    metric=metric, median=g[metric].median(), q25=g[metric].quantile(.25),
                                    q75=g[metric].quantile(.75), n=int(g[metric].notna().sum())))
    for keys, g in spear.groupby(["condition", "layer", "score", "degree_graph"]):
        summary.append(dict(analysis="gene_score_vs_degree_spearman", condition=keys[0], layer=keys[1], score=keys[2],
                            degree_graph=keys[3], metric="spearman_rho", median=g.spearman_rho.median(),
                            q25=g.spearman_rho.quantile(.25), q75=g.spearman_rho.quantile(.75), n=len(g)))
    out = ROOT / "outputs/v2/followup"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / f"attention_degree_{cohort}.csv", index=False)
    pairs.to_csv(out / f"attention_degree_{cohort}_pairs.csv", index=False)
    spear.to_csv(out / f"attention_degree_{cohort}_spearman.csv", index=False)
    print("done", cohort)


if __name__ == "__main__":
    main(sys.argv[1])
