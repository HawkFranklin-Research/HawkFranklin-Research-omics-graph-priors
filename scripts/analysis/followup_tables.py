"""Follow-up tables for M1 (HNSCC E1 + BRCA E5). CPU only; reads the frozen run folders and never writes to them.

Items from RESULTS_OVERVIEW.md §7:
  A1      concordance.csv                24 primary tests, HNSCC vs BRCA, protocol_brca §9 label, CI-in-margin flag
  A5      untrained_and_pca_rna.csv      gatv2_string vs untrained and vs pca_rna, every downstream endpoint (exploratory)
  B2      readout_matched_h1.csv         H1 downstream comparators under gene_latent_pca and mean_pool (exploratory)
  A4      graph_specificity.csv          control-gene knockout and grad x input percentiles, Hallmark fold counts, >=4/5 rule
          graph_specificity_folds.csv    per-fold values behind it
  A6      compute_scale.csv              fits, failures, read-outs, knockout tests, VM hours and cost

The paired test is graphr1g.evaluate.stats.corrected_resampled_ttest with the pipeline's n_train / n_test
(graphr1g/stage_evaluation.py), and the script first reproduces both comparisons.csv files exactly.
Usage: python3 scripts/analysis/followup_tables.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from graphr1g.evaluate.aggregate import HYPOTHESES  # noqa: E402
from graphr1g.evaluate.stats import corrected_resampled_ttest, holm  # noqa: E402

RUNS = {"HNSCC": ROOT / "outputs/v2/hnscc_v2_e1", "BRCA": ROOT / "outputs/v2/brca_v2_e5"}
OUT = ROOT / "outputs/v2/followup"
REF = "gatv2_string"
# E1 predates the manifest hf_revision field; value from SPENDS.md / EXPERIMENTS.md (E1 evidence)
HF_FALLBACK = {"HNSCC": "bf953018da4f918e628a5c63a4143f19ebe6e663"}
AUROC = {"HNSCC": "hpv_auroc_gene_latent_pca", "BRCA": "luma_lumb_auroc_gene_latent_pca"}
PRIMARY = ["node_rna_r2", "AUROC", "os_delta_c_gene_latent_pca"]
DOWNSTREAM = {"HNSCC": ["hpv_auroc", "os_delta_c"],
              "BRCA": ["luma_lumb_auroc", "pam50_macro_auroc", "basal_auroc", "os_delta_c", "pfs_delta_c"]}
ATTR_CONDITIONS = ["gatv2_string", "gatv2_string_untrained", "gatv2_string_rewired", "gatv2_coexpr", "gatv2_coexpr_rewired"]
CONTROLS = {  # (cohort, attribution target): control genes (first = pre-specified) and expected Hallmarks
    ("HNSCC", "hpv_logit"): (["CDKN2A"], ["HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT"]),
    ("BRCA", "basal_logit"): (["ESR1", "FOXA1", "GATA3", "PGR"], ["HALLMARK_ESTROGEN_RESPONSE_EARLY", "HALLMARK_ESTROGEN_RESPONSE_LATE"]),
    ("BRCA", "luma_lumb_logit"): (["MKI67"], ["HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT"]),
}


def margin(metric):
    return 0.01 if "delta_c" in metric else 0.02


def split_sizes(run):
    cfg = yaml.safe_load((run / "config.resolved.yaml").read_text())
    n = len(pd.read_csv(run / "01_cube/patients.csv"))
    k = cfg["cv"]["n_folds"]
    return n * (1 - 1 / k) * (1 - cfg["cv"]["validation_fraction"]), n / k


def paired(metrics, metric, a, b, sizes):
    frame = metrics[metrics.metric.eq(metric)]
    m = frame[frame.condition.eq(a)][["repeat", "fold", "value"]].merge(
        frame[frame.condition.eq(b)][["repeat", "fold", "value"]], on=["repeat", "fold"], suffixes=("_a", "_b"))
    m = m[np.isfinite(m.value_a) & np.isfinite(m.value_b)]
    if len(m) < 2:
        return dict(n_pairs=len(m), mean=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    r = corrected_resampled_ttest(m.value_a - m.value_b, *sizes)
    return dict(n_pairs=len(m), mean=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"], p=r["p"])


def label(b_mean, b_p, h_mean, h_p):
    """protocol_brca.md §9."""
    if pd.isna(b_p) or pd.isna(h_p):
        return "not estimable"
    same = np.sign(b_mean) == np.sign(h_mean)
    if b_p < .05 and not same:
        return "reversed"
    if h_p < .05:
        return "replicated" if (b_p < .05 and same) else "not replicated (same direction)" if same else "not replicated (opposite direction, n.s.)"
    return "new effect" if b_p < .05 else "consistent null"


def primary_tests(cohort, metrics, sizes):
    rows = []
    for generic in PRIMARY:
        metric = AUROC[cohort] if generic == "AUROC" else generic
        block = []
        for hyp, others in HYPOTHESES.items():
            for other in others:
                block.append(dict(cohort=cohort, metric_generic=generic, metric=metric, hypothesis=hyp, other=other,
                                  **paired(metrics, metric, REF, other, sizes)))
        block = pd.DataFrame(block)
        ok = block.p.notna()
        block["p_holm"] = np.nan
        block.loc[ok, "p_holm"] = holm(block.loc[ok, "p"].to_numpy())
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def concordance(per_run, sizes):
    tests = {c: primary_tests(c, per_run[c], sizes[c]) for c in RUNS}
    for cohort, t in tests.items():  # must reproduce the pipeline's comparisons.csv
        saved = pd.read_csv(RUNS[cohort] / "07_eval/comparisons.csv")
        chk = t.merge(saved, on=["metric", "hypothesis", "other"], suffixes=("", "_saved"))
        assert len(chk) == len(t) == 24, (cohort, len(chk))
        for col in ("mean", "p_holm"):
            diff = (chk[col] - chk[col + "_saved"]).abs().max()
            assert not diff > 1e-9, (cohort, col, diff)
        print(f"{cohort}: comparisons.csv reproduced (24 tests)")
    h, b = tests["HNSCC"], tests["BRCA"]
    keys = ["metric_generic", "hypothesis", "other"]
    m = b.drop(columns="cohort").merge(h.drop(columns="cohort"), on=keys, suffixes=("_brca", "_hnscc"))
    m["label"] = [label(*v) for v in m[["mean_brca", "p_holm_brca", "mean_hnscc", "p_holm_hnscc"]].to_numpy()]
    for c in ("brca", "hnscc"):
        mg = m["metric_" + c].map(margin)
        m[f"ci_within_margin_{c}"] = (m[f"ci_low_{c}"] > -mg) & (m[f"ci_high_{c}"] < mg)
    m["margin"] = m.metric_brca.map(margin)
    m["note"] = np.where(m.other.eq("node_mlp") & m.metric_generic.eq("node_rna_r2"),
                         "trivial by construction (node_mlp has no input for a fully masked node)", "")
    return m


def downstream_tables(per_run, sizes):
    unt, readout = [], []
    for cohort, eps in DOWNSTREAM.items():
        met = per_run[cohort]
        for ep in eps:
            metric = f"{ep}_gene_latent_pca"
            for other in ("gatv2_string_untrained", "pca_rna"):
                r = paired(met, metric, REF, other, sizes[cohort])
                unt.append(dict(cohort=cohort, metric=metric, reference=REF, other=other, margin=margin(metric), **r,
                                ci_within_margin=bool(r["ci_low"] > -margin(metric) and r["ci_high"] < margin(metric))))
            for other in HYPOTHESES["H1"]:
                for readout_name in ("gene_latent_pca", "mean_pool"):
                    metric = f"{ep}_{readout_name}"
                    ref_vals = met[met.metric.eq(metric) & met.condition.eq(REF)]
                    oth_vals = met[met.metric.eq(metric) & met.condition.eq(other)]
                    r = paired(met, metric, REF, other, sizes[cohort])
                    readout.append(dict(cohort=cohort, endpoint=ep, readout=readout_name, reference=REF, other=other,
                                        reference_mean=ref_vals.value.mean(), other_mean=oth_vals.value.mean(), **r))
    readout = pd.DataFrame(readout)
    readout["note"] = np.where(readout.other.eq("dense_ae"),
                               "dense_ae has no gene latents: both readouts are its 16-d patient bottleneck", "")
    return pd.DataFrame(unt), readout


def graph_specificity():
    folds, summary = [], []
    for (cohort, target), (genes, terms) in CONTROLS.items():
        run = RUNS[cohort]
        for cond in ATTR_CONDITIONS:
            for f in range(5):
                d = run / f"10_attribution/{cond}/r0_f{f}"
                for method in ("knockout", "gradxinput"):
                    s = pd.read_csv(d / f"{method}_{target}.csv")
                    s["pct"] = s.score.rank(pct=True)  # same call as graphr1g/stage_interpret.py
                    for g in genes:
                        hit = s[s.gene.eq(g)]
                        folds.append(dict(cohort=cohort, target=target, condition=cond, fold=f, kind=f"{method}_percentile",
                                          feature=g, value=hit.pct.iloc[0] if len(hit) else np.nan,
                                          pass_threshold=None))
                gsea = pd.read_csv(run / f"11_enrichment/{cond}/r0_f{f}/knockout_{target}/gsea_results.csv")
                for t in terms:
                    hit = gsea[gsea.Term.eq(t)]
                    nes = hit.NES.iloc[0] if len(hit) else np.nan
                    fdr = hit["FDR q-val"].iloc[0] if len(hit) else np.nan
                    folds.append(dict(cohort=cohort, target=target, condition=cond, fold=f, kind="hallmark_nes", feature=t,
                                      value=nes, pass_threshold=bool(nes > 0 and fdr < .25), fdr=fdr))
    folds = pd.DataFrame(folds)
    for (cohort, target, kind, feature), g in folds.groupby(["cohort", "target", "kind", "feature"]):
        w = g.pivot(index="fold", columns="condition", values="value")
        spec = int(((w[REF] > w["gatv2_string_untrained"]) & (w[REF] > w["gatv2_string_rewired"])).sum())
        for cond in ATTR_CONDITIONS:
            x = g[g.condition.eq(cond)]
            row = dict(cohort=cohort, target=target, kind=kind, feature=feature, condition=cond,
                       mean=x.value.mean(), min=x.value.min(), max=x.value.max(), n_folds=len(x),
                       string_beats_untrained_and_rewired_folds=spec, graph_dependent=spec >= 4)
            if kind == "hallmark_nes":
                row["folds_fdr_lt_0.25_positive"] = int(x.pass_threshold.sum())
            summary.append(row)
    return pd.DataFrame(summary), folds


def compute_scale():
    spends = (ROOT / "SPENDS.md").read_text()
    vm = {"HNSCC": re.search(r"E1–E3 Primary Grid.*?\|\s*([\d.]+) h\s*\|\s*\*\*\$([\d.]+)\*\*", spends),
          "BRCA": re.search(r"E5 BRCA Replication Grid.*?\|\s*([\d.]+) h\s*\|\s*\*\*\$([\d.]+)\*\*", spends)}
    rows = []
    for cohort, run in RUNS.items():
        cfg = yaml.safe_load((run / "config.resolved.yaml").read_text())
        status = lambda p: json.loads(p.read_text()).get("status")  # noqa: E731
        neural = [status(p) for p in run.glob("05_runs/*/r*_f*/status.json") if "_eval_masks" not in str(p)]
        base = [status(p) for p in run.glob("06_baselines/*/r*_f*/status.json")]
        genes = len(pd.read_csv(run / "03_nodes/nodes.csv"))
        units = sorted(run.glob("10_attribution/*/r0_f*"))
        targets = len([t for t in (units[0]).glob("knockout_*.csv")])
        ppf = cfg["interpret"]["patients_per_fold"]
        rows.append(dict(cohort=cohort, patients=len(pd.read_csv(run / "01_cube/patients.csv")), genes=genes,
                         neural_fits_planned=len(neural), neural_fits_done=neural.count("done"),
                         neural_fits_failed=len(pd.read_csv(run / "07_eval/missing_runs.csv")),
                         baseline_fits_done=base.count("done"), baseline_fits_total=len(base),
                         attention_readouts=len(list(run.glob("09_attention/*/r*_f*/edge_stats_L1.parquet"))),
                         attribution_units=len(units), attribution_targets=targets, patients_per_fold=ppf,
                         knockout_forward_tests=len(units) * ppf * genes * targets,
                         gsea_runs=len(list(run.glob("11_enrichment/*/*/*/gsea_results.csv"))),
                         vm_hours=float(vm[cohort].group(1)), cloud_cost_usd=float(vm[cohort].group(2)),
                         git_commit=json.loads((run / "run_manifest.json").read_text()).get("git_commit", "")[:7],
                         hf_revision=json.loads((run / "run_manifest.json").read_text()).get("hf_revision")
                         or HF_FALLBACK.get(cohort, "")))
    out = pd.DataFrame(rows)
    total = out.select_dtypes("number").sum()
    total[["genes", "patients_per_fold", "attribution_targets"]] = np.nan
    return pd.concat([out, pd.DataFrame([dict(cohort="TOTAL", **total.to_dict())])], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_run = {c: pd.read_csv(r / "07_eval/per_run_metrics.csv") for c, r in RUNS.items()}
    sizes = {c: split_sizes(r) for c, r in RUNS.items()}
    conc = concordance(per_run, sizes)
    conc.to_csv(OUT / "concordance.csv", index=False)
    print(conc.label.value_counts().to_string())
    unt, readout = downstream_tables(per_run, sizes)
    unt.to_csv(OUT / "untrained_and_pca_rna.csv", index=False)
    readout.to_csv(OUT / "readout_matched_h1.csv", index=False)
    spec, spec_folds = graph_specificity()
    spec.to_csv(OUT / "graph_specificity.csv", index=False)
    spec_folds.to_csv(OUT / "graph_specificity_folds.csv", index=False)
    compute_scale().to_csv(OUT / "compute_scale.csv", index=False)
    print("wrote", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
