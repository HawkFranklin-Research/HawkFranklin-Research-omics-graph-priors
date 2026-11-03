"""Summarize and compare matched fold-level metrics."""
import pandas as pd
import numpy as np
from .stats import corrected_resampled_ttest, holm


HYPOTHESES = {"H1": ["dense_ae", "node_mlp", "pca_concat", "lowrank_impute"], "H2": ["gatv2_string_rewired"], "H3": ["gcn_string", "sage_string"], "H4": ["gatv2_coexpr"]}


def aggregate_metrics(metrics):
    return metrics.groupby(["condition", "metric"], dropna=False).value.agg(["mean", "std", "count"]).reset_index().sort_values(["metric", "mean"], ascending=[True, False])


def compare_metrics(metrics, n_train, n_test, reference="gatv2_string", exploratory=False, *, primary_metrics):
    comparisons = []
    metric_names = sorted(set(metrics.metric) - set(primary_metrics)) if exploratory else sorted(primary_metrics)
    for metric in metric_names:
        frame = metrics[metrics.metric.eq(metric)]
        ref = frame[frame.condition.eq(reference)][["repeat", "fold", "value"]]
        hypotheses = {"exploratory": sorted(set(frame.condition) - {reference})} if exploratory else HYPOTHESES
        for hypothesis, others in hypotheses.items():
            for other in others:
                matched = ref.merge(frame[frame.condition.eq(other)][["repeat", "fold", "value"]], on=["repeat", "fold"], suffixes=("_ref", "_other")).dropna()
                matched = matched[np.isfinite(matched.value_ref) & np.isfinite(matched.value_other)]
                result = corrected_resampled_ttest(matched.value_ref - matched.value_other, n_train, n_test) if len(matched) >= 2 else dict(reason="fewer than two finite paired folds")
                comparisons.append(dict(metric=metric, hypothesis=hypothesis, reference=reference, other=other, n_pairs=len(matched), **result))
    out = pd.DataFrame(comparisons).reindex(columns=["metric", "hypothesis", "reference", "other", "n_pairs", "reason", "mean", "t", "df", "p", "ci_low", "ci_high"])
    if not exploratory:
        out["p_holm"] = np.nan
        for metric, group in out[out.p.notna()].groupby("metric"):
            out.loc[group.index, "p_holm"] = holm(group.p.to_numpy())
    return out
