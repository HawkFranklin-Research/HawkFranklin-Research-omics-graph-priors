"""Pre-specified paired comparison and multiple-testing utilities."""
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def corrected_resampled_ttest(diffs, n_train, n_test):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2 or not np.isfinite(diffs).all() or n_train <= 0 or n_test <= 0:
        raise ValueError("Need at least two finite differences and positive split sizes")
    mean = float(diffs.mean())
    se = float(np.sqrt((1 / len(diffs) + n_test / n_train) * diffs.var(ddof=1)))
    if se == 0:
        statistic = 0.0 if mean == 0 else np.copysign(np.inf, mean)
    else:
        statistic = mean / se
    critical = stats.t.ppf(.975, len(diffs) - 1)
    return dict(mean=mean, t=float(statistic), df=len(diffs)-1, p=float(2 * stats.t.sf(abs(statistic), len(diffs)-1)), ci_low=mean-critical*se, ci_high=mean+critical*se)


def bootstrap_ci(fn, data, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(data)
    if not len(values) or n < 1:
        raise ValueError("A nonempty sample and positive bootstrap count are required")
    estimates = []
    attempts = 0
    while len(estimates) < n and attempts < 20*n:
        attempts += 1
        try:
            estimate = fn(values[rng.integers(0, len(values), len(values))])
        except ValueError:
            continue
        if np.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) < n:
        raise ValueError(f"Only {len(estimates)} valid bootstrap draws in {attempts} attempts")
    estimates = np.asarray(estimates)
    if not np.isfinite(estimates).all():
        raise ValueError("Bootstrap statistic returned a non-finite value")
    return tuple(np.quantile(estimates, [.025, .975]))


def holm(pvals):
    return multipletests(pvals, method="holm")[1]


def bh(pvals):
    return multipletests(pvals, method="fdr_bh")[1]


def paired_compare(metric_table, reference, others, metric, n_train, n_test):
    import pandas as pd
    selected = metric_table[metric_table.metric.eq(metric)]
    ref = selected[selected.condition.eq(reference)][["repeat", "fold", "value"]]
    rows = []
    for other in others:
        values = selected[selected.condition.eq(other)][["repeat", "fold", "value"]]
        joined = ref.merge(values, on=["repeat", "fold"], suffixes=("_ref", "_other")).dropna()
        if len(joined) < 2:
            continue
        rows.append(dict(metric=metric, reference=reference, other=other, **corrected_resampled_ttest(joined.value_ref - joined.value_other, n_train, n_test)))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = holm(result.p)
    return result
