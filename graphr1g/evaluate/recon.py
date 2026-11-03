"""Metrics for fixed masked-entry and masked-node reconstruction."""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score


def reconstruction_metrics(target, prediction, mask, channels, min_positives=5):
    rows = []
    for c, spec in enumerate(channels):
        selected = np.asarray(mask[..., c], bool)
        y, p = np.asarray(target[..., c])[selected], np.asarray(prediction[..., c])[selected]
        if not len(y):
            rows.append(dict(channel=spec["name"], metric="unavailable", value=None, n_entries=0, reason="no masked observed entries"))
            continue
        if spec["type"] == "binary":
            if (y == 1).sum() < min_positives or len(np.unique(y)) < 2:
                rows.append(dict(channel=spec["name"], metric="unavailable", value=None, n_entries=len(y), reason="insufficient binary classes"))
                continue
            metrics = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p)}
        else:
            mse = np.mean((p-y)**2)
            variance = np.var(y)
            metrics = {"mse": mse, "r2": 1-mse/variance if variance > 0 else None, "pearson": pearsonr(y, p).statistic if len(y) > 1 and np.std(p) > 0 and variance > 0 else None}
        rows.extend(dict(channel=spec["name"], metric=k, value=v, n_entries=len(y), reason=None if v is not None else "constant targets or predictions") for k, v in metrics.items())
    return rows
