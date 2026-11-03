"""Independent classification and survival readout evaluations."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def multiclass_endpoint(embedding, labels, train_idx, test_idx):
    labels = np.asarray(labels, dtype=object)
    valid = pd.notna(labels)
    train = np.asarray(train_idx)[valid[train_idx]]
    test = np.asarray(test_idx)[valid[test_idx]]
    scaler = StandardScaler().fit(embedding[train])
    model = LogisticRegression(C=1, max_iter=5000).fit(scaler.transform(embedding[train]), labels[train])
    if set(labels[test]) != set(model.classes_):
        raise ValueError("Multiclass evaluation requires every trained class in test")
    p = model.predict_proba(scaler.transform(embedding[test]))
    from sklearn.metrics import balanced_accuracy_score
    return dict(macro_auroc=roc_auc_score(labels[test], p, labels=model.classes_, multi_class="ovr", average="macro"), balanced_accuracy=balanced_accuracy_score(labels[test], model.classes_[p.argmax(axis=1)]))


def binary_endpoint(embedding, labels, train_idx, test_idx):
    labels = np.asarray(labels, dtype=float)
    train = np.asarray(train_idx)[np.isfinite(labels[train_idx])]
    test = np.asarray(test_idx)[np.isfinite(labels[test_idx])]
    if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
        raise ValueError("Endpoint requires both classes in train and test")
    scaler = StandardScaler().fit(embedding[train])
    model = LogisticRegression(C=1, max_iter=5000).fit(scaler.transform(embedding[train]), labels[train])
    score = model.predict_proba(scaler.transform(embedding[test]))[:, 1]
    return dict(auroc=roc_auc_score(labels[test], score), auprc=average_precision_score(labels[test], score)), pd.DataFrame(dict(index=test, label=labels[test], score=score)), (scaler, model)


def survival_endpoint(features, time, event, train_idx, test_idx):
    frame = pd.DataFrame(features).copy()
    frame.columns = [f"x{i}" for i in range(frame.shape[1])]
    feature_columns = frame.columns.tolist()
    scaler = StandardScaler().fit(frame.iloc[train_idx][feature_columns])
    frame.loc[:, feature_columns] = scaler.transform(frame[feature_columns])
    frame["time"], frame["event"] = np.asarray(time), np.asarray(event)
    train = frame.iloc[train_idx].dropna()
    test = frame.iloc[test_idx].dropna()
    errors = []
    for penalty in (.1, 1., 5.):
        try:
            model = CoxPHFitter(penalizer=penalty).fit(train, duration_col="time", event_col="event")
            risk = model.predict_partial_hazard(test).to_numpy()
            harrell = concordance_index(test.time, -risk, test.event)
            from sksurv.metrics import concordance_index_ipcw
            from sksurv.util import Surv
            tau = min(5 * 365.25, .9 * train.time.max())
            ytrain = Surv.from_arrays(train.event.astype(bool), train.time)
            eligible = test.time < min(tau, train.time.max())
            ytest = Surv.from_arrays(test.loc[eligible, "event"].astype(bool), test.loc[eligible, "time"])
            uno = concordance_index_ipcw(ytrain, ytest, risk[eligible], tau=tau)[0]
            model.v2_predictions_ = pd.DataFrame(dict(index=test.index, time=test.time, event=test.event, risk=risk))
            return dict(harrell=float(harrell), uno=float(uno), penalizer=penalty, fit_failed=0, uno_n=int(eligible.sum()), uno_tau=float(tau)), model
        except Exception as exc:
            errors.append(f"penalty {penalty}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Cox fitting or scoring failed: " + "; ".join(errors))
