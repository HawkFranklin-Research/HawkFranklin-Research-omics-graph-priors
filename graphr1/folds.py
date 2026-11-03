"""Repeated patient-level stratified folds with inner validation."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import math


def make_folds(patients, n_folds=5, n_repeats=3, label_col="hpv", event_col="OS_event", validation_fraction=0.15, report=None):
    patients = patients.assign(patient_id=patients.patient_id.astype(str)).sort_values("patient_id").reset_index(drop=True)
    ids = patients["patient_id"].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("Each patient must occur exactly once")
    label = patients[label_col].fillna("NA").astype(str).to_numpy()
    event = patients[event_col].fillna("NA").astype(str).to_numpy()
    combined = np.char.add(np.char.add(label.astype(str), ":"), event.astype(str))
    report = report if report is not None else []
    def choose(indices, inner=False):
        threshold = max(n_folds, 2)
        for name, values in (("hpv×event", combined), ("hpv", label), ("event", event)):
            subset = values[indices]
            _, counts = np.unique(subset, return_counts=True)
            classes = len(counts)
            viable = counts.min() >= threshold
            if inner:
                n_val = math.ceil(len(indices) * validation_fraction)
                viable &= n_val >= classes and len(indices) - n_val >= classes
            if viable:
                report.append(dict(scope="inner" if inner else "outer", selected=name))
                return subset
            report.append(dict(scope="inner" if inner else "outer", rejected=name, counts=counts.tolist()))
        report.append(dict(scope="inner" if inner else "outer", selected="unstratified"))
        return None
    result = []
    all_idx = np.arange(len(ids))
    strata = choose(all_idx)
    for repeat in range(n_repeats):
        cv = (StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1000 + repeat)
              if strata is not None else KFold(n_splits=n_folds, shuffle=True, random_state=1000 + repeat))
        for fold, (outer_train, test) in enumerate(cv.split(all_idx, strata)):
            inner_strata = choose(outer_train, inner=True)
            train, val = train_test_split(outer_train, test_size=validation_fraction, random_state=2000 + repeat, stratify=inner_strata)
            result.append(dict(repeat=int(repeat), fold=int(fold), train_ids=sorted(ids[train].tolist()), val_ids=sorted(ids[val].tolist()), test_ids=sorted(ids[test].tolist())))
    return result


def check_folds(folds, patient_ids):
    expected = set(map(str, patient_ids))
    for repeat in sorted({f["repeat"] for f in folds}):
        test_ids = []
        for f in (f for f in folds if f["repeat"] == repeat):
            groups = [set(f[key]) for key in ("train_ids", "val_ids", "test_ids")]
            if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)) or set.union(*groups) != expected:
                raise ValueError(f"Invalid fold {repeat}/{f['fold']}")
            test_ids.extend(f["test_ids"])
        if len(test_ids) != len(expected) or set(test_ids) != expected:
            raise ValueError(f"Test folds do not partition patients in repeat {repeat}")
