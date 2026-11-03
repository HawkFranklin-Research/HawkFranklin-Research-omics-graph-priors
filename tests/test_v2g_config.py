import copy
import json
import numpy as np
import pandas as pd
import pytest

from graphr1g.config import load_config, validate_config


def test_missing_protocol_and_unknown_endpoints_raise():
    original = load_config("configs/v2g/brca_smoke.yaml")
    cfg = copy.deepcopy(original)
    del cfg["protocol"]["BRCA"]
    with pytest.raises(ValueError, match="Missing protocol block for BRCA"): validate_config(cfg)
    for field in ("primary_metrics", "single_gene", "attribution_endpoints"):
        cfg = copy.deepcopy(original)
        if field == "primary_metrics": cfg["protocol"]["BRCA"][field] = ["missing_auroc_gene_latent_pca"]
        elif field == "single_gene": cfg["protocol"]["BRCA"][field]["endpoint"] = "missing"
        else: cfg["protocol"]["BRCA"][field] = ["missing"]
        with pytest.raises(ValueError, match="Unknown.*endpoint"): validate_config(cfg)


def test_hnscc_equivalent_resolved_config():
    from graphr1.config import load_config as frozen
    expected = frozen("configs/v2/smoke.yaml")
    candidate = copy.deepcopy(load_config("configs/v2g/hnscc_equiv_smoke.yaml"))
    candidate.pop("protocol")
    candidate.pop("cloud")
    candidate["data"].pop("hf")
    candidate["experiment_name"] = expected["experiment_name"]
    assert candidate == expected


def test_fold_assignments_match_frozen_bytes():
    from graphr1.folds import make_folds as frozen
    from graphr1g.folds import make_folds
    patients = pd.DataFrame(dict(patient_id=[f"p{i:02}" for i in range(40)], hpv=[0, 1]*20, OS_event=[0, 0, 1, 1]*10))
    first, second = frozen(patients, 2, 2), make_folds(patients, 2, 2)
    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(second, indent=2, sort_keys=True)


def test_smoke_subsample_preserves_hnscc_joint_split_and_handles_rare_labels():
    from sklearn.model_selection import train_test_split
    from graphr1g.cube import subsample_patients
    patients = pd.DataFrame(dict(patient_id=[f"p{i}" for i in range(80)], hpv=[0., 1., np.nan, 0.]*20, OS_event=[0, 0, 1, 1]*20))
    strata = patients.hpv.fillna("NA").astype(str) + ":" + patients.OS_event.fillna("NA").astype(str)
    expected, _ = train_test_split(patients, train_size=40, random_state=42, stratify=strata)
    pd.testing.assert_frame_equal(subsample_patients(patients, 40, "hpv", "OS_event", 42), expected)
    rare = patients.assign(SUBTYPE=["BRCA_LumA"]*39+["BRCA_LumB"]*39+["BRCA_Normal", None])
    a = subsample_patients(rare, 40, "SUBTYPE", "OS_event", 42)
    b = subsample_patients(rare, 40, "SUBTYPE", "OS_event", 42)
    assert len(a) == 40
    pd.testing.assert_frame_equal(a, b)
