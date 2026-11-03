from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from graphr1g.clinical import build_patient_table, encode_covariates
from graphr1g.config import load_config
from graphr1g.protocol import cancer_protocol, check_fold_counts, check_single_gene, single_gene_name, baseline_names


def test_fixed_categorical_covariates_and_unknowns():
    frame = pd.DataFrame({"SUBTYPE": ["BRCA_LumA", "BRCA_LumB", "BRCA_Basal", "BRCA_Her2", "BRCA_Normal", None, "unrecognized"]})
    cfg = load_config("configs/v2g/brca_smoke.yaml")
    spec = cfg["endpoints"]["BRCA"]["clinical_covariates"][-1]
    encoded = encode_covariates(frame, [spec], [0, 1])
    assert encoded.columns.tolist() == ["pam50_BRCA_LumB", "pam50_BRCA_Basal", "pam50_BRCA_Her2", "pam50_BRCA_Normal", "pam50_unknown"]
    assert encoded.iloc[0].sum() == 0  # Reference dropped.
    assert encoded.iloc[2].tolist() == [0, 1, 0, 0, 0]  # Category absent from train still has its fixed column.
    assert encoded.pam50_unknown.tolist() == [0, 0, 0, 0, 0, 1, 1]


def test_hpv_covariates_match_frozen():
    from graphr1.clinical import encode_covariates as frozen
    frame = pd.DataFrame(dict(age=[30., 50., 70.], stage_num=[1., 2., np.nan], hpv=[0., 1., np.nan]))
    cols = ["age", "stage_num", "hpv"]
    pd.testing.assert_frame_equal(encode_covariates(frame, cols, [0, 1]), frozen(frame, cols, [0, 1]), check_exact=True)


def test_fold_count_checks_strict_relaxed_and_failure():
    cfg = load_config("configs/v2g/brca_smoke.yaml")
    checks = cancer_protocol(cfg)["fold_checks"]
    full = pd.DataFrame(dict(SUBTYPE=["BRCA_LumB"]*10+["BRCA_LumA"]*10+["BRCA_Basal"]*10, OS_event=[1]*10+[0]*20))
    check_fold_counts(full, checks)
    small = pd.DataFrame(dict(SUBTYPE=["BRCA_LumB", "BRCA_LumA", "BRCA_Basal"]*2, OS_event=[1, 1, 0, 0, 0, 0]))
    check_fold_counts(small, checks, relaxed=True)
    with pytest.raises(ValueError, match="Fold check failed"):
        check_fold_counts(small, checks)
    with pytest.raises(ValueError, match="OS_event"):
        check_fold_counts(small.assign(OS_event=0), checks, relaxed=True)
    hn = cancer_protocol(load_config("configs/v2g/base.yaml"))
    check_fold_counts(pd.DataFrame(dict(hpv=[1, 0], OS_event=[1, 1])), hn["fold_checks"], relaxed=True)


@pytest.mark.parametrize("config,gene,endpoint,minimum", [
    ("configs/v2g/base.yaml", "CDKN2A", "hpv", .75),
    ("configs/v2g/brca_smoke.yaml", "ESR1", "basal", .8),
])
def test_single_gene_names_and_gate(config, gene, endpoint, minimum):
    cfg = load_config(config)
    protocol = cancer_protocol(cfg)
    name = "single_gene_" + gene.lower()
    assert single_gene_name(protocol) == name
    assert baseline_names(cfg)[-1] == name
    frame = pd.DataFrame([dict(condition=name, metric=endpoint+"_auroc_gene_latent_pca", value=minimum)])
    check_single_gene(frame, protocol)
    with pytest.raises(ValueError, match="HARD BLOCKER 3"):
        check_single_gene(frame.assign(value=minimum-.01), protocol)
    with pytest.raises(ValueError, match=gene):
        check_single_gene(frame.assign(metric="wrong_auroc_gene_latent_pca"), protocol)


def test_brca_endpoint_mapping_keeps_unlabelled_patients(monkeypatch, tmp_path):
    cfg = load_config("configs/v2g/brca_smoke.yaml")
    ids = [f"p{i}" for i in range(6)]
    master = pd.DataFrame(dict(patient_id=ids, cancer="BRCA", AGE=50, SEX="FEMALE", AJCC_PATHOLOGIC_TUMOR_STAGE="Stage II",
                               SUBTYPE=["BRCA_LumA", "BRCA_LumB", "BRCA_Basal", "BRCA_Her2", "BRCA_Normal", None]))
    targets = pd.DataFrame(dict(patient_id=ids, OS_days=100, OS_event=1, PFS_days=50, PFS_event=1))
    tables = {"patient_master.parquet": master, "targets.parquet": targets}
    monkeypatch.setattr(pd, "read_parquet", lambda path: tables[Path(path).name].copy())
    samples = pd.DataFrame(dict(patient_id=ids, cohort="tcga", cancer="BRCA"))
    result = build_patient_table(tmp_path, "BRCA", samples, cfg["endpoints"]["BRCA"])
    assert result.patient_id.tolist() == ids
    assert result.luma_lumb.iloc[:2].tolist() == [0, 1]
    assert result.luma_lumb.iloc[2:].isna().all()
    assert result.basal.iloc[:4].tolist() == [0, 0, 1, 0]
    assert result.basal.iloc[4:].isna().all()
    assert result.pam50.iloc[4:].isna().all()
    assert result.OS_event.notna().all()
