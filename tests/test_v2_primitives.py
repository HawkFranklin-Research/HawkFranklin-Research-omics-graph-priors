import numpy as np
import pandas as pd
import torch

from graphr1.evaluate.stats import corrected_resampled_ttest
from graphr1.graphs import GraphArtifact
from graphr1.graphs.rewire import rewire_graph
from graphr1.masking import fixed_mask, hide_entries
from graphr1.normalize import ChannelNormalizer
from graphr1.symbols import HGNCHarmonizer


def test_density_trimming_exact_and_deterministic():
    from graphr1.graphs.coexpr import build_coexpression_graph
    x = np.tile(np.arange(4, dtype=float)[:, None], (1, 4))
    first = build_coexpression_graph(x, 2)
    second = build_coexpression_graph(x, 2)
    assert first.diagnostics["k"] == 1
    assert first.diagnostics["edges_before_trim"] == 3
    assert first.diagnostics["edges_target"] == first.diagnostics["edges_final"] == 2
    assert torch.equal(first.edge_index, second.edge_index)
    assert first.edge_index.T.tolist() == [[0, 2], [0, 3], [2, 0], [3, 0]]


def test_symbols_priority_and_ambiguity():
    reference = pd.DataFrame([
        dict(symbol="A", hgnc_id="1", status="Approved", prev_symbol="OLD", alias_symbol="SHARED|ALIAS_A"),
        dict(symbol="B", hgnc_id="2", status="Approved", prev_symbol="A", alias_symbol="SHARED"),
    ])
    result = HGNCHarmonizer(reference).map(["a", "old", "shared", "missing", "alias_a"])
    assert result.method.tolist() == ["approved", "previous", "ambiguous", "unmapped", "alias"]
    assert result.approved.iloc[0] == "A"
    assert pd.isna(result.approved.iloc[2])
    from graphr1.symbols import source_symbols
    sources = source_symbols(["GENE:a; B"])
    assert sources == [("GENE:a; B", "a"), ("GENE:a; B", "B")]
    assert HGNCHarmonizer(reference).map([s for _, s in sources]).approved.tolist() == ["A", "B"]


def test_normalizer_excludes_test_patients():
    values = np.array([[[1.]], [[3.]], [[1000.]]])
    observed = np.ones_like(values)
    normalizer = ChannelNormalizer(["continuous"]).fit(values, observed, [0, 1], ["a", "b", "c"])
    assert normalizer.mean[0, 0] == 2
    assert normalizer.std[0, 0] == 1


def test_masks_are_seeded_and_hide_observation_indicators():
    obs = np.ones((4, 5, 2), dtype=bool)
    mask = fixed_mask(obs, 0.5, 42)
    assert np.array_equal(mask, fixed_mask(obs, 0.5, 42))
    x, remaining = hide_entries(np.ones(obs.shape), obs, mask)
    assert np.all(x[mask] == 0) and np.all(remaining[mask] == 0)


def test_corrected_variance():
    result = corrected_resampled_ttest([1, 2, 3, 4], 80, 20)
    expected = 2.5 / np.sqrt((1 / 4 + 20 / 80) * np.var([1, 2, 3, 4], ddof=1))
    assert np.isclose(result["t"], expected)


def test_rewire_preserves_degree():
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    directed = pairs + [(b, a) for a, b in pairs]
    graph = GraphArtifact(torch.tensor(directed).T, torch.ones(len(directed)), "toy")
    result = rewire_graph(graph, 6, seed=4, swaps_per_edge=3)
    assert torch.equal(torch.bincount(graph.edge_index[0], minlength=6), torch.bincount(result.edge_index[0], minlength=6))


def test_singleton_stratum_fallback_is_deterministic():
    from graphr1.folds import make_folds, check_folds
    patients = pd.DataFrame(dict(patient_id=[str(i) for i in range(24)], hpv=[1]*8+[0]*15+[np.nan], OS_event=[0,1]*12))
    report = []
    first = make_folds(patients, 2, 1, report=report)
    assert first == make_folds(patients, 2, 1)
    check_folds(first, patients.patient_id)
    assert any(row.get("rejected") == "hpv×event" for row in report)


def test_cindex_risk_sign():
    from sksurv.metrics import concordance_index_censored
    times = np.arange(1., 11.)
    assert concordance_index_censored(np.ones(10, dtype=bool), times, -times)[0] == 1.


def test_survival_endpoint_uses_negative_risk_for_lifelines(monkeypatch):
    from graphr1.evaluate import downstream
    class FixedRisk:
        def __init__(self, penalizer): pass
        def fit(self, frame, duration_col, event_col): return self
        def predict_partial_hazard(self, frame): return 1 / frame.time
    monkeypatch.setattr(downstream, "CoxPHFitter", FixedRisk)
    times = np.r_[np.arange(1., 9.), [1.5, 2.5, 3.5, 4.5]]
    result, _ = downstream.survival_endpoint(times[:, None], times, np.ones(12), np.arange(8), np.arange(8, 12))
    assert result["harrell"] == result["uno"] == 1.


def test_masks_shared_across_conditions_and_canonical_archive(tmp_path):
    import json
    from graphr1.runtime import save_eval_masks, sha256_file
    from graphr1.paths import ExperimentPaths
    paths = ExperimentPaths(tmp_path, "toy")
    observed = np.ones((3, 4, 2), dtype=np.uint8)
    observed[0, 0, 0] = 0
    masks = {mode: fixed_mask(observed, .5, 3000 + (mode == "node"), mode) for mode in ("entry", "node")}
    file = paths.eval_mask(0, 0)
    file.parent.mkdir(parents=True)
    save_eval_masks(file, ["a", "b", "c"], masks)
    original = file.read_bytes()
    save_eval_masks(file, ["a", "b", "c"], dict(reversed(list(masks.items()))))
    assert file.read_bytes() == original
    assert json.loads(file.with_suffix(".status.json").read_text())["sha256"] == sha256_file(file)
    # Every architecture and baseline consumes this single fold archive.
    loaded = {condition: np.load(paths.eval_mask(0, 0))["entry"] for condition in ("gatv2_string", "gcn_string", "node_mlp", "pca_concat")}
    for mask in loaded.values():
        assert mask.dtype == np.bool_
        assert np.array_equal(mask, masks["entry"])
        assert not mask[0, 0, 0]


def test_folds_ignore_patient_row_order():
    from graphr1.folds import make_folds
    patients = pd.DataFrame(dict(patient_id=[f"p{i:02}" for i in range(40)], hpv=[0, 1]*20, OS_event=[0, 0, 1, 1]*10))
    assert make_folds(patients, 2, 2) == make_folds(patients.iloc[::-1], 2, 2)


def test_density_shortfall_warns_without_failing(caplog):
    from graphr1.graphs.coexpr import build_coexpression_graph
    from graphr1.checks import check_graph
    graph = build_coexpression_graph(np.tile(np.arange(4.)[:, None], (1, 4)), 7)
    assert graph.diagnostics["k"] == 3  # All available neighbours, capped at G-1.
    assert graph.diagnostics["shortfall"] == 1
    assert graph.diagnostics["edges_final"] == 6
    check_graph(graph, 4)
    assert "shortfall" in caplog.text


def test_comparisons_pair_by_fold_and_separate_exploratory():
    from graphr1.evaluate.aggregate import compare_metrics
    from graphr1.evaluate.stats import holm
    rows = [dict(condition=condition, repeat=0, fold=fold, metric=metric, value=value)
            for metric in ("node_rna_r2", "entry_meth_mse")
            for condition, values in (("gatv2_string", [1., 2., 3.]), ("gcn_string", [.4, 1.7, 2.1]), ("pca_concat", [.2, 1.1, 2.4]))
            for fold, value in enumerate(values)]
    frame = pd.DataFrame(rows).iloc[::-1]
    primary = compare_metrics(frame, 80, 20)
    exploratory = compare_metrics(frame, 80, 20, exploratory=True)
    assert set(primary.metric) == {"node_rna_r2", "hpv_auroc_gene_latent_pca", "os_delta_c_gene_latent_pca"}
    assert set(primary.hypothesis) == {"H1", "H2", "H3", "H4"}
    tested = primary[primary.p.notna()]
    assert np.allclose(tested.p_holm, holm(tested.p.to_numpy()))
    assert primary[primary.n_pairs.lt(2)].reason.notna().all()
    assert set(exploratory.metric) == {"entry_meth_mse"}
    assert "p_holm" not in exploratory
    assert set(exploratory.other) == {"gcn_string", "pca_concat"}
    expected = corrected_resampled_ttest(np.array([1., 2., 3.]) - [.4, 1.7, 2.1], 80, 20)
    assert np.isclose(exploratory.loc[exploratory.other.eq("gcn_string"), "t"].iloc[0], expected["t"])


def test_training_improvement_skips_only_untrained():
    import pytest
    from graphr1.checks import check_training
    history = pd.DataFrame(dict(train_loss=[1., 1.], val_loss_EM=[1., 1.], val_loss_NM=[2., 2.]))
    check_training(history, trained=False)
    with pytest.raises(ValueError, match="improve"):
        check_training(history)
