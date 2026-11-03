import json
import pandas as pd
import pytest
from scripts.compare_v2_runs import compare_runs, compare_metric_rows


def tiny_output(root, baseline=.5, model=.3):
    for path in ("02_folds", "03_nodes", "05_runs/_eval_masks", "06_baselines/pca_concat", "07_eval"):
        (root/path).mkdir(parents=True, exist_ok=True)
    (root/"02_folds/folds.json").write_text("[]\n")
    (root/"03_nodes/nodes.csv").write_text("gene\nA\n")
    (root/"05_runs/_eval_masks/r0_f0.npz").write_bytes(b"same mask archive fixture")
    (root/"run_manifest.json").write_text(json.dumps({"git_commit": "fixture"}))
    pd.DataFrame([dict(condition=name, repeat=0, fold=0, metric="node_rna_r2", value=value)
                  for name, value in (("pca_concat", baseline), ("gatv2_string", model))]).to_csv(root/"07_eval/per_run_metrics.csv", index=False)


@pytest.mark.parametrize("baseline,model,passes", [(.5, .3, True), (.50000001, .3, False), (.5, .30005, True), (.5, .3002, False)])
def test_compare_values(tmp_path, baseline, model, passes):
    reference, candidate = tmp_path/"ref", tmp_path/"new"
    tiny_output(reference)
    tiny_output(candidate, baseline, model)
    rows = compare_runs(reference, candidate)
    assert all(row["status"] == "PASS" for row in rows) == passes


def test_compare_missing_files_and_rows_fail(tmp_path):
    reference, candidate = tmp_path/"ref", tmp_path/"new"
    tiny_output(reference)
    tiny_output(candidate)
    (candidate/"03_nodes/nodes.csv").unlink()
    rows = compare_runs(reference, candidate)
    assert any(row["status"] == "FAIL" for row in rows)


def test_compare_expected_additions(tmp_path):
    reference, candidate = tmp_path/"ref", tmp_path/"new"
    tiny_output(reference)
    tiny_output(candidate)
    target = candidate/"10_attribution/gatv2_string/r0_f0/cdkn2a_rank_percentile_hpv.csv"
    target.parent.mkdir(parents=True)
    target.write_text("gene,score\nCDKN2A,1\n")
    (candidate/"protocol.md").write_text("copied protocol\n")
    (candidate/"run_manifest.json").write_text(json.dumps({"git_commit": "new", "hf_revision": None}))
    assert all(row["status"] == "PASS" for row in compare_runs(reference, candidate))


def test_compare_cli_missing_row_exits_nonzero(tmp_path, monkeypatch, capsys):
    import sys
    from scripts.compare_v2_runs import main
    reference, candidate = tmp_path/"ref", tmp_path/"new"
    tiny_output(reference)
    tiny_output(candidate)
    metrics = candidate/"07_eval/per_run_metrics.csv"
    pd.read_csv(metrics).iloc[:1].to_csv(metrics, index=False)
    monkeypatch.setattr(sys, "argv", ["compare_v2_runs.py", str(reference), str(candidate)])
    with pytest.raises(SystemExit) as exc: main()
    assert exc.value.code == 1
    assert "FAIL | metric values" in capsys.readouterr().out


def test_rank_metric_difference_stays_a_failure_with_detailed_keys():
    ref = pd.DataFrame([dict(condition="gcn_string", repeat=0, fold=0, metric="node_mut_auroc", value=.4249786871270247)])
    candidate = ref.assign(value=.4260869565217391)
    with pytest.raises(ValueError) as exc:
        compare_metric_rows(ref, candidate, set(), 1e-4)
    message = str(exc.value)
    assert "gcn_string" in message and "node_mut_auroc" in message
    assert "reference" in message and "candidate" in message and "absdiff" in message
    assert "1 failing rows" in message


def test_rank_audit_exposes_small_score_changes_without_tolerance_exception():
    from scripts.compare_v2_runs import rank_diagnostics
    result = rank_diagnostics([1, 0], [.50000001, .5], [.5, .50000001])
    assert result["max_score_absdiff"] < 1e-7
    assert result["changed_pair_orders"] == 1
    assert result["max_changed_pair_score_margin"] < 1e-7
    assert result["replay_reference_auroc"] == 1.
    assert result["replay_candidate_auroc"] == 0.
    with pytest.raises(ValueError, match="both binary classes"):
        rank_diagnostics([0, 0], [.5, .5], [.5, .5])


def test_checkpoint_audit_replays_identical_inputs_and_restores_torch_policy(monkeypatch):
    import copy
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    import torch
    import scripts.compare_v2_runs as compare
    model = torch.nn.Linear(1, 1)
    first = dict(cfg={"channels": [{"name": "mut", "type": "binary"}]}, genes=["A"], ids=np.array(["a", "b"]),
                 data=SimpleNamespace(x=np.array([[[1.]], [[0.]]]), m=np.ones((2, 1, 1))), test=np.array([0, 1]),
                 masks={"node": np.ones((2, 1, 1), dtype=bool)},
                 graph=SimpleNamespace(edge_index=torch.tensor([[0], [0]]), edge_weight=torch.ones(1)), model=model)
    contexts = {"reference": first, "candidate": copy.deepcopy(first)}
    monkeypatch.setattr(compare, "load_prediction_context", lambda root, *args: contexts[Path(root).name])
    monkeypatch.setattr(compare, "replay_scores", lambda context, *args: (np.array([1, 0]), np.array([.6, .4])))
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    result = compare.audit_prediction_case(Path("reference"), Path("candidate"), "gatv2_string_untrained", 0, 0, "node_mut_auroc")
    assert result["checkpoint_parameters_exact"] and result["max_checkpoint_absdiff"] == 0
    assert result["inputs_exact"] and result["max_score_absdiff"] == 0
    assert result["device"] == "cpu"
    assert torch.get_num_threads() == old_threads
    assert torch.are_deterministic_algorithms_enabled() == old_deterministic
    contexts["candidate"]["data"].x[0, 0, 0] = 0
    with pytest.raises(ValueError, match="inputs, masks or graph differ"):
        compare.audit_prediction_case(Path("reference"), Path("candidate"), "gatv2_string_untrained", 0, 0, "node_mut_auroc")


def test_diagnostic_replay_cannot_turn_failed_metric_comparison_into_pass(tmp_path, monkeypatch, capsys):
    import sys
    import scripts.compare_v2_runs as compare
    reference, candidate = tmp_path/"ref", tmp_path/"new"
    tiny_output(reference)
    tiny_output(candidate, model=.4)
    monkeypatch.setattr(compare, "audit_failed_predictions", lambda *args: [{"checkpoint_parameters_exact": True}])
    monkeypatch.setattr(sys, "argv", ["compare_v2_runs.py", str(reference), str(candidate), "--audit-predictions"])
    with pytest.raises(SystemExit) as exc: compare.main()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "FAIL | metric values" in output and "strict CSV verdict unchanged" in output
