import subprocess
import os
import sys
from types import SimpleNamespace
import pytest
from graphr1g.runtime import manifest


def test_cuda_determinism_is_configured_before_device_initialization(monkeypatch):
    from graphr1g.runtime import configure_determinism
    calls = []
    def deterministic(enabled, warn_only):
        calls.append((enabled, warn_only, os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = SimpleNamespace(use_deterministic_algorithms=deterministic)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert configure_determinism() is fake
    assert calls == [(True, False, ":4096:8")]
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    configure_determinism()
    assert calls[-1] == (True, False, ":16:8")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="Deterministic CUDA"):
        configure_determinism()


def test_resource_setup_enables_determinism_after_thread_limits(monkeypatch):
    import graphr1g.runtime as runtime
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    for name in names: monkeypatch.setenv(name, "1")
    observed = []
    monkeypatch.setattr(runtime, "configure_determinism", lambda: observed.append([os.environ[n] for n in names]))
    runtime.limit_threads(2)
    assert observed == [["2"]*4]


def test_direct_seeding_sets_determinism_before_cuda_seed(monkeypatch):
    from graphr1g.runtime import seed_everything
    calls = []
    fake = SimpleNamespace(
        use_deterministic_algorithms=lambda enabled, warn_only: calls.append(("deterministic", enabled, warn_only)),
        manual_seed=lambda seed: calls.append(("cpu_seed", seed)),
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda seed: calls.append(("cuda_seed", seed))),
        set_num_threads=lambda threads: calls.append(("threads", threads)),
        backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True)),
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    seed_everything(7, threads=2)
    assert calls == [("deterministic", True, False), ("cpu_seed", 7), ("cuda_seed", 7), ("threads", 2)]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert fake.backends.cudnn.deterministic and not fake.backends.cudnn.benchmark


def test_cache_hash_includes_numeric_execution_policy(monkeypatch):
    import hashlib
    import json
    from graphr1g.config import config_hash
    cfg = {"experiment_name": "fixture"}
    old_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    default_hash = config_hash(cfg)
    assert default_hash != old_hash
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert config_hash(cfg) == default_hash
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    assert config_hash(cfg) != default_hash


def test_manifest_tracks_code_and_untracked_files_separately(tmp_path, monkeypatch):
    repo = tmp_path/"repo"
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    git("init")
    source = repo/"code.py"
    source.write_text("original = True\n")
    protocol = repo/"protocol.md"
    protocol.write_text("test protocol\n")
    git("add", "code.py", "protocol.md")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
    monkeypatch.delenv("HF_DATASET_REPO", raising=False)
    monkeypatch.delenv("HF_DATASET_REVISION", raising=False)
    cfg = {"data": {"processed_root": "local/source"}, "protocol": {"BRCA": {"protocol_file": str(protocol)}}}
    out = tmp_path/"output"
    clean = manifest("hash", cfg, "BRCA", out, repo)
    assert clean["git_dirty_tracked"] is False and clean["git_untracked_count"] == 0
    (repo/"scratch.txt").write_text("untracked data\n")
    untracked = manifest("hash", cfg, "BRCA", out, repo)
    assert untracked["git_dirty_tracked"] is False and untracked["git_untracked_count"] == 1
    assert untracked["hf_repo"] is None and untracked["hf_revision"] is None
    assert untracked["local_processed_root"] == "local/source"
    assert len(untracked["protocol_sha256"]) == 64
    source.write_text("modified = True\n")
    dirty = manifest("hash", cfg, "BRCA", out, repo)
    assert dirty["git_dirty_tracked"] is True and dirty["git_untracked_count"] == 1
    assert "+modified = True" in (out/"source_diff.patch").read_text()
    monkeypatch.setenv("HF_DATASET_REPO", "owner/data")
    monkeypatch.setenv("HF_DATASET_REVISION", "a"*40)
    remote = manifest("hash", cfg, "BRCA", out, repo)
    assert remote["hf_repo"] == "owner/data" and remote["hf_revision"] == "a"*40
    assert remote["local_processed_root"] is None
