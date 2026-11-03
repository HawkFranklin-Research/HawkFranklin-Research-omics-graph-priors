"""Resource limits and reproducible experiment metadata."""
import os
import random
import resource
import subprocess
import time
from contextlib import contextmanager


@contextmanager
def run_guard(directory, cfg):
    """Record interpretation-unit failures and let the enclosing grid continue."""
    import json
    import traceback
    from .config import config_hash
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        yield
        record = dict(status="done", config_hash=config_hash(cfg))
    except Exception as exc:
        record = dict(status="failed", config_hash=config_hash(cfg), error=str(exc), traceback=traceback.format_exc())
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            record["suggestion"] = "CUDA out of memory: use a smaller resources.batch_size"
            torch.cuda.empty_cache()
    record["seconds"] = time.monotonic() - started
    (directory / "status.json").write_text(json.dumps(record, indent=2))


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def successful_run(directory, cfg):
    import json
    from .config import config_hash
    status = directory / "status.json"
    try:
        record = json.loads(status.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "done" and record.get("config_hash") == config_hash(cfg)


def save_eval_masks(path, patient_ids, masks):
    """Canonical ZIP metadata, member order, dtypes and patient order."""
    import io
    import json
    import zipfile
    import numpy as np
    arrays = {"patient_ids": np.asarray(patient_ids, dtype="U64"), **{k: np.asarray(v, dtype=np.bool_) for k, v in masks.items()}}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[key], allow_pickle=False)
            member = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, buffer.getvalue())
    path.with_suffix(".status.json").write_text(json.dumps(dict(status="done", sha256=sha256_file(path), dtypes={k: str(v.dtype) for k, v in arrays.items()}), indent=2, sort_keys=True))


def configure_determinism():
    """Seeded CUDA scatter/GEMM operations must also use deterministic kernels."""
    from .config import DEFAULT_CUBLAS_WORKSPACE
    # This must precede CUDA initialization (including manual_seed_all).
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", DEFAULT_CUBLAS_WORKSPACE)
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    return torch


def limit_threads(threads):
    if not 1 <= int(threads) <= 6:
        raise ValueError("resources.threads must be between 1 and 6")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    configure_determinism()


def seed_everything(seed, threads=6):
    import numpy as np
    torch = configure_determinism()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def manifest(config_hash, cfg, cancer, output_dir=None, repo_dir=None):
    import importlib.metadata
    from pathlib import Path
    from .protocol import cancer_protocol
    from .config import NUMERICS_POLICY
    import torch
    packages = {}
    for package in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "lifelines", "scikit-survival", "gseapy"):
        try: packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: packages[package] = None
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True).stdout
    commit = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    untracked = len(git("ls-files", "--others", "--exclude-standard").splitlines())
    protocol_file = cancer_protocol(cfg, cancer)["protocol_file"]
    hf = cfg["data"].get("hf") or {}
    hf_repo = os.environ.get("HF_DATASET_REPO") or hf.get("repo")
    hf_revision = os.environ.get("HF_DATASET_REVISION") or hf.get("revision")
    if output_dir is not None and dirty:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "source_diff.patch").write_text(git("diff", "HEAD"))
    return dict(config_hash=config_hash, git_commit=commit, git_dirty=dirty,
                numerics_policy=NUMERICS_POLICY,
                torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                git_dirty_tracked=dirty, git_untracked_count=untracked,
                protocol_file=protocol_file, protocol_sha256=sha256_file(protocol_file),
                hf_repo=hf_repo, hf_revision=hf_revision if hf_repo else None,
                local_processed_root=None if hf_repo else str(cfg["data"]["processed_root"]),
                packages=packages, start_time=time.time(), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
