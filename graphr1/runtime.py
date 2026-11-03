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


def limit_threads(threads):
    if not 1 <= int(threads) <= 6:
        raise ValueError("resources.threads must be between 1 and 6")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def seed_everything(seed, threads=6):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def manifest(config_hash):
    import importlib.metadata
    packages = {}
    for package in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "lifelines", "scikit-survival", "gseapy"):
        try: packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: packages[package] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    return dict(config_hash=config_hash, git_commit=git.stdout.strip(), git_dirty=bool(dirty.stdout), packages=packages, start_time=time.time(), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
