"""Fold-specific, observed-only normalization."""
import hashlib
import json
from pathlib import Path

import numpy as np


class ChannelNormalizer:
    def __init__(self, channel_types, sd_floor=1e-3, clip=6.0):
        self.channel_types = list(channel_types)
        self.sd_floor, self.clip = sd_floor, clip

    def fit(self, values, observed, train_idx, patient_ids):
        idx = np.asarray(train_idx, dtype=int)
        if not len(idx) or len(set(idx.tolist())) != len(idx):
            raise ValueError("Training indices must be nonempty and unique")
        mask = np.asarray(observed[idx], dtype=bool)
        x = np.asarray(values[idx], dtype=np.float64)
        if np.any(~np.isfinite(x[mask])):
            raise ValueError("Observed training values must be finite")
        count = mask.sum(axis=0)
        total = np.where(mask, x, 0).sum(axis=0)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        delta = np.where(mask, x - mean, 0)
        variance = np.divide((delta * delta).sum(axis=0), count, out=np.zeros_like(total), where=count > 0)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(np.sqrt(variance), self.sd_floor).astype(np.float32)
        for c, kind in enumerate(self.channel_types):
            if kind == "binary":
                self.mean[:, c], self.std[:, c] = 0, 1
            elif kind != "continuous":
                raise ValueError(f"Unknown channel type: {kind}")
        ids = sorted(str(patient_ids[i]) for i in idx)
        self.train_patient_hash = hashlib.sha1("\n".join(ids).encode()).hexdigest()
        return self

    def transform(self, values, observed):
        mask = np.asarray(observed, dtype=bool)
        x = np.asarray(values, dtype=np.float32)
        if np.any(~np.isfinite(x[mask])):
            raise ValueError("Observed values must be finite")
        transformed = np.where(mask, (x - self.mean) / self.std, 0)
        for c, kind in enumerate(self.channel_types):
            if kind == "continuous":
                transformed[..., c] = np.clip(transformed[..., c], -self.clip, self.clip)
        return transformed.astype(np.float32), mask.astype(np.uint8)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "statistics.npz", mean=self.mean, std=self.std)
        (path / "normalizer.json").write_text(json.dumps(dict(channel_types=self.channel_types, sd_floor=self.sd_floor, clip=self.clip, train_patient_hash=self.train_patient_hash), indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads((path / "normalizer.json").read_text())
        obj = cls(metadata["channel_types"], metadata["sd_floor"], metadata["clip"])
        with np.load(path / "statistics.npz") as stats:
            obj.mean, obj.std = stats["mean"], stats["std"]
        obj.train_patient_hash = metadata["train_patient_hash"]
        return obj
