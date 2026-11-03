"""Deterministic held-out entry and whole-node masks."""
import numpy as np


def fixed_mask(observed, rate, seed, mode="entry"):
    observed = np.asarray(observed, dtype=bool)
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between zero and one")
    rng = np.random.default_rng(seed)
    if mode == "entry":
        return observed & (rng.random(observed.shape) < rate)
    if mode == "node":
        return observed & (rng.random(observed.shape[:-1] + (1,)) < rate)
    raise ValueError(f"Unknown mask mode: {mode}")


def hide_entries(values, observed, mask):
    values, observed, mask = np.asarray(values), np.asarray(observed, dtype=bool), np.asarray(mask, dtype=bool)
    if values.shape != observed.shape or mask.shape != observed.shape or np.any(mask & ~observed):
        raise ValueError("Mask must match and be a subset of observed entries")
    return np.where(mask, 0, values), np.where(mask, 0, observed).astype(np.uint8)
