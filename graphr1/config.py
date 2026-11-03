"""Configuration loading, deep merging, and reproducibility hashes."""
import hashlib
import json
from pathlib import Path

import yaml


def deep_merge(base, update):
    out = dict(base)
    for key, value in update.items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path, overrides=()):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        validate_override(base, cfg)
        cfg = deep_merge(base, cfg)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Expected key=value: {item}")
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown config path: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown config key: {key}")
        target[parts[-1]] = yaml.safe_load(raw)
    validate_config(cfg)
    return cfg


def validate_override(base, update, prefix=""):
    for key, value in update.items():
        name = prefix + key
        if key not in base: raise ValueError(f"Unknown config key: {name}")
        if isinstance(value, dict) and isinstance(base[key], dict) and name != "endpoints":
            validate_override(base[key], value, name + ".")


def validate_config(cfg):
    required = {"experiment_name", "data", "resources", "cancers", "cohorts", "channels", "cv", "nodes", "conditions", "endpoints", "references", "symbols", "cube", "graphs", "model", "training", "evaluation", "interpret", "checks"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if "train_ready" in cfg["data"]["processed_root"]:
        raise ValueError("V2 must read pre-normalization harmonized tables")
    if not 1 <= cfg["resources"]["threads"] <= 6:
        raise ValueError("Thread budget must be between 1 and 6")
    if len({c["name"] for c in cfg["channels"]}) != len(cfg["channels"]):
        raise ValueError("Duplicate channel names")
    if len({c["name"] for c in cfg["conditions"]}) != len(cfg["conditions"]):
        raise ValueError("Duplicate conditions")
    if cfg["graphs"]["coexpr"]["k_min"] != 1 or cfg["graphs"]["coexpr"]["k_max"] != 50:
        raise ValueError("Revised density matching requires k in 1–50")
    if cfg["resources"]["batch_size"] < 1 or cfg["training"]["epochs"] < 1:
        raise ValueError("batch_size and epochs must be positive")
    for spec in cfg["conditions"] + cfg.get("optional_conditions", []):
        if spec["model"] not in {"gatv2", "gcn", "sage", "node_mlp", "dense_ae"} or spec["graph"] not in {"string", "coexpr", "string_rewired", "coexpr_rewired", "none"}:
            raise ValueError(f"Unknown model or graph in {spec['name']}")
    for cancer in cfg["cancers"]:
        if cancer not in cfg["endpoints"]:
            raise ValueError(f"No endpoint registry for {cancer}")


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
