"""Configuration loading, deep merging, and reproducibility hashes."""
import hashlib
import json
import os
from pathlib import Path

import yaml

NUMERICS_POLICY = "torch-deterministic-v1"
DEFAULT_CUBLAS_WORKSPACE = ":4096:8"


def deep_merge(base, update):
    out = dict(base)
    for key, value in update.items():
        if key == "endpoints" and isinstance(value, dict):
            # Endpoint registries are complete per cancer, allowing an exact legacy registry in the equivalence config.
            out[key] = {**out.get(key, {}), **value}
        else:
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
        protocol = cfg.get("protocol", {}).get(cancer)
        if not isinstance(protocol, dict):
            raise ValueError(f"Missing protocol block for {cancer}")
        fields = {"primary_metrics", "fold_checks", "single_gene", "attribution_endpoints", "control_genes", "report_note", "protocol_file"}
        if fields - set(protocol):
            raise ValueError(f"Incomplete protocol.{cancer}: missing {sorted(fields-set(protocol))}")
        endpoints = cfg["endpoints"][cancer]
        binary = endpoints.get("binary", {})
        valid_metrics = {"node_rna_r2"}
        valid_metrics.update(f"{name}_auroc_gene_latent_pca" for name in binary)
        valid_metrics.update(f"{name}_macro_auroc_gene_latent_pca" for name in endpoints.get("multiclass", {}))
        valid_metrics.update(f"{name}_delta_c_gene_latent_pca" for name in endpoints.get("survival", {}))
        for metric in protocol["primary_metrics"]:
            if metric not in valid_metrics:
                raise ValueError(f"Unknown primary metric or endpoint in protocol.{cancer}: {metric}")
        for endpoint in [protocol["single_gene"]["endpoint"], *protocol["attribution_endpoints"], *protocol["control_genes"]]:
            if endpoint not in binary:
                raise ValueError(f"Unknown binary endpoint in protocol.{cancer}: {endpoint}")


def config_hash(cfg):
    # Invalidate caches produced before deterministic CUDA reductions were required.
    policy = dict(numerics_policy=NUMERICS_POLICY,
                  cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG", DEFAULT_CUBLAS_WORKSPACE),
                  config=cfg)
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
