#!/usr/bin/env python3
"""Run selected stages of the v2 multiomics experiment."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphr1g.config import load_config
from graphr1g.runtime import limit_threads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--conditions")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if cfg["data"].get("hf", {}).get("repo"):
        # The runner's effective pin must also appear in the resolved config and its hash.
        for field, variable in (("repo", "HF_DATASET_REPO"), ("revision", "HF_DATASET_REVISION")):
            if os.environ.get(variable): cfg["data"]["hf"][field] = os.environ[variable]
    limit_threads(cfg["resources"]["threads"])
    from graphr1g.pipeline import run_pipeline
    run_pipeline(cfg, args)


if __name__ == "__main__":
    main()
