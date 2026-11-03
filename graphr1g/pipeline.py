"""Stage runner for v2 artifacts; failures are recorded with their tracebacks."""
import json
import logging
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .config import config_hash
from .paths import ExperimentPaths
from .references import fetch_all, load_hgnc, load_hallmark
from .runtime import manifest, seed_everything
from .protocol import cancer_protocol, check_fold_counts

STAGES = ["references", "cube", "folds", "nodes", "graphs", "train", "baselines", "evaluate", "circularity", "attention", "attribution", "enrichment", "report"]
DEPENDENCIES = {stage: STAGES[:i] for i, stage in enumerate(STAGES)}
LOGGER = logging.getLogger(__name__)


def run_pipeline(cfg, args):
    selected = STAGES if args.stages == "all" else args.stages.split(",")
    unknown = set(selected) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    selected = [stage for stage in STAGES if stage in selected]
    if cfg.get("enable_optional_conditions"):
        cfg = dict(cfg, conditions=cfg["conditions"] + cfg.get("optional_conditions", []))
    if args.conditions:
        names = set(args.conditions.split(","))
        cfg = dict(cfg, conditions=[c for c in cfg["conditions"] if c["name"] in names])
        if names != {c["name"] for c in cfg["conditions"]}:
            raise ValueError("Unknown requested condition")
    if args.max_folds:
        cfg = dict(cfg, execution_max_folds=args.max_folds)
    if args.max_seeds:
        cfg = dict(cfg, execution_max_seeds=args.max_seeds)
    digest = config_hash(cfg)
    for cancer in cfg["cancers"]:
        paths = ExperimentPaths(cfg["data"]["output_root"], cfg["experiment_name"], cancer if len(cfg["cancers"]) > 1 else None)
        paths.root.mkdir(parents=True, exist_ok=True)
        resolved = paths.root / "config.resolved.yaml"
        if resolved.exists() and config_hash(yaml.safe_load(resolved.read_text())) != digest and not args.force:
            raise ValueError("Existing output has a different configuration; choose a new experiment name")
        resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
        shutil.copyfile(cancer_protocol(cfg, cancer)["protocol_file"], paths.root / "protocol.md")
        meta = manifest(digest, cfg, cancer, paths.root)
        (paths.root / "run_manifest.json").write_text(json.dumps(meta, indent=2))
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        for stage in selected:
            handler = logging.FileHandler(paths.stage("logs") / f"{stage}.log")
            logging.getLogger().addHandler(handler)
            stage_dir = paths.stage("logs") / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            status = stage_dir / "status.json"
            if status.exists() and not args.force:
                old = json.loads(status.read_text())
                if old.get("status") == "done" and old.get("config_hash") == digest:
                    logging.getLogger().removeHandler(handler)
                    handler.close()
                    continue
            started = time.monotonic()
            try:
                for upstream in DEPENDENCIES[stage]:
                    upstream_status = paths.stage("logs") / upstream / "status.json"
                    if not upstream_status.exists() or (lambda x: x.get("status") != "done" or x.get("config_hash") != digest)(json.loads(upstream_status.read_text())):
                        raise RuntimeError(f"Stage {stage} requires completed {upstream} with matching config hash")
                if stage == "train":
                    train(cfg, cancer, paths, force=args.force)
                elif stage == "baselines":
                    baselines(cfg, cancer, paths, force=args.force)
                    metadata = {}
                else:
                    metadata = HANDLERS[stage](cfg, cancer, paths) or {}
                record = dict(status="done", config_hash=digest, seconds=time.monotonic()-started)
                if stage != "train": record.update(metadata)
            except Exception:
                record = dict(status="failed", config_hash=digest, seconds=time.monotonic()-started, traceback=traceback.format_exc())
                status.write_text(json.dumps(record, indent=2))
                raise
            status.write_text(json.dumps(record, indent=2))
            logging.getLogger().removeHandler(handler)
            handler.close()
        meta["end_time"] = time.time()
        (paths.root / "run_manifest.json").write_text(json.dumps(meta, indent=2))


def references(cfg, cancer, paths):
    fetch_all(cfg["references"], cfg["data"]["reference_root"])


def cube(cfg, cancer, paths):
    from .cube import build_cube
    from .checks import check_cube
    hgnc = load_hgnc(Path(cfg["data"]["reference_root"]) / cfg["references"]["hgnc"]["filename"])
    values, observed, patients, _ = build_cube(cfg, cancer, hgnc, paths.stage("01_cube"))
    check_cube(values, observed, paths.stage("01_cube"), patients, cfg["cohorts"])


def folds(cfg, cancer, paths):
    from .folds import make_folds, check_folds
    patients = pd.read_csv(paths.stage("01_cube") / "patients.csv")
    cv = cfg["cv"]
    strata_report = []
    result = make_folds(patients, cv["n_folds"], cv["n_repeats"], cv["stratify_label"], cv["stratify_event"], cv["validation_fraction"], strata_report)
    (paths.stage("02_folds") / "strata_report.json").write_text(json.dumps(strata_report, indent=2))
    check_folds(result, patients.patient_id)
    assert result == make_folds(patients, cv["n_folds"], cv["n_repeats"], cv["stratify_label"], cv["stratify_event"], cv["validation_fraction"])
    for fold in result:
        test = patients.set_index("patient_id").loc[fold["test_ids"]]
        check_fold_counts(test, cancer_protocol(cfg, cancer)["fold_checks"], cfg["checks"]["relaxed"])
    from .runtime import sha256_file
    file = paths.stage("02_folds") / "folds.json"
    file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"sha256": {"02_folds/folds.json": sha256_file(file)}}


def nodes(cfg, cancer, paths):
    from .nodes import select_nodes
    from .symbols import HGNCHarmonizer
    source = paths.stage("01_cube")
    with np.load(source / "cube.npz") as cube_data:
        hgnc = load_hgnc(Path(cfg["data"]["reference_root"]) / cfg["references"]["hgnc"]["filename"])
        hallmark = load_hallmark(Path(cfg["data"]["reference_root"]) / cfg["references"]["hallmark"]["filename"])
        harmonizer = HGNCHarmonizer(hgnc, cfg["symbols"]["locus_groups"])
        for name, members in hallmark.items():
            hallmark[name] = harmonizer.map(members)["approved"].dropna().tolist()
        if cfg["nodes"]["selection_scope"] != "all_patients":
            raise NotImplementedError("Fold-specific node universes require fold-specific downstream artifacts")
        table = select_nodes(cube_data["values"], cube_data["observed"], cube_data["genes"], pd.read_csv(source / "genes.csv"), hgnc, hallmark, cfg["nodes"])
        if not cfg["checks"]["relaxed"]:
            for name, threshold in (("rna", .95), ("cnv", .90), ("mut", .90)):
                if name + "_coverage" in table and (table[name + "_coverage"] >= .9).mean() < threshold:
                    raise ValueError(f"Node coverage gate failed for {name}")
        from .checks import check_node_coverage
        check_node_coverage(table)
        table.to_csv(paths.stage("03_nodes") / "nodes.csv", index=False)
        pd.DataFrame([dict(hallmark=name, n_node_genes=len(set(members) & set(table.gene))) for name, members in hallmark.items()]).sort_values("n_node_genes", ascending=False).to_csv(paths.stage("03_nodes") / "hallmark_composition.csv", index=False)
        table.nlargest(30, "rna_var").to_csv(paths.stage("03_nodes") / "top30_variance.csv", index=False)
        (paths.stage("03_nodes") / "selection_report.json").write_text(json.dumps(dict(n_nodes=len(table), selection_scope=cfg["nodes"]["selection_scope"]), indent=2))


def graphs(cfg, cancer, paths):
    from .graphs.string_ppi import build_string_graph
    from .graphs.io import save_graph
    from .checks import check_graph
    from .symbols import HGNCHarmonizer
    ref = Path(cfg["data"]["reference_root"])
    hgnc = load_hgnc(ref / cfg["references"]["hgnc"]["filename"])
    genes = pd.read_csv(paths.stage("03_nodes") / "nodes.csv").gene.tolist()
    graph = build_string_graph(ref / cfg["references"]["string_links"]["filename"], ref / cfg["references"]["string_info"]["filename"], genes, HGNCHarmonizer(hgnc, cfg["symbols"]["locus_groups"]), cfg["graphs"]["string"]["min_score"])
    check_graph(graph, len(genes))
    assert 0 < graph.diagnostics["isolated_fraction"] < .5
    save_graph(graph, genes, paths.graph("string"))
    from .stage_runs import run_training
    run_training(cfg, cancer, paths, graphs_only=True)


def train(cfg, cancer, paths, force=False):
    from .stage_runs import run_training
    run_training(cfg, cancer, paths, force=force)


def baselines(cfg, cancer, paths, force=False):
    from .stage_evaluation import baseline_stage
    baseline_stage(cfg, cancer, paths, force=force)


def evaluate(cfg, cancer, paths):
    from .stage_evaluation import evaluate_stage
    evaluate_stage(cfg, cancer, paths)


def attention(cfg, cancer, paths):
    from .stage_interpret import attention_stage
    attention_stage(cfg, cancer, paths)


def circularity(cfg, cancer, paths):
    from .stage_interpret import circularity_stage
    circularity_stage(cfg, cancer, paths)


def attribution(cfg, cancer, paths):
    from .stage_interpret import attribution_stage
    attribution_stage(cfg, cancer, paths)


def enrichment(cfg, cancer, paths):
    from .stage_interpret import enrichment_stage
    enrichment_stage(cfg, cancer, paths)


def report(cfg, cancer, paths):
    metrics = paths.stage("07_eval") / "summary_by_condition.csv"
    if not metrics.exists(): raise FileNotFoundError(metrics)
    text = "# V2 experiment summary\n\n" + cancer_protocol(cfg, cancer)["report_note"] + "\n\n"
    for filename in ("summary_by_condition.csv", "pooled_classification.csv", "comparisons.csv", "comparisons_exploratory.csv", "missing_runs.csv", "endpoint_failures.csv"):
        file = paths.stage("07_eval") / filename
        text += f"[{filename}](../07_eval/{filename})\n\n"
        try: text += pd.read_csv(file).to_markdown(index=False) + "\n\n"
        except pd.errors.EmptyDataError: pass
    paths.stage("12_report").joinpath("summary.md").write_text(text)



HANDLERS = dict(references=references, cube=cube, folds=folds, nodes=nodes, graphs=graphs, train=train, baselines=baselines, evaluate=evaluate, circularity=circularity, attention=attention, attribution=attribution, enrichment=enrichment, report=report)
