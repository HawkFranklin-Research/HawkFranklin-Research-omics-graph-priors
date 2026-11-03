"""Run fold-specific graph construction and training conditions."""
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .data import FoldTensorData
from .config import config_hash
from .evaluate.recon import reconstruction_metrics
from .graphs.coexpr import build_coexpression_graph
from .graphs.io import load_graph, save_graph
from .graphs.rewire import rewire_graph
from .masking import fixed_mask
from .models import build_model
from .normalize import ChannelNormalizer
from .readouts import gene_latent_pca, mean_pool
from .runtime import seed_everything, save_eval_masks
from .train import fit


def run_training(cfg, cancer, paths, force=False, graphs_only=False):
    cube = np.load(paths.stage("01_cube") / "cube.npz")
    patient_ids = cube["patient_ids"].astype(str)
    universe = cube["genes"].astype(str).tolist()
    genes = pd.read_csv(paths.stage("03_nodes") / "nodes.csv").gene.astype(str).tolist()
    selected = np.array([universe.index(gene) for gene in genes])
    values, observed = cube["values"][:, selected, :], cube["observed"][:, selected, :]
    lookup = {patient: i for i, patient in enumerate(patient_ids)}
    folds = json.loads((paths.stage("02_folds") / "folds.json").read_text())
    folds = [f for f in folds if f["repeat"] < cfg.get("execution_max_seeds", cfg["cv"]["n_repeats"]) and f["fold"] < cfg.get("execution_max_folds", cfg["cv"]["n_folds"])]
    string = load_graph(paths.graph("string"))
    missing = []
    for fold in folds:
        repeat, number = fold["repeat"], fold["fold"]
        train_idx = np.array([lookup[x] for x in fold["train_ids"]])
        val_idx = np.array([lookup[x] for x in fold["val_ids"]])
        test_idx = np.array([lookup[x] for x in fold["test_ids"]])
        normalizer = ChannelNormalizer([c["type"] for c in cfg["channels"]]).fit(values, observed, train_idx, patient_ids)
        data = FoldTensorData(values, observed, normalizer, patient_ids)
        from .checks import check_normalization, check_graph, check_embeddings
        check_normalization(normalizer, data.x, observed, train_idx, test_idx, patient_ids)
        normalizer.save(paths.normalization(repeat, number))
        mask_seed = 3000 + 10 * repeat + number
        masks = {mode: fixed_mask(data.m[test_idx], cfg["training"]["entry_rate"] if mode == "entry" else cfg["training"]["node_rate"], mask_seed + (mode == "node"), mode) for mode in ("entry", "node")}
        mask_dir = paths.stage("05_runs") / "_eval_masks"
        mask_dir.mkdir(exist_ok=True)
        save_eval_masks(paths.eval_mask(repeat, number), patient_ids[test_idx], masks)
        coexpr_cfg = cfg["graphs"]["coexpr"]
        rna_index = next(i for i, c in enumerate(cfg["channels"]) if c["name"] == cfg["cube"]["coverage_channel"])
        coexpr = build_coexpression_graph(data.x[train_idx, :, rna_index], string.diagnostics["undirected_edges"], coexpr_cfg["k_min"], coexpr_cfg["k_max"], coexpr_cfg["tolerance"], number)
        graph_map = {"string": string, "coexpr": coexpr, "none": string}
        graph_map["string_rewired"] = rewire_graph(string, len(genes), repeat, cfg["graphs"]["swaps_per_edge"])
        graph_map["coexpr_rewired"] = rewire_graph(coexpr, len(genes), 1000 + repeat * 100 + number, cfg["graphs"]["swaps_per_edge"])
        for name, graph in graph_map.items():
            if name != "none":
                check_graph(graph, len(genes))
                if "rewired" in name:
                    source = graph_map[name.removesuffix("_rewired")]
                    assert torch.equal(torch.bincount(graph.edge_index[0], minlength=len(genes)), torch.bincount(source.edge_index[0], minlength=len(genes)))
                save_graph(graph, genes, paths.graph(name, repeat, number))
        if graphs_only:
            continue
        for spec in cfg["conditions"]:
            directory = paths.run(spec["name"], repeat, number)
            directory.mkdir(parents=True, exist_ok=True)
            status = directory / "status.json"
            if status.exists() and not force:
                existing = json.loads(status.read_text())
                if existing.get("status") == "done" and existing.get("config_hash") == config_hash(cfg):
                    continue
            started = time.monotonic()
            try:
                seed_everything(repeat, cfg["resources"]["threads"])
                device = cfg["resources"]["device"]
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                model = build_model(spec, len(genes), len(cfg["channels"]), cfg["model"])
                graph = graph_map[spec["graph"]]
                train_cfg = dict(cfg["training"], batch_size=cfg["resources"]["batch_size"], channel_types=[c["type"] for c in cfg["channels"]])
                if "binary_loss" in spec:
                    train_cfg["binary_loss"] = spec["binary_loss"]
                if spec.get("train", True):
                    state, history = fit(model, data, graph, train_idx, val_idx, train_cfg, repeat, device)
                else:
                    model.to(device); state = model.state_dict()
                    from .train import initial_history
                    history = initial_history(model, data, graph, train_idx, val_idx, train_cfg, repeat, device)
                history.to_csv(directory / "history.csv", index=False)
                torch.save(dict(state_dict=state, spec=spec, model_config=cfg["model"], config_hash=config_hash(cfg)), directory / "model.pt")
                from .checks import check_training
                check_training(history, trained=spec.get("train", True))
                model.eval()
                rows = []
                with torch.no_grad():
                    for mode, mask in masks.items():
                        predictions = []
                        for start in range(0, len(test_idx), cfg["resources"]["batch_size"]):
                            idx = test_idx[start:start+cfg["resources"]["batch_size"]]
                            batch = data.make_batch(idx, graph, mask[start:start+len(idx)]).to(device)
                            predictions.append(model(batch)["recon"].cpu().numpy().reshape(len(idx), len(genes), -1))
                        metrics = reconstruction_metrics(data.x[test_idx], np.concatenate(predictions), mask, cfg["channels"])
                        rows.extend(dict(condition=spec["name"], repeat=repeat, fold=number, task=mode, **row) for row in metrics)
                pd.DataFrame(rows).to_csv(directory / "recon_metrics.csv", index=False)
                embeddings = []
                with torch.no_grad():
                    for idx in range(len(patient_ids)):
                        z = model(data.make_batch([idx], graph).to(device))["z"].cpu().numpy()
                        embeddings.append(z.reshape(-1))
                embeddings = np.stack(embeddings)
                if spec["model"] == "dense_ae":
                    primary = embeddings
                    pooled = embeddings
                else:
                    latent = embeddings.reshape(len(patient_ids), len(genes), -1)
                    primary, _ = gene_latent_pca(latent, train_idx, cfg["model"]["latent"])
                    pooled = mean_pool(latent)
                check_embeddings(primary)
                assert np.isfinite(pd.DataFrame(rows).query("metric == 'r2'").value).all()
                split = np.full(len(patient_ids), "test", dtype=object)
                split[train_idx], split[val_idx] = "train", "val"
                for name, array in (("gene_latent_pca", primary), ("mean_pool", pooled)):
                    frame = pd.DataFrame(array, columns=[f"emb_{i}" for i in range(array.shape[1])])
                    frame.insert(0, "split", split); frame.insert(0, "patient_id", patient_ids)
                    frame.to_parquet(directory / f"embeddings_{name}.parquet", index=False)
                import resource
                vram = torch.cuda.max_memory_reserved(device) / 1024**2 if str(device).startswith("cuda") else 0
                status.write_text(json.dumps(dict(status="done", config_hash=config_hash(cfg), seconds=time.monotonic()-started, peak_vram_reserved_mb=vram, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), indent=2))
            except Exception as exc:
                info = dict(status="failed", config_hash=config_hash(cfg), error=str(exc), traceback=traceback.format_exc(), seconds=time.monotonic()-started)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    info["suggestion"] = "CUDA out of memory: retry with a smaller resources.batch_size (for example --set resources.batch_size=1)"
                    torch.cuda.empty_cache()
                status.write_text(json.dumps(info, indent=2))
                missing.append(dict(condition=spec["name"], repeat=repeat, fold=number, error=str(exc)))
    pd.DataFrame(missing, columns=["condition", "repeat", "fold", "error"]).to_csv(paths.stage("07_eval") / "missing_runs.csv", index=False)
