"""Cross-validated attention extraction and stability summaries."""
import json
import numpy as np
import pandas as pd
import torch
from .graphs.io import load_graph
from .models import build_model
from .normalize import ChannelNormalizer
from .data import FoldTensorData
from .interpret.attention import summarize_attention
from .interpret.stability import jaccard_at_k, rbo, shuffled_null
from .runtime import successful_run, run_guard


def load_context(cfg, paths, directory):
    repeat, fold = map(int, directory.name.replace("r", "").split("_f"))
    entries = json.loads((paths.stage("02_folds") / "folds.json").read_text())
    split = next(x for x in entries if x["repeat"] == repeat and x["fold"] == fold)
    cube = np.load(paths.stage("01_cube") / "cube.npz")
    ids, allgenes = cube["patient_ids"].astype(str), cube["genes"].astype(str).tolist()
    genes = pd.read_csv(paths.stage("03_nodes") / "nodes.csv").gene.astype(str).tolist()
    positions = [allgenes.index(g) for g in genes]
    normalizer = ChannelNormalizer.load(paths.normalization(repeat, fold))
    data = FoldTensorData(cube["values"][:, positions, :], cube["observed"][:, positions, :], normalizer, ids)
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    spec = checkpoint["spec"]
    model = build_model(spec, len(genes), len(cfg["channels"]), checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(cfg["resources"]["device"]).eval()
    graph = load_graph(paths.graph(spec["graph"], repeat, fold))
    lookup = {pid: i for i, pid in enumerate(ids)}
    return repeat, fold, split, data, genes, model, graph, lookup


def attention_stage(cfg, cancer, paths):
    out = paths.stage("09_attention")
    patients = pd.read_csv(paths.stage("01_cube") / "patients.csv").set_index("patient_id")
    endpoint = next(iter(cfg["endpoints"][cancer].get("binary", {})), None)
    for directory in paths.stage("05_runs").glob("gatv2_*/r*_f*"):
        status = directory / "status.json"
        if not successful_run(directory, cfg): continue
        target = out / directory.parent.name / directory.name
        with run_guard(target, cfg):
            repeat, fold, split, data, genes, model, graph, lookup = load_context(cfg, paths, directory)
            edges, weights = [[], []], [[], []]
            labels = []
            for pid in split["test_ids"]:
                batch = data.make_batch([lookup[pid]], graph).to(cfg["resources"]["device"])
                with torch.no_grad(): out_model = model(batch, return_attention=True)
                for layer, (edge, alpha) in enumerate(out_model["attentions"]):
                    edges[layer].append(edge.cpu()); weights[layer].append(alpha.cpu())
                labels.append(patients.loc[pid, endpoint] if endpoint else None)
            for layer in range(2):
                edge_stats, gene_scores = summarize_attention(edges[layer], weights[layer], genes, labels if endpoint else None)
                edge_stats.to_parquet(target / f"edge_stats_L{layer+1}.parquet", index=False)
                gene_scores.to_csv(target / f"gene_scores_L{layer+1}.csv", index=False)
    stability = []
    for condition in out.iterdir():
        if not condition.is_dir(): continue
        for layer in (1, 2):
            files = sorted(file for file in condition.glob(f"r*_f*/gene_scores_L{layer}.csv") if successful_run(file.parent, cfg))
            for i, first in enumerate(files):
                for second in files[i+1:]:
                    a = pd.read_csv(first).sort_values("attention_gene_score", ascending=False).gene.tolist()
                    b = pd.read_csv(second).sort_values("attention_gene_score", ascending=False).gene.tolist()
                    edges_a = pd.read_parquet(first.parent / f"edge_stats_L{layer}.parquet").sort_values("mean_enrichment", ascending=False)
                    edges_b = pd.read_parquet(second.parent / f"edge_stats_L{layer}.parquet").sort_values("mean_enrichment", ascending=False)
                    a_edges = list(zip(edges_a.src_gene, edges_a.dst_gene))
                    b_edges = list(zip(edges_b.src_gene, edges_b.dst_gene))
                    item = dict(condition=condition.name, layer=layer, run_a=first.parent.name, run_b=second.parent.name, gene_rbo=rbo(a,b), edge_rbo=rbo(a_edges,b_edges))
                    for k in (50,100,500):
                        item[f"gene_jaccard_{k}"] = jaccard_at_k(a,b,k)
                        item[f"edge_jaccard_{k}"] = jaccard_at_k(a_edges,b_edges,k)
                        item[f"edge_null_expected_{k}"] = min(k, len(a_edges)) / max(1, 2*len(a_edges)-min(k, len(a_edges)))
                    rng = np.random.default_rng(0)
                    for label, a_rank, b_rank in (("gene", a, b), ("edge", a_edges, b_edges)):
                        null_rbo = []
                        null_jaccard = {k: [] for k in (50, 100, 500)}
                        for _ in range(100):
                            shuffled = [b_rank[j] for j in rng.permutation(len(b_rank))]
                            null_rbo.append(rbo(a_rank, shuffled))
                            for k in null_jaccard: null_jaccard[k].append(jaccard_at_k(a_rank, shuffled, k))
                        item[label + "_null_rbo"] = float(np.mean(null_rbo))
                        for k, values in null_jaccard.items(): item[f"{label}_null_jaccard_{k}"] = float(np.mean(values))
                    stability.append(item)
    trained = out / "gatv2_string"
    for control in ("gatv2_string_untrained", "gatv2_string_rewired"):
        for source in trained.glob("r*_f*/gene_scores_L*.csv"):
            other = out / control / source.parent.name / source.name
            if not other.exists() or not successful_run(source.parent, cfg) or not successful_run(other.parent, cfg): continue
            a = pd.read_csv(source).sort_values("attention_gene_score", ascending=False).gene.tolist()
            b = pd.read_csv(other).sort_values("attention_gene_score", ascending=False).gene.tolist()
            stability.append(dict(condition="gatv2_string_vs_" + control, layer=int(source.stem[-1]), run_a=source.parent.name, run_b=other.parent.name, gene_rbo=rbo(a,b), gene_jaccard_100=jaccard_at_k(a,b,100), gene_null_jaccard_100=float(np.mean(shuffled_null(a,b,100,seed=0)))))
    table = pd.DataFrame(stability)
    table.to_csv(out / "stability.csv", index=False)
    relevant = table[(table.condition == "gatv2_string") & (table.layer == 1) & table.apply(lambda row: row.run_a.split("_f")[-1] == row.run_b.split("_f")[-1], axis=1)] if not table.empty else table
    score = float(relevant.gene_rbo.median()) if not relevant.empty else None
    (out / "gate.json").write_text(json.dumps(dict(median_gene_rbo=score, attention_stable=score is not None and score >= cfg["interpret"]["stability_gate"]), indent=2))


def circularity_stage(cfg, cancer, paths):
    rows = []
    for directory in paths.stage("05_runs").glob("gatv2_string/r*_f*"):
        if not successful_run(directory, cfg): continue
        with run_guard(paths.stage("08_circularity") / "runs" / directory.name, cfg):
            rows.extend(circularity_run(cfg, cancer, paths, directory))
    pd.DataFrame(rows).to_csv(paths.stage("08_circularity") / "circularity_auc.csv", index=False)


def circularity_run(cfg, cancer, paths, directory):
    from .circularity import cluster_recovery
    rows = []
    if successful_run(directory, cfg):
        repeat, fold, split, data, genes, model, graph, lookup = load_context(cfg, paths, directory)
        train = np.array([lookup[x] for x in split["train_ids"]]); test = np.array([lookup[x] for x in split["test_ids"]])
        for name in ("gatv2_string", "gatv2_string_untrained", "random_gaussian"):
            if name == "random_gaussian":
                emb = np.random.default_rng(repeat).normal(size=(len(data.patient_ids), 16))
            else:
                file = paths.run(name, repeat, fold) / "embeddings_gene_latent_pca.parquet"
                if not file.exists() or not successful_run(file.parent, cfg): continue
                frame = pd.read_parquet(file).set_index("patient_id").loc[data.patient_ids]
                emb = frame.filter(regex="^emb_").to_numpy()
            try:
                result = cluster_recovery(emb, data.x[:,:,0], train, test, repeat)
                rows.extend(dict(condition=name, repeat=repeat, fold=fold, classifier=k, auroc=v) for k,v in result.items())
            except Exception as exc:
                import traceback
                target = paths.stage("08_circularity") / name / directory.name
                target.mkdir(parents=True, exist_ok=True)
                (target / "status.json").write_text(json.dumps(dict(status="failed", error=str(exc), traceback=traceback.format_exc()), indent=2))
                rows.append(dict(condition=name, repeat=repeat, fold=fold, error=str(exc)))
    return rows


def attribution_stage(cfg, cancer, paths):
    output = paths.stage("10_attribution")
    agreement = []
    for directory in paths.stage("05_runs").glob("gatv2_*/r*_f*"):
        if not successful_run(directory, cfg): continue
        with run_guard(output / directory.parent.name / directory.name, cfg):
            agreement.extend(attribution_run(cfg, cancer, paths, directory))
    pd.DataFrame(agreement).to_csv(output / "agreement.csv", index=False)


def attribution_run(cfg, cancer, paths, directory):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    from .interpret.attribution import knockouts, grad_input
    patients = pd.read_csv(paths.stage("01_cube") / "patients.csv").set_index("patient_id")
    endpoint = next(iter(cfg["endpoints"][cancer].get("binary", {})), None)
    if endpoint is None: raise ValueError("Attribution requires a configured binary endpoint")
    output = paths.stage("10_attribution")
    agreement = []
    if successful_run(directory, cfg):
        repeat, fold, split, data, genes, model, graph, lookup = load_context(cfg, paths, directory)
        if repeat != 0 and cfg["interpret"].get("patients_per_fold") is not None: return []
        train_idx = [lookup[x] for x in split["train_ids"]]
        indices = [lookup[x] for x in split["test_ids"]][:cfg["interpret"].get("patients_per_fold")]
        if not indices: return []
        train_latent = []
        for idx in train_idx:
            with torch.no_grad(): train_latent.append(model(data.make_batch([idx], graph).to(cfg["resources"]["device"]))["z"].cpu().numpy().reshape(-1))
        pca = PCA(n_components=cfg["model"]["latent"], random_state=0).fit(np.stack(train_latent))
        labels = patients.loc[[data.patient_ids[i] for i in train_idx], endpoint].to_numpy(dtype=float)
        valid = np.isfinite(labels)
        if len(np.unique(labels[valid])) < 2: raise ValueError("Training endpoint has fewer than two classes")
        projected = pca.transform(np.stack(train_latent))
        scaler = StandardScaler().fit(projected[valid])
        lr = LogisticRegression(max_iter=5000).fit(scaler.transform(projected[valid]), labels[valid])
        device = cfg["resources"]["device"]
        components = torch.as_tensor(pca.components_.T, dtype=torch.float32, device=device)
        center = torch.as_tensor(pca.mean_, dtype=torch.float32, device=device)
        scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
        loc = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device)
        coef = torch.as_tensor(lr.coef_.reshape(-1), dtype=torch.float32, device=device)
        bias = float(lr.intercept_[0])
        def embedding_target(result):
            return (result["z"].reshape(-1, len(genes)*result["z"].shape[-1]) - center) @ components
        def hpv_target(result):
            return (((embedding_target(result)-loc)/scale) @ coef + bias).reshape(-1, 1)
        dest = output / directory.parent.name / directory.name
        dest.mkdir(parents=True, exist_ok=True)
        for name, target in (("embedding", embedding_target), ("hpv_logit", hpv_target)):
            knockout = knockouts(model, data, graph, indices, genes, device, target, cfg["resources"]["batch_size"])
            from .checks import check_attribution
            check_attribution(knockout, genes)
            knockout.to_csv(dest / f"knockout_{name}.csv", index=False)
            if name == "hpv_logit":
                ranked = knockout.assign(rank_percentile=knockout.score.rank(pct=True))
                ranked[ranked.gene.eq("CDKN2A")].to_csv(dest / "cdkn2a_rank_percentile.csv", index=False)
            gradients = []
            for idx in indices:
                batch = data.make_batch([idx], graph).to(device)
                gradients.append(grad_input(model, batch, len(cfg["channels"]), target).cpu().numpy())
            grad = pd.DataFrame(dict(gene=genes, score=np.mean(gradients, axis=0)))
            grad.to_csv(dest / f"gradxinput_{name}.csv", index=False)
            attn_file = paths.stage("09_attention") / directory.parent.name / directory.name / "gene_scores_L1.csv"
            if attn_file.exists():
                joined = knockout.merge(grad, on="gene", suffixes=("_knockout", "_gradient")).merge(pd.read_csv(attn_file)[["gene", "attention_gene_score"]], on="gene")
                agreement.append(dict(condition=directory.parent.name, repeat=repeat, fold=fold, target=name, knockout_gradient=spearmanr(joined.score_knockout, joined.score_gradient).statistic, attention_knockout=spearmanr(joined.attention_gene_score, joined.score_knockout).statistic))
    return agreement


def enrichment_stage(cfg, cancer, paths):
    from .references import load_hallmark
    from .interpret.enrichment import prerank
    ref = paths.stage("09_attention") / "gate.json"
    gate = json.loads(ref.read_text())
    hallmark = load_hallmark(__import__("pathlib").Path(cfg["data"]["reference_root"]) / cfg["references"]["hallmark"]["filename"])
    genes = pd.read_csv(paths.stage("03_nodes") / "nodes.csv").gene.tolist()
    out = paths.stage("11_enrichment")
    if not gate["attention_stable"]:
        (out / "SKIPPED_attention_unstable.md").write_text(f"Attention stability gate not met: {gate}\n")
    for ranking in paths.stage("10_attribution").glob("*/*/*.csv"):
        if not successful_run(ranking.parent, cfg): continue
        if not ranking.name.startswith(("knockout_", "gradxinput_")): continue
        if "hpv_logit" not in ranking.name and not gate["attention_stable"]: continue
        target = out / ranking.parent.parent.name / ranking.parent.name / ranking.stem
        with run_guard(target, cfg):
            scores = pd.read_csv(ranking)
            result = prerank(scores, hallmark, genes, seed=0, permutations=cfg["interpret"]["permutations"])
            result.to_csv(target / "gsea_results.csv", index=False)
            result.head(10).to_csv(target / "top10.csv", index=False)
    if gate["attention_stable"]:
        for ranking in paths.stage("09_attention").glob("*/*/gene_scores_L*.csv"):
            if not successful_run(ranking.parent, cfg): continue
            target = out / ranking.parent.parent.name / ranking.parent.name / ranking.stem
            with run_guard(target, cfg):
                scores = pd.read_csv(ranking).rename(columns={"attention_gene_score": "score"})
                result = prerank(scores, hallmark, genes, seed=0, permutations=cfg["interpret"]["permutations"])
                result.to_csv(target / "gsea_results.csv", index=False)
                result.head(10).to_csv(target / "top10.csv", index=False)
