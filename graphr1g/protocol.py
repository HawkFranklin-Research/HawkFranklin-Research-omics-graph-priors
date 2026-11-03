"""Per-cancer settings shared by stages; scientific choices live in YAML."""
def cancer_protocol(cfg, cancer=None):
    if cancer is None:
        if len(cfg["cancers"]) != 1:
            raise ValueError("A cancer is required for a multi-cancer experiment")
        cancer = cfg["cancers"][0]
    try:
        return cfg["protocol"][cancer]
    except KeyError as exc:
        raise ValueError(f"Missing protocol block for {cancer}") from exc


def single_gene_name(protocol):
    return "single_gene_" + protocol["single_gene"]["gene"].lower()


def baseline_names(cfg, cancer=None):
    names = ["pca_concat", "pca_rna", "lowrank_impute", "gene_mean", single_gene_name(cancer_protocol(cfg, cancer))]
    if cfg.get("baselines", {}).get("mofa"): names.append("mofa")
    return names


def check_fold_counts(test, checks, relaxed=False):
    for check in checks:
        minimum = check["min_relaxed"] if relaxed else check["min"]
        count = int(test[check["column"]].eq(check["value"]).sum())
        if count < minimum:
            raise ValueError(f"Fold check failed: {check['column']}={check['value']} has {count}, requires {minimum}")


def check_single_gene(table, protocol):
    spec = protocol["single_gene"]
    metric = spec["endpoint"] + "_auroc_gene_latent_pca"
    control = table[table.condition.eq(single_gene_name(protocol)) & table.metric.eq(metric)]
    if control.empty or control.value.isna().any() or (control.value < spec["min_auroc"]).any():
        raise ValueError(f"HARD BLOCKER 3: {spec['gene']} {spec['endpoint'].upper()} AUROC below {spec['min_auroc']:g}: " + control.to_json())
