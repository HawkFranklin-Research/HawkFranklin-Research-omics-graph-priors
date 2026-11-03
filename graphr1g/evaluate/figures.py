"""Plot tables with the source data saved alongside each figure."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def metric_figure(frame, metric, directory):
    directory = Path(directory)
    (directory / "figures").mkdir(parents=True, exist_ok=True)
    (directory / "plot_data").mkdir(parents=True, exist_ok=True)
    data = frame[frame.metric.eq(metric)].copy()
    if data.empty:
        raise ValueError(f"No data for {metric}")
    data.to_csv(directory / "plot_data" / f"{metric}.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(data=data, x="condition", y="value", color="#167D7F", ax=ax)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(directory / "figures" / f"{metric}.png", dpi=200)
    plt.close(fig)

PROJECT_PALETTE = {"ink": "#24323B", "graph_green": "#2E7D68", "signal_gold": "#D79A32", "risk_crimson": "#B84A62", "cohort_blue": "#427AA1", "modality_violet": "#7B5EA7", "soft_gray": "#D7DEE2", "paper": "#F7F8F4"}
GROUP_COLORS = ["#2E7D68", "#D79A32", "#B84A62", "#427AA1", "#7B5EA7", "#5D6B74", "#C46D3B", "#4F9D9D"]


def required_figures(table, paths, primary_metrics):
    import pandas as pd
    out = paths.stage("07_eval")
    for sub in ("figures", "plot_data"): (out / sub).mkdir(exist_ok=True)
    comparisons = pd.read_csv(out / "comparisons.csv") if (out / "comparisons.csv").stat().st_size > 1 else pd.DataFrame()
    histories = []
    for file in paths.stage("05_runs").glob("*/*/history.csv"):
        histories.append(pd.read_csv(file).assign(condition=file.parent.parent.name, run=file.parent.name))
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    coverage = pd.read_csv(paths.stage("01_cube") / "qc_coverage.csv")
    datasets = {"reconstruction": table[table.metric.str.endswith("_r2")]}
    zero_lines = set()
    for metric in primary_metrics:
        if metric.endswith("_r2"): continue
        prefix = metric.removesuffix("_gene_latent_pca")
        name = prefix.removesuffix("_auroc").removesuffix("_c")
        datasets[name] = table[table.metric.str.startswith(prefix)]
        if "_delta_c" in metric: zero_lines.add(name)
    datasets.update(paired_differences=comparisons, training=history, harmonization=coverage)
    for name, data in datasets.items():
        data.to_csv(out / "plot_data" / f"{name}.csv", index=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        if name == "harmonization":
            data.set_index("channel")[["rna_gene_overlap_before", "rna_gene_overlap_after"]].plot.bar(ax=ax, color=GROUP_COLORS[:2])
        elif name == "training" and not data.empty:
            median = data.groupby(["condition", "epoch"]).val_loss_EM.median().reset_index()
            sns.lineplot(data=median, x="epoch", y="val_loss_EM", hue="condition", ax=ax)
        elif name == "paired_differences" and not data.empty:
            data = data.dropna(subset=["mean", "ci_low", "ci_high"])
            labels = data.hypothesis + ":" + data.other + ":" + data.metric
            ax.errorbar(data["mean"], range(len(data)), xerr=[data["mean"]-data.ci_low, data.ci_high-data["mean"]], fmt="o")
            ax.set_yticks(range(len(data)), labels, fontsize=6); ax.axvline(0, color="gray")
        elif "value" in data and not data.empty:
            sns.pointplot(data=data, x="condition", y="value", hue="metric", ax=ax, errorbar=("ci", 95), seed=0)
            ax.tick_params(axis="x", rotation=60)
            if name in zero_lines: ax.axhline(0, color="gray")
        fig.tight_layout(); fig.savefig(out / "figures" / f"{name}.png", dpi=150); plt.close(fig)
