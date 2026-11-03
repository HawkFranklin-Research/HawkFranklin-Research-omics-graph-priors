import json
from pathlib import Path
import pandas as pd
import torch
from . import GraphArtifact


def save_graph(graph, genes, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(graph.edge_index, directory / "edge_index.pt")
    torch.save(graph.edge_weight, directory / "edge_weight.pt")
    src, dst = graph.edge_index.cpu().numpy()
    pd.DataFrame(dict(src_gene=[genes[i] for i in src], dst_gene=[genes[i] for i in dst], weight=graph.edge_weight.cpu().numpy())).to_csv(directory / "edges.csv", index=False)
    (directory / "diagnostics.json").write_text(json.dumps(dict(name=graph.name, fold=graph.fold, seed=graph.seed, **graph.diagnostics), indent=2))


def load_graph(directory):
    directory = Path(directory)
    info = json.loads((directory / "diagnostics.json").read_text())
    return GraphArtifact(torch.load(directory / "edge_index.pt", map_location="cpu", weights_only=True), torch.load(directory / "edge_weight.pt", map_location="cpu", weights_only=True), info.pop("name"), info.pop("fold"), info.pop("seed"), info)
