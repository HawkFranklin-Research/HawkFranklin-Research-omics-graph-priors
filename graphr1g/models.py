"""Autoencoders with matched masking and explicit self-information paths."""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv


class NodeAutoencoder(nn.Module):
    def __init__(self, kind, channels, hidden=64, latent=16, heads=1, dropout=.1):
        super().__init__()
        self.kind, self.dropout = kind, nn.Dropout(dropout)
        def layer(a, b):
            if kind == "gatv2":
                return GATv2Conv(a, b, heads=heads, concat=False, edge_dim=1, add_self_loops=True, fill_value="mean")
            if kind == "gcn":
                return GCNConv(a, b, add_self_loops=True)
            if kind == "sage":
                return SAGEConv(a, b, aggr="mean", root_weight=True)
            if kind == "node_mlp":
                return nn.Linear(a, b)
            raise ValueError(f"Unknown model {kind}")
        self.conv1, self.conv2 = layer(2 * channels, hidden), layer(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ELU(), nn.Linear(hidden, channels))

    def _apply_layer(self, layer, x, graph, attention):
        if self.kind == "gatv2":
            return layer(x, graph.edge_index, graph.edge_attr, return_attention_weights=True) if attention else layer(x, graph.edge_index, graph.edge_attr)
        if self.kind == "gcn":
            return layer(x, graph.edge_index, graph.edge_weight)
        if self.kind == "sage":
            return layer(x, graph.edge_index)
        return layer(x)

    def forward(self, batch, return_attention=False):
        x = batch.x
        attentions = []
        for index, layer in enumerate((self.conv1, self.conv2)):
            output = self._apply_layer(layer, x, batch, return_attention and self.kind == "gatv2")
            if isinstance(output, tuple):
                x, attention = output
                attentions.append(attention)
            else:
                x = output
            if index == 0:
                x = self.dropout(torch.nn.functional.elu(x))
        return {"z": x, "recon": self.decoder(x), "attentions": attentions}


class DensePatientAE(nn.Module):
    def __init__(self, genes, channels, hidden=512, latent=16):
        super().__init__()
        self.genes, self.channels = genes, channels
        self.encoder = nn.Sequential(nn.Linear(genes * channels * 2, hidden), nn.ELU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ELU(), nn.Linear(hidden, genes * channels))

    def forward(self, batch, return_attention=False):
        z = self.encoder(batch.x.reshape(-1, self.genes * self.channels * 2))
        return {"z": z, "recon": self.decoder(z).reshape(-1, self.channels), "attentions": []}


def build_model(spec, genes, channels, config):
    latent = spec.get("latent", config.get("latent", 16))
    if spec["model"] == "dense_ae":
        return DensePatientAE(genes, channels, config.get("dense_hidden", 512), latent)
    return NodeAutoencoder(spec["model"], channels, config.get("hidden", 64), latent, spec.get("heads", config.get("heads", 1)), config.get("dropout", .1))
