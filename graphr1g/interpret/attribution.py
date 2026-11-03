"""Perturb all genes while retaining observation indicators."""
import numpy as np
import pandas as pd
import torch


def knockouts(model, data, graph, indices, genes, device, target_fn, batch_size=8):
    totals = np.zeros(len(genes), dtype=np.float64)
    model.eval()
    for idx in indices:
        base = data.make_batch([idx], graph).to(device)
        with torch.no_grad():
            reference = target_fn(model(base)).detach()
            for start in range(0, len(genes), batch_size):
                nodes = list(range(start, min(start + batch_size, len(genes))))
                batch = data.make_batch([idx] * len(nodes), graph).to(device)
                for copy_index, node in enumerate(nodes):
                    row = copy_index * len(genes) + node
                    channels = data.x.shape[-1]
                    batch.x[row, :channels] = 0
                    batch.x[row, channels:] = 1
                outputs = target_fn(model(batch))
                diffs = (outputs - reference).reshape(len(nodes), -1).norm(dim=1).cpu().numpy()
                totals[nodes] += diffs
    return pd.DataFrame(dict(gene=genes, score=totals / len(indices)))


def grad_input(model, batch, channels, target_fn):
    model.eval()
    batch.x = batch.x.detach().requires_grad_(True)
    target = target_fn(model(batch)).sum()
    gradient, = torch.autograd.grad(target, batch.x)
    return (gradient[:, :channels] * batch.x[:, :channels]).abs().sum(dim=1).detach()
