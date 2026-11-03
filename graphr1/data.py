"""PyG batches preserving shared graph topology and missingness masks."""
import numpy as np
import torch
from torch_geometric.data import Data, Batch


class FoldTensorData:
    def __init__(self, values, observed, normalizer, patient_ids):
        self.x, self.m = normalizer.transform(values, observed)
        self.patient_ids = list(patient_ids)

    def make_batch(self, indices, graph, mask=None):
        items = []
        for local, idx in enumerate(indices):
            values = self.x[idx].copy()
            observed = self.m[idx].copy()
            if mask is not None:
                hidden = np.asarray(mask[local], dtype=bool)
                values[hidden] = 0
                observed[hidden] = 0
            node_features = np.concatenate([values, observed], axis=1)
            items.append(Data(x=torch.from_numpy(node_features), edge_index=graph.edge_index, edge_weight=graph.edge_weight, edge_attr=graph.edge_attr))
        return Batch.from_data_list(items)
