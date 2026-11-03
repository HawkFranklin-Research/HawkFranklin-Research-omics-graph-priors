"""Graph-free training-fold baselines."""
import numpy as np
from sklearn.decomposition import PCA


def fit_pca(values, train_idx, components=16, channel=None):
    x = values if channel is None else values[:, :, channel:channel+1]
    flat = x.reshape(len(x), -1)
    model = PCA(n_components=components, random_state=0).fit(flat[train_idx])
    return model, model.transform(flat)


class LowRankImputer:
    def __init__(self, rank=16, iterations=20):
        self.rank, self.iterations = rank, iterations

    def fit(self, values, observed):
        x = np.asarray(values).reshape(len(values), -1)
        mask = np.asarray(observed, dtype=bool).reshape(x.shape)
        filled = np.where(mask, x, 0)
        for _ in range(self.iterations):
            _, _, vt = np.linalg.svd(filled, full_matrices=False)
            self.components = vt[:self.rank]
            filled = np.where(mask, x, (filled @ self.components.T) @ self.components)
        return self

    def transform(self, values, observed):
        x = np.asarray(values).reshape(len(values), -1)
        mask = np.asarray(observed, dtype=bool).reshape(x.shape)
        results = []
        for row, keep in zip(x, mask):
            design = self.components[:, keep].T
            coefficient = np.linalg.lstsq(design, row[keep], rcond=None)[0] if keep.any() else np.zeros(len(self.components))
            results.append(coefficient @ self.components)
        return np.asarray(results).reshape(values.shape)


def fit_mofa(values, observed, train_idx, factors=16, seed=0):
    """Optional MOFA factors with fixed training weights for test projection."""
    from mofapy2.run.entry_point import entry_point
    model = entry_point()
    model.set_data_options(scale_groups=False, scale_views=False)
    views = [[np.where(observed[train_idx, :, c], values[train_idx, :, c], np.nan)] for c in range(values.shape[-1])]
    model.set_data_matrix(views)
    model.set_model_options(factors=factors)
    model.set_train_options(iter=1000, seed=seed, verbose=False)
    model.build()
    model.run()
    weights = model.model.nodes["W"].getExpectation()
    projections = []
    for c, weight in enumerate(weights):
        projected = []
        for row, keep in zip(values[:, :, c], observed[:, :, c].astype(bool)):
            projected.append(np.linalg.lstsq(weight[keep], row[keep], rcond=None)[0] if keep.any() else np.zeros(factors))
        projections.append(projected)
    return np.mean(projections, axis=0)
