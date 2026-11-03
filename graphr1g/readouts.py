"""Patient readouts; PCA is always fitted on the training fold."""
import numpy as np
from sklearn.decomposition import PCA


def gene_latent_pca(node_embeddings, train_idx, n_components=16):
    x = np.asarray(node_embeddings).reshape(len(node_embeddings), -1)
    if min(len(train_idx), x.shape[1]) < n_components:
        raise ValueError("Not enough training samples or features for requested PCA dimension")
    model = PCA(n_components=n_components, random_state=0).fit(x[train_idx])
    return model.transform(x), model


def mean_pool(node_embeddings):
    return np.asarray(node_embeddings).mean(axis=1)
