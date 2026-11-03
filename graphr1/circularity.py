"""Cluster recovery is an internal consistency check, not external validation."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def cluster_recovery(embedding, rna, train_idx, test_idx, seed, top_genes=200):
    km = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(embedding[train_idx])
    ytrain, ytest = km.labels_, km.predict(embedding[test_idx])
    scores, _ = f_classif(rna[train_idx], ytrain)
    chosen = np.argsort(np.nan_to_num(scores, nan=-np.inf))[-min(top_genes, rna.shape[1]):]
    results = {}
    if len(np.unique(ytest)) != 3:
        raise ValueError("Cluster-recovery AUROC requires all three clusters in test")
    for name, model in [("logistic", LogisticRegression(max_iter=5000)), ("random_forest", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1))]:
        model.fit(rna[train_idx][:, chosen], ytrain)
        probabilities = model.predict_proba(rna[test_idx][:, chosen])
        results[name] = roc_auc_score(label_binarize(ytest, classes=[0,1,2]), probabilities, average="macro", multi_class="ovr")
    return results
