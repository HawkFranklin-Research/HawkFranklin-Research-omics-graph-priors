"""Seeded undirected degree-preserving double-edge swaps."""
import random
import numpy as np
import torch
from . import GraphArtifact


def rewire_graph(graph, n_nodes, seed, swaps_per_edge=10, max_attempt_factor=20):
    src, dst = graph.edge_index.cpu().numpy()
    weights = graph.edge_weight.cpu().numpy()
    edges = {}
    for a, b, w in zip(src, dst, weights):
        if a != b:
            key = tuple(sorted((int(a), int(b))))
            edges.setdefault(key, float(w))
    pairs = list(sorted(edges))
    if len(pairs) < 2:
        raise ValueError("Rewiring requires at least two undirected edges")
    original = set(pairs)
    current = set(pairs)
    rng = random.Random(seed)
    requested = int(len(pairs) * swaps_per_edge)
    accepted = attempts = 0
    while accepted < requested and attempts < max_attempt_factor * max(requested, 1):
        attempts += 1
        i, j = rng.sample(range(len(pairs)), 2)
        a, b = pairs[i]
        c, d = pairs[j]
        if rng.randrange(2):
            c, d = d, c
        if len({a, b, c, d}) != 4:
            continue
        p, q = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if p in current or q in current:
            continue
        current.remove(pairs[i]); current.remove(pairs[j])
        current.add(p); current.add(q)
        pairs[i], pairs[j] = p, q
        accepted += 1
    old_degree = np.bincount(np.array(list(original)).ravel(), minlength=n_nodes)
    new_degree = np.bincount(np.array(pairs).ravel(), minlength=n_nodes)
    if not np.array_equal(old_degree, new_degree):
        raise AssertionError("Rewiring changed node degrees")
    shuffled_weights = list(edges.values())
    rng.shuffle(shuffled_weights)
    directed = pairs + [(b, a) for a, b in pairs]
    return GraphArtifact(torch.tensor(directed, dtype=torch.long).T.contiguous(), torch.tensor(shuffled_weights * 2, dtype=torch.float32), graph.name + "_rewired", graph.fold, seed, dict(requested_swaps=requested, accepted_swaps=accepted, attempts=attempts, degree_preserved=True, retained_fraction=len(original & current) / len(original)))
