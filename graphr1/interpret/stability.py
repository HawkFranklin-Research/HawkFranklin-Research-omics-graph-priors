"""Overlap and rank-biased overlap for attention stability."""
import numpy as np


def jaccard_at_k(a, b, k):
    a, b = set(a[:k]), set(b[:k])
    return len(a & b) / len(a | b) if a | b else 1.0


def rbo(a, b, p=.98):
    a, b = list(a), list(b)
    seen_a, seen_b, score = set(), set(), 0.0
    depth = max(len(a), len(b))
    if depth == 0:
        return 1.0
    overlap = 0
    for i in range(depth):
        if i < len(a):
            x = a[i]
            if x not in seen_a:
                seen_a.add(x)
                if x in seen_b:
                    overlap += 1
        if i < len(b):
            y = b[i]
            if y not in seen_b:
                seen_b.add(y)
                if y in seen_a:
                    overlap += 1
        score += (1-p) * p**i * overlap / (i+1)
    return score + p**depth * overlap / depth


def shuffled_null(a, b, k, seed, draws=100):
    rng = np.random.default_rng(seed)
    return [jaccard_at_k(a, rng.permutation(b), k) for _ in range(draws)]
