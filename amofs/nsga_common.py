"""Shared multi-objective primitives: dominance, fast non-dominated sort,
crowding distance, and environmental selection. All objectives are minimised.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff a (weakly) dominates b and is strictly better in >=1 objective."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> List[np.ndarray]:
    """Return a list of fronts; each front is an array of row indices into F."""
    n = F.shape[0]
    S = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return [np.array(f, dtype=int) for f in fronts]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Crowding distance for a set of objective vectors (NSGA-II)."""
    n, m = F.shape
    if n == 0:
        return np.array([])
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j])
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = F[order[0], j], F[order[-1], j]
        span = fmax - fmin
        if span <= 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (F[order[k + 1], j] - F[order[k - 1], j]) / span
    return dist


def normalised_crowding(F: np.ndarray) -> np.ndarray:
    """Crowding distance squashed into [0, 1] (used by AGDM)."""
    cd = crowding_distance(F)
    finite = cd[np.isfinite(cd)]
    hi = finite.max() if finite.size else 1.0
    cd = np.where(np.isinf(cd), 1.0, cd / (hi + 1e-12))
    return cd


def environmental_selection(X: np.ndarray, F: np.ndarray,
                            n_keep: int) -> Tuple[np.ndarray, np.ndarray]:
    """NSGA-II environmental selection: fill by fronts, break ties by crowding."""
    fronts = fast_non_dominated_sort(F)
    chosen: List[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= n_keep:
            chosen.extend(front.tolist())
        else:
            cd = crowding_distance(F[front])
            order = front[np.argsort(-cd)]
            need = n_keep - len(chosen)
            chosen.extend(order[:need].tolist())
            break
    idx = np.array(chosen, dtype=int)
    return X[idx], F[idx]


def non_dominated(X: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the non-dominated subset (first front) of (X, F)."""
    if F.shape[0] == 0:
        return X, F
    fronts = fast_non_dominated_sort(F)
    idx = fronts[0]
    return X[idx], F[idx]


def archive_update(arch_X: np.ndarray, arch_F: np.ndarray,
                   new_X: np.ndarray, new_F: np.ndarray,
                   cap: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Merge new solutions into the archive, keep non-dominated, dedupe, cap."""
    if arch_X.size == 0:
        X, F = new_X, new_F
    else:
        X = np.vstack([arch_X, new_X])
        F = np.vstack([arch_F, new_F])
    # dedupe identical masks
    _, uniq = np.unique(X, axis=0, return_index=True)
    X, F = X[np.sort(uniq)], F[np.sort(uniq)]
    X, F = non_dominated(X, F)
    if X.shape[0] > cap:  # keep the most diverse by crowding
        cd = crowding_distance(F)
        keep = np.argsort(-cd)[:cap]
        X, F = X[keep], F[keep]
    return X, F