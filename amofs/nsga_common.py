"""Multi-objective primitives (minimisation): dominance matrix, fast
non-dominated sorting, crowding distance, environmental selection, and an
elitist archive with optional cap."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def dominance_matrix(F: np.ndarray) -> np.ndarray:
    """D[i, j] = True iff F[i] dominates F[j]."""
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    return le & lt


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    if F.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if F.shape[0] > 4000:  # chunk to bound memory
        mask = np.ones(F.shape[0], dtype=bool)
        for s in range(0, F.shape[0], 2000):
            blk = F[s:s + 2000]
            le = np.all(F[:, None, :] <= blk[None, :, :], axis=2)
            lt = np.any(F[:, None, :] < blk[None, :, :], axis=2)
            mask[s:s + 2000] = ~np.any(le & lt, axis=0)
        return mask
    return ~np.any(dominance_matrix(F), axis=0)


def fast_non_dominated_sort(F: np.ndarray) -> List[np.ndarray]:
    n = F.shape[0]
    if n == 0:
        return []
    D = dominance_matrix(F)
    count = D.sum(axis=0).astype(int)          # number of dominators of j
    remaining = np.ones(n, dtype=bool)
    fronts: List[np.ndarray] = []
    while remaining.any():
        front = np.flatnonzero(remaining & (count == 0))
        if front.size == 0:  # numerical safety
            front = np.flatnonzero(remaining)
        fronts.append(front)
        remaining[front] = False
        count -= D[front].sum(axis=0).astype(int)
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    if n == 0:
        return np.zeros(0)
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        fj = F[order, j]
        span = fj[-1] - fj[0]
        dist[order[0]] = dist[order[-1]] = np.inf
        if span <= 0 or n < 3:
            continue
        dist[order[1:-1]] += (fj[2:] - fj[:-2]) / span
    return dist


def normalised_crowding(F: np.ndarray) -> np.ndarray:
    cd = crowding_distance(F)
    finite = cd[np.isfinite(cd)]
    hi = finite.max() if finite.size else 1.0
    return np.where(np.isinf(cd), 1.0, cd / (hi + 1e-12))


def environmental_selection(X: np.ndarray, F: np.ndarray, n_keep: int) -> Tuple[np.ndarray, np.ndarray]:
    chosen: List[int] = []
    for front in fast_non_dominated_sort(F):
        if len(chosen) + len(front) <= n_keep:
            chosen.extend(front.tolist())
        else:
            cd = crowding_distance(F[front])
            order = front[np.argsort(-cd, kind="stable")]
            chosen.extend(order[: n_keep - len(chosen)].tolist())
            break
    idx = np.asarray(chosen, dtype=int)
    return X[idx], F[idx]


def dedupe(X: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        return X, F
    _, first = np.unique(X, axis=0, return_index=True)
    keep = np.sort(first)
    return X[keep], F[keep]


class Archive:
    """Elitist archive of non-dominated, unique masks.

    ``cap=0`` keeps every non-dominated solution (Assumption A1 of the
    manuscript).  A positive cap truncates by crowding distance and records
    ``truncations`` so that runs can report whether the cap was ever active.
    """

    def __init__(self, n_features: int, n_obj: int = 4, cap: int = 0):
        self.X = np.empty((0, n_features), dtype=np.int8)
        self.F = np.empty((0, n_obj), dtype=float)
        self.cap = cap
        self.truncations = 0
        self.max_size = 0

    def update(self, X_new: np.ndarray, F_new: np.ndarray) -> None:
        X = np.vstack([self.X, X_new.astype(np.int8)])
        F = np.vstack([self.F, F_new])
        X, F = dedupe(X, F)
        nd = non_dominated_mask(F)
        X, F = X[nd], F[nd]
        if self.cap and X.shape[0] > self.cap:
            cd = crowding_distance(F)
            keep = np.argsort(-cd, kind="stable")[: self.cap]
            X, F = X[keep], F[keep]
            self.truncations += 1
        self.X, self.F = X, F
        self.max_size = max(self.max_size, X.shape[0])

    def __len__(self) -> int:
        return self.X.shape[0]
