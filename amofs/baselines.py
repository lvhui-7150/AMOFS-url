"""Baseline feature selectors.

Evolutionary multi-objective baselines (same encoding, budget, archive and
four objectives as AMOFS):
    NSGA-II, SPEA2, MOEA/D (Tchebycheff, Das-Dennis weights)
Two-objective reference (accuracy + cardinality only, the classical
evolutionary FS formulation):  NSGA-II-2obj
Single-population metaheuristics on a weighted sum, run as a multi-start over
K weight vectors so that they also produce a trade-off archive:  BPSO, BGWO
Filter rankings swept over k (non-evolutionary references): MI-rank, RF-rank

Every function returns (arch_X, arch_F, hv_log).
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from .amofs import (ensure_nonempty, rank_and_crowding, tournament, uniform_crossover)
from .indicators import hypervolume
from .nsga_common import Archive, environmental_selection
from .objectives import ObjectiveDropEvaluator

Result = Tuple[np.ndarray, np.ndarray, List[float]]


def _init_pop(n, d, rng, p=0.3):
    P = (rng.random((n, d)) < p).astype(np.int8)
    for row in P:
        ensure_nonempty(row, rng)
    return P


def _bitflip(child, rate, rng):
    out = child.copy()
    out[rng.random(child.shape[0]) < rate] ^= 1
    return ensure_nonempty(out, rng)


# ---------------------------------------------------------------------------
def run_nsga2(ev, d, pop_size, generations, seed=0, archive_cap=0, pc=0.9) -> Result:
    rng = np.random.default_rng(seed)
    P = _init_pop(pop_size, d, rng)
    FP = ev.evaluate_population(P)
    arch = Archive(d, FP.shape[1], archive_cap)
    arch.update(P, FP)
    hv_log = []
    for _ in range(generations):
        rank, cd = rank_and_crowding(FP)
        Q = np.empty_like(P)
        for j in range(pop_size):
            child = uniform_crossover(P[tournament(rank, cd, rng)], P[tournament(rank, cd, rng)], pc, rng)
            Q[j] = _bitflip(child, 1.0 / d, rng)
        FQ = ev.evaluate_population(Q)
        P, FP = environmental_selection(np.vstack([P, Q]), np.vstack([FP, FQ]), pop_size)
        arch.update(Q, FQ)
        hv_log.append(hypervolume(arch.F, exact=False))
    return arch.X, arch.F, hv_log


def run_nsga2_2obj(ev, d, pop_size, generations, seed=0, archive_cap=0) -> Result:
    """Classical accuracy/cardinality NSGA-II; archive re-scored on all four objectives."""
    search_ev = ObjectiveDropEvaluator(ev, (2, 3))
    aX, _, hv_log = run_nsga2(search_ev, d, pop_size, generations, seed, archive_cap)
    aF = ev.evaluate_population(aX)
    arch = Archive(d, aF.shape[1], archive_cap)
    arch.update(aX, aF)
    return arch.X, arch.F, hv_log


# ---------------------------------------------------------------------------
def _spea2_fitness(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt                                  # dom[i, j]: i dominates j
    strength = dom.sum(axis=1).astype(float)
    raw = (dom * strength[:, None]).sum(axis=0)    # sum of strengths of dominators
    k = max(1, int(np.sqrt(n)))
    dist = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(2))
    kth = np.sort(dist, axis=1)[:, min(k, n - 1)]
    return raw + 1.0 / (kth + 2.0)


def run_spea2(ev, d, pop_size, generations, seed=0, archive_cap=0, pc=0.9) -> Result:
    rng = np.random.default_rng(seed)
    P = _init_pop(pop_size, d, rng)
    FP = ev.evaluate_population(P)
    arch = Archive(d, FP.shape[1], archive_cap)
    arch.update(P, FP)
    hv_log = []
    for _ in range(generations):
        fit = _spea2_fitness(FP)

        def pick():
            i, j = rng.integers(0, pop_size, 2)
            return i if fit[i] <= fit[j] else j

        Q = np.empty_like(P)
        for j in range(pop_size):
            Q[j] = _bitflip(uniform_crossover(P[pick()], P[pick()], pc, rng), 1.0 / d, rng)
        FQ = ev.evaluate_population(Q)
        XU, FU = np.vstack([P, Q]), np.vstack([FP, FQ])
        fit_u = _spea2_fitness(FU)
        keep = np.argsort(fit_u, kind="stable")[:pop_size]   # SPEA2 truncation by fitness
        P, FP = XU[keep], FU[keep]
        arch.update(Q, FQ)
        hv_log.append(hypervolume(arch.F, exact=False))
    return arch.X, arch.F, hv_log


# ---------------------------------------------------------------------------
def das_dennis(m: int, n_points: int, rng) -> np.ndarray:
    H = 1
    while True:
        pts = [np.bincount(c, minlength=m) / H for c in combinations_with_replacement(range(m), H)]
        if len(pts) >= n_points:
            break
        H += 1
    W = np.asarray(pts, dtype=float)
    if W.shape[0] > n_points:
        W = W[rng.choice(W.shape[0], n_points, replace=False)]
    return np.clip(W, 1e-3, None)


def run_moead(ev, d, pop_size, generations, seed=0, archive_cap=0, n_neighbors=10, pc=0.9) -> Result:
    rng = np.random.default_rng(seed)
    P = _init_pop(pop_size, d, rng)
    FP = ev.evaluate_population(P)
    m = FP.shape[1]
    W = das_dennis(m, pop_size, rng)
    Bn = np.argsort(np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2).sum(2)), axis=1)[:, :n_neighbors]
    z = FP.min(axis=0)
    arch = Archive(d, m, archive_cap)
    arch.update(P, FP)
    hv_log = []
    tch = lambda f, w: np.max(w * np.abs(f - z))
    for _ in range(generations):
        new_children, new_F = [], []
        for i in range(pop_size):
            k, l = rng.choice(Bn[i], size=2, replace=False)
            child = _bitflip(uniform_crossover(P[k], P[l], pc, rng), 1.0 / d, rng)
            fc = ev.evaluate(child)
            z = np.minimum(z, fc)
            for nb in Bn[i]:
                if tch(fc, W[nb]) <= tch(FP[nb], W[nb]):
                    P[nb], FP[nb] = child, fc
            new_children.append(child); new_F.append(fc)
        arch.update(np.vstack(new_children), np.vstack(new_F))
        hv_log.append(hypervolume(arch.F, exact=False))
    return arch.X, arch.F, hv_log


# ---------------------------------------------------------------------------
def _weight_vectors(m: int, k: int) -> np.ndarray:
    """K weight vectors: uniform + each objective emphasised."""
    W = [np.full(m, 1.0 / m)]
    for j in range(m):
        w = np.full(m, 0.5 / (m - 1)); w[j] = 0.5
        W.append(w)
    return np.asarray(W[:k])


def run_bpso(ev, d, pop_size, generations, seed=0, archive_cap=0, n_weights=5) -> Result:
    rng = np.random.default_rng(seed)
    F0 = ev.evaluate(_init_pop(1, d, rng)[0])
    W = _weight_vectors(F0.shape[0], n_weights)
    gens_each = max(1, generations // len(W))
    arch = Archive(d, F0.shape[0], archive_cap)
    hv_log: List[float] = []
    for w in W:
        V = rng.normal(scale=1.0, size=(pop_size, d))
        X = _init_pop(pop_size, d, rng, p=0.5)
        F = ev.evaluate_population(X)
        scal = F @ w
        pbest, pbest_f = X.copy(), scal.copy()
        gbest = X[int(np.argmin(scal))].copy()
        arch.update(X, F)
        for _ in range(gens_each):
            r1, r2 = rng.random((pop_size, d)), rng.random((pop_size, d))
            V = 0.7 * V + 1.5 * r1 * (pbest - X) + 1.5 * r2 * (gbest - X)
            V = np.clip(V, -6, 6)
            X = (rng.random((pop_size, d)) < 1.0 / (1.0 + np.exp(-V))).astype(np.int8)
            for row in X:
                ensure_nonempty(row, rng)
            F = ev.evaluate_population(X)
            scal = F @ w
            imp = scal < pbest_f
            pbest[imp], pbest_f[imp] = X[imp], scal[imp]
            gbest = pbest[int(np.argmin(pbest_f))].copy()
            arch.update(X, F)
            hv_log.append(hypervolume(arch.F, exact=False))
    return arch.X, arch.F, hv_log


def run_bgwo(ev, d, pop_size, generations, seed=0, archive_cap=0, n_weights=5) -> Result:
    rng = np.random.default_rng(seed)
    F0 = ev.evaluate(_init_pop(1, d, rng)[0])
    W = _weight_vectors(F0.shape[0], n_weights)
    gens_each = max(1, generations // len(W))
    arch = Archive(d, F0.shape[0], archive_cap)
    hv_log: List[float] = []
    for w in W:
        X = _init_pop(pop_size, d, rng)
        F = ev.evaluate_population(X)
        order = np.argsort(F @ w)
        leaders = [X[order[i]].astype(float) for i in range(3)]
        arch.update(X, F)
        for t in range(gens_each):
            a = 2.0 - 2.0 * t / gens_each
            for i in range(pop_size):
                votes = np.zeros(d)
                for L in leaders:
                    A = 2 * a * rng.random(d) - a
                    D = np.abs(2 * rng.random(d) * L - X[i])
                    votes += 1.0 / (1.0 + np.exp(-(L - A * D)))
                X[i] = ensure_nonempty((rng.random(d) < votes / 3.0).astype(np.int8), rng)
            F = ev.evaluate_population(X)
            order = np.argsort(F @ w)
            leaders = [X[order[i]].astype(float) for i in range(3)]
            arch.update(X, F)
            hv_log.append(hypervolume(arch.F, exact=False))
    return arch.X, arch.F, hv_log


# ---------------------------------------------------------------------------
def _rank_sweep(ev, d, scores: np.ndarray, archive_cap: int) -> Result:
    order = np.argsort(-scores, kind="stable")
    masks = np.zeros((d, d), dtype=np.int8)
    for k in range(d):
        masks[k, order[: k + 1]] = 1
    F = ev.evaluate_population(masks)
    arch = Archive(d, F.shape[1], archive_cap)
    arch.update(masks, F)
    return arch.X, arch.F, [hypervolume(arch.F, exact=False)]


def run_mi_rank(ev, d, pop_size, generations, seed=0, archive_cap=0) -> Result:
    mi = mutual_info_classif(ev.Xp_tr, ev.yp_tr, random_state=seed)
    return _rank_sweep(ev, d, mi, archive_cap)


def run_rf_rank(ev, d, pop_size, generations, seed=0, archive_cap=0) -> Result:
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1, min_samples_leaf=2)
    rf.fit(ev.Xp_tr, ev.yp_tr)
    return _rank_sweep(ev, d, rf.feature_importances_, archive_cap)


BASELINES: Dict[str, Callable[..., Result]] = {
    "NSGA-II": run_nsga2,
    "NSGA-II-2obj": run_nsga2_2obj,
    "MOEA/D": run_moead,
    "SPEA2": run_spea2,
    "BPSO": run_bpso,
    "BGWO": run_bgwo,
    "MI-rank": run_mi_rank,
    "RF-rank": run_rf_rank,
}
METHOD_ORDER = ["MI-rank", "RF-rank", "BPSO", "BGWO", "NSGA-II-2obj", "MOEA/D", "SPEA2", "NSGA-II", "AMOFS"]
