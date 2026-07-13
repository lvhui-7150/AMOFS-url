"""Baseline feature selectors, all optimising the same four objectives.

Multi-objective baselines (NSGA-II, MOEA/D, SPEA2) return a Pareto archive.
Single-population baselines (BPSO, BGWO) optimise a validation-tuned weighted
scalarisation and return the non-dominated subset of their visited solutions,
so that every method is comparable on HV/IGD.

Each function returns (arch_X, arch_F, hv_log) for a uniform interface with
:func:`amofs.amofs.run_amofs`.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .indicators import hypervolume
from .nsga_common import (archive_update, crowding_distance,
                          environmental_selection, fast_non_dominated_sort,
                          non_dominated)
from .objectives import Evaluator


def _init_pop(n, d, rng, p=0.3):
    P = (rng.random((n, d)) < p).astype(np.int8)
    P[P.sum(axis=1) == 0, 0] = 1
    return P


def _bitflip(child, rate, rng):
    flip = rng.random(child.shape[0]) < rate
    out = child.copy()
    out[flip] ^= 1
    if out.sum() == 0:
        out[rng.integers(0, child.shape[0])] = 1
    return out


def _tournament(F, rng):
    i, j = rng.integers(0, F.shape[0], size=2)
    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
        return int(i)
    return int(j)


def _uniform_xover(p1, p2, pc, rng):
    if rng.random() > pc:
        return p1.copy()
    swap = rng.random(p1.shape[0]) < 0.5
    return np.where(swap, p1, p2).astype(np.int8)


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------
def run_nsga2(ev, d, pop_size, generations, seed=0, pc=0.9):
    rng = np.random.default_rng(seed)
    P = _init_pop(pop_size, d, rng)
    FP = ev.evaluate_population(P)
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), P, FP)
    hv_log = []
    rate = 1.0 / d
    for _ in range(generations):
        Q = np.empty_like(P)
        for j in range(pop_size):
            i1, i2 = _tournament(FP, rng), _tournament(FP, rng)
            child = _uniform_xover(P[i1], P[i2], pc, rng)
            Q[j] = _bitflip(child, rate, rng)
        FQ = ev.evaluate_population(Q)
        XU, FU = np.vstack([P, Q]), np.vstack([FP, FQ])
        P, FP = environmental_selection(XU, FU, pop_size)
        aX, aF = archive_update(aX, aF, P, FP)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


# ---------------------------------------------------------------------------
# SPEA2 (strength-based fitness, archive truncation)
# ---------------------------------------------------------------------------
def _spea2_fitness(F):
    n = F.shape[0]
    dom = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                dom[i, j] = True
    strength = dom.sum(axis=1)
    raw = np.array([strength[dom[:, j]].sum() for j in range(n)], dtype=float)
    # density via k-th nearest neighbour
    k = int(np.sqrt(n)) or 1
    dens = np.zeros(n)
    for i in range(n):
        dists = np.sort(np.sqrt(((F - F[i]) ** 2).sum(axis=1)))
        kth = dists[min(k, n - 1)]
        dens[i] = 1.0 / (kth + 2.0)
    return raw + dens


def run_spea2(ev, d, pop_size, generations, seed=0, pc=0.9):
    rng = np.random.default_rng(seed)
    P = _init_pop(pop_size, d, rng)
    FP = ev.evaluate_population(P)
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), P, FP)
    hv_log = []
    rate = 1.0 / d
    for _ in range(generations):
        fit = _spea2_fitness(FP)
        Q = np.empty_like(P)
        for j in range(pop_size):
            i1, i2 = rng.integers(0, pop_size, 2)
            parent_a = i1 if fit[i1] < fit[i2] else i2
            i3, i4 = rng.integers(0, pop_size, 2)
            parent_b = i3 if fit[i3] < fit[i4] else i4
            child = _uniform_xover(P[parent_a], P[parent_b], pc, rng)
            Q[j] = _bitflip(child, rate, rng)
        FQ = ev.evaluate_population(Q)
        XU, FU = np.vstack([P, Q]), np.vstack([FP, FQ])
        P, FP = environmental_selection(XU, FU, pop_size)
        aX, aF = archive_update(aX, aF, P, FP)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


# ---------------------------------------------------------------------------
# MOEA/D (Tchebycheff decomposition)
# ---------------------------------------------------------------------------
def run_moead(ev, d, pop_size, generations, seed=0, n_neighbors=10):
    rng = np.random.default_rng(seed)
    m = 4
    # weight vectors on the simplex (random; adequate for m=4)
    W = rng.random((pop_size, m))
    W /= W.sum(axis=1, keepdims=True)
    W = np.clip(W, 1e-3, None)
    # neighbourhoods
    dist = np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2).sum(axis=2))
    B = np.argsort(dist, axis=1)[:, :n_neighbors]

    P = _init_pop(pop_size, d, rng)
    FP = ev.evaluate_population(P)
    z = FP.min(axis=0)
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), P, FP)
    hv_log = []
    rate = 1.0 / d

    def tcheby(fv, w):
        return np.max(w * np.abs(fv - z))

    for _ in range(generations):
        for i in range(pop_size):
            k, l = rng.choice(B[i], size=2, replace=True)
            child = _uniform_xover(P[k], P[l], 0.9, rng)
            child = _bitflip(child, rate, rng)
            fc = ev.evaluate(child)
            z = np.minimum(z, fc)
            for nb in B[i]:
                if tcheby(fc, W[nb]) <= tcheby(FP[nb], W[nb]):
                    P[nb] = child
                    FP[nb] = fc
        aX, aF = archive_update(aX, aF, P, FP)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


# ---------------------------------------------------------------------------
# Single-population baselines on a tuned scalarisation
# ---------------------------------------------------------------------------
def _tune_weights(ev: Evaluator, d, rng, n_trials=20):
    """Pick scalarisation weights maximising validation BACC - lambda*card."""
    best_w, best_score = None, -np.inf
    for _ in range(n_trials):
        w = rng.random(4)
        w /= w.sum()
        mask = (rng.random(d) < 0.5).astype(np.int8)
        f = ev.evaluate(mask)
        score = -np.dot(w, f)
        if score > best_score:
            best_score, best_w = score, w
    return best_w


def run_bpso(ev, d, pop_size, generations, seed=0):
    rng = np.random.default_rng(seed)
    w = _tune_weights(ev, d, rng)
    # binary PSO with sigmoid transfer
    Xpos = rng.random((pop_size, d))
    V = rng.normal(scale=1.0, size=(pop_size, d))
    Xbin = (Xpos > 0.5).astype(np.int8)
    Xbin[Xbin.sum(axis=1) == 0, 0] = 1
    F = ev.evaluate_population(Xbin)
    scal = F @ w
    pbest, pbest_f = Xbin.copy(), scal.copy()
    g = int(np.argmin(scal))
    gbest = Xbin[g].copy()
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), Xbin, F)
    hv_log = []
    c1 = c2 = 1.5
    for _ in range(generations):
        r1, r2 = rng.random((pop_size, d)), rng.random((pop_size, d))
        V = 0.7 * V + c1 * r1 * (pbest - Xbin) + c2 * r2 * (gbest - Xbin)
        prob = 1.0 / (1.0 + np.exp(-V))
        Xbin = (rng.random((pop_size, d)) < prob).astype(np.int8)
        Xbin[Xbin.sum(axis=1) == 0, 0] = 1
        F = ev.evaluate_population(Xbin)
        scal = F @ w
        improved = scal < pbest_f
        pbest[improved], pbest_f[improved] = Xbin[improved], scal[improved]
        g = int(np.argmin(pbest_f))
        gbest = pbest[g].copy()
        aX, aF = archive_update(aX, aF, Xbin, F)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


def run_bgwo(ev, d, pop_size, generations, seed=0):
    rng = np.random.default_rng(seed)
    w = _tune_weights(ev, d, rng)
    X = _init_pop(pop_size, d, rng)
    F = ev.evaluate_population(X)
    scal = F @ w
    order = np.argsort(scal)
    alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), X, F)
    hv_log = []
    for t in range(generations):
        a = 2 - 2 * t / generations
        for i in range(pop_size):
            new = np.zeros(d, dtype=np.int8)
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(d), rng.random(d)
                A = 2 * a * r1 - a
                Dist = np.abs(2 * r2 * leader - X[i])
                cand = leader - A * Dist
                # binary transfer
                prob = 1.0 / (1.0 + np.exp(-cand))
                new = new | (rng.random(d) < prob).astype(np.int8)
            if new.sum() == 0:
                new[rng.integers(0, d)] = 1
            X[i] = new
        F = ev.evaluate_population(X)
        scal = F @ w
        order = np.argsort(scal)
        alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
        aX, aF = archive_update(aX, aF, X, F)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


BASELINES: dict = {
    "NSGA-II": run_nsga2,
    "MOEA/D": run_moead,
    "SPEA2": run_spea2,
    "BPSO": run_bpso,
    "BGWO": run_bgwo,
}