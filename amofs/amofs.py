"""The AMOFS algorithm: ANCF schedule + Archive-Guided Discrete Mutation.

Implements Algorithm 1 of the manuscript. Returns the final archive (mask
matrix and objective matrix) plus a per-generation hypervolume log and the
per-feature archive selection-frequency log (used to validate the guided-drift
assumption empirically).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import AMOFSParams
from .indicators import hypervolume
from .nsga_common import (archive_update, crowding_distance,
                          environmental_selection, normalised_crowding)
from .objectives import Evaluator


@dataclass
class RunResult:
    arch_X: np.ndarray
    arch_F: np.ndarray
    hv_log: List[float] = field(default_factory=list)
    sel_freq_log: List[np.ndarray] = field(default_factory=list)


def _binary_tournament(F: np.ndarray, rng) -> int:
    i, j = rng.integers(0, F.shape[0], size=2)
    # prefer the one that dominates; otherwise random (cheap proxy of rank)
    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
        return int(i)
    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
        return int(j)
    return int(i if rng.random() < 0.5 else j)


def _uniform_crossover(p1, p2, pc, rng):
    if rng.random() > pc:
        return p1.copy()
    swap = rng.random(p1.shape[0]) < 0.5
    child = np.where(swap, p1, p2)
    return child.astype(np.int8)


def _ancf(t, t_max, hv_now, hv_ref, alpha, beta):
    g = (1.0 - t / t_max) ** alpha
    h = (1.0 - min(hv_now / (hv_ref + 1e-12), 1.0)) ** beta
    return 2.0 * g * h


def _generation_schedule(t, t_max, alpha):
    return 2.0 * (1.0 - t / t_max) ** alpha


def _agdm(child, s, c, eta, lam, eps, rng):
    """Archive-Guided Discrete Mutation (Eqs. p_on / p_off, with clipping)."""
    on_score = lam * s + (1 - lam) * c
    off_score = lam * (1 - s) + (1 - lam) * (1 - c)
    p_on = np.clip(eta * on_score, eta * eps, eta * (1 - eps))
    p_off = np.clip(eta * off_score, eta * eps, eta * (1 - eps))

    r = rng.random(child.shape[0])
    out = child.copy()
    off_mask = child == 1
    on_mask = child == 0
    out[on_mask & (r < p_on)] = 1
    out[off_mask & (r < p_off)] = 0
    return out


def _uniform_bitflip(child, eta, rng):
    rate = min(max(eta * 0.5, 1.0 / child.shape[0]), 0.5)
    out = child.copy()
    flip = rng.random(child.shape[0]) < rate
    out[flip] ^= 1
    return out


def run_amofs(ev: Evaluator, n_features: int, pop_size: int, generations: int,
              params: AMOFSParams, seed: int = 0,
              hv_ref_init: float = 0.5) -> RunResult:
    rng = np.random.default_rng(seed)

    # initialise population (avoid all-zero masks)
    P = (rng.random((pop_size, n_features)) < 0.3).astype(np.int8)
    P[P.sum(axis=1) == 0, 0] = 1
    FP = ev.evaluate_population(P)

    arch_X, arch_F = archive_update(np.empty((0, n_features), np.int8),
                                    np.empty((0, 4)), P, FP)

    hv_ref = max(hv_ref_init, hypervolume(arch_F))
    res = RunResult(arch_X=arch_X, arch_F=arch_F)

    for t in range(1, generations + 1):
        hv_now = hypervolume(arch_F)
        hv_ref = max(hv_ref, hv_now)
        if params.use_ancf:
            a = _ancf(t, generations, hv_now, hv_ref,
                      params.alpha, params.beta)
        else:
            a = _generation_schedule(t, generations, params.alpha)
        pc = params.pc_min + (params.pc_max - params.pc_min) * (1 - a / 2)
        eta = params.eta_min + (params.eta_max - params.eta_min) * (a / 2)

        # archive statistics for AGDM
        if arch_X.shape[0] > 0:
            s = arch_X.mean(axis=0)
            cd = normalised_crowding(arch_F)
            denom = cd.sum() + 1e-12
            c = (arch_X * cd[:, None]).sum(axis=0) / denom
        else:
            s = np.full(n_features, 0.5)
            c = np.full(n_features, 0.5)

        # generate offspring
        Q = np.empty_like(P)
        for j in range(pop_size):
            i1 = _binary_tournament(FP, rng)
            i2 = _binary_tournament(FP, rng)
            child = _uniform_crossover(P[i1], P[i2], pc, rng)
            if params.use_agdm:
                child = _agdm(child, s, c, eta, params.lam, params.eps, rng)
            else:
                child = _uniform_bitflip(child, eta, rng)
            if child.sum() == 0:
                child[rng.integers(0, n_features)] = 1
            Q[j] = child
        FQ = ev.evaluate_population(Q)

        # environmental selection on combined population
        XU = np.vstack([P, Q])
        FU = np.vstack([FP, FQ])
        P, FP = environmental_selection(XU, FU, pop_size)

        arch_X, arch_F = archive_update(arch_X, arch_F, P, FP)

        res.hv_log.append(hypervolume(arch_F))
        res.sel_freq_log.append(arch_X.mean(axis=0) if arch_X.shape[0] else s)

    res.arch_X, res.arch_F = arch_X, arch_F
    return res
