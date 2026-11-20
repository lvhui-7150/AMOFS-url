"""AMOFS: archive-progress adaptive schedule (ANCF) + Archive-Guided Discrete
Mutation (AGDM) on top of an elitist, NSGA-II-style environmental selection.

ANCF.  Let HV_t be the archive hypervolume after generation t and
    kappa_t = clip( (HV_t - HV_{t-w}) / (HV_t - HV_0 + 1e-9), 0, 1 )
the share of the total hypervolume gain obtained in the last w generations
(kappa_t = 1 while the archive is still improving, -> 0 once it stagnates).
The convergence factor is
    a(t) = 2 (1 - t/T)^alpha * kappa_t^beta,
    p_c(t) = pc_min + (pc_max - pc_min)(1 - a/2),
    eta(t) = eta_min + (eta_max - eta_min)(a/2).

AGDM.  With archive selection frequency s_i and crowding-weighted frequency
c_i, r_i = lam*s_i + (1-lam)*c_i and
    p_on_i  = clip(eta * r_i,       eta*eps, eta*(1-eps)),
    p_off_i = clip(eta * (1 - r_i), eta*eps, eta*(1-eps)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import AMOFSParams
from .indicators import hypervolume
from .nsga_common import (Archive, crowding_distance, environmental_selection,
                          fast_non_dominated_sort, normalised_crowding)


@dataclass
class RunResult:
    arch_X: np.ndarray
    arch_F: np.ndarray
    hv_log: List[float] = field(default_factory=list)          # fast HV per generation
    sel_freq_log: List[np.ndarray] = field(default_factory=list)
    a_log: List[float] = field(default_factory=list)
    eta_log: List[float] = field(default_factory=list)
    archive_size_log: List[int] = field(default_factory=list)
    truncations: int = 0
    n_evaluations: int = 0


def rank_and_crowding(F: np.ndarray):
    rank = np.empty(F.shape[0], dtype=int)
    cd = np.empty(F.shape[0], dtype=float)
    for r, front in enumerate(fast_non_dominated_sort(F)):
        rank[front] = r
        cd[front] = crowding_distance(F[front])
    return rank, cd


def tournament(rank, cd, rng) -> int:
    i, j = rng.integers(0, rank.shape[0], size=2)
    if rank[i] != rank[j]:
        return int(i if rank[i] < rank[j] else j)
    return int(i if cd[i] >= cd[j] else j)


def uniform_crossover(p1, p2, pc, rng):
    if rng.random() > pc:
        return p1.copy()
    swap = rng.random(p1.shape[0]) < 0.5
    return np.where(swap, p1, p2).astype(np.int8)


def agdm(child, r, eta, eps, rng):
    lo, hi = eta * eps, eta * (1.0 - eps)
    p_on = np.clip(eta * r, lo, hi)
    p_off = np.clip(eta * (1.0 - r), lo, hi)
    u = rng.random(child.shape[0])
    out = child.copy()
    out[(child == 0) & (u < p_on)] = 1
    out[(child == 1) & (u < p_off)] = 0
    return out


def uniform_bitflip(child, eta, rng):
    """Unguided counterpart of AGDM with the same expected number of flips."""
    rate = min(max(eta * 0.5, 1.0 / child.shape[0]), 0.5)
    out = child.copy()
    out[rng.random(child.shape[0]) < rate] ^= 1
    return out


def ensure_nonempty(child, rng):
    if child.sum() == 0:
        child[rng.integers(0, child.shape[0])] = 1
    return child


def run_amofs(ev, n_features: int, pop_size: int, generations: int, params: AMOFSParams,
              seed: int = 0, archive_cap: int = 0, init_density: float = 0.3) -> RunResult:
    rng = np.random.default_rng(seed)
    P = (rng.random((pop_size, n_features)) < init_density).astype(np.int8)
    for row in P:
        ensure_nonempty(row, rng)
    FP = ev.evaluate_population(P)

    arch = Archive(n_features, FP.shape[1], cap=archive_cap)
    arch.update(P, FP)
    hv0 = hypervolume(arch.F, exact=False)
    hv_hist = [hv0]
    res = RunResult(arch_X=arch.X, arch_F=arch.F)
    w = max(1, int(params.window))

    for t in range(1, generations + 1):
        # ---- ANCF -----------------------------------------------------
        decay = (1.0 - t / generations) ** params.alpha
        if params.use_ancf:
            hv_now, hv_prev = hv_hist[-1], hv_hist[max(0, len(hv_hist) - 1 - w)]
            kappa = (hv_now - hv_prev) / (hv_now - hv0 + 1e-9) if t > w else 1.0
            kappa = float(np.clip(kappa, 0.0, 1.0))
            a = 2.0 * decay * kappa ** params.beta
        else:
            a = 2.0 * decay
        pc = params.pc_min + (params.pc_max - params.pc_min) * (1.0 - a / 2.0)
        eta = params.eta_min + (params.eta_max - params.eta_min) * (a / 2.0)

        # ---- archive statistics for AGDM ---------------------------------
        if len(arch) > 0:
            s = arch.X.mean(axis=0)
            cdn = normalised_crowding(arch.F)
            c = (arch.X * cdn[:, None]).sum(axis=0) / (cdn.sum() + 1e-12)
        else:
            s = c = np.full(n_features, 0.5)
        r = params.lam * s + (1.0 - params.lam) * c

        # ---- variation ------------------------------------------------
        rank, cd = rank_and_crowding(FP)
        Q = np.empty_like(P)
        for j in range(pop_size):
            p1, p2 = P[tournament(rank, cd, rng)], P[tournament(rank, cd, rng)]
            child = uniform_crossover(p1, p2, pc, rng)
            child = agdm(child, r, eta, params.eps, rng) if params.use_agdm else uniform_bitflip(child, eta, rng)
            Q[j] = ensure_nonempty(child, rng)
        FQ = ev.evaluate_population(Q)

        # ---- selection + archive ----------------------------------------
        P, FP = environmental_selection(np.vstack([P, Q]), np.vstack([FP, FQ]), pop_size)
        arch.update(Q, FQ)

        hv_hist.append(hypervolume(arch.F, exact=False))
        res.hv_log.append(hv_hist[-1])
        res.sel_freq_log.append(arch.X.mean(axis=0))
        res.a_log.append(a)
        res.eta_log.append(eta)
        res.archive_size_log.append(len(arch))

    res.arch_X, res.arch_F = arch.X, arch.F
    res.truncations = arch.truncations
    res.n_evaluations = getattr(ev, "n_evaluations", 0)
    return res
