"""Quality indicators: hypervolume (HV) and inverted generational distance (IGD).

All objectives are minimised and bounded in [0, 1], so the reference point for
HV is (1, 1, 1, 1). HV is computed by Monte-Carlo estimation, which is exact in
expectation and dimension-agnostic; for the four-objective setting this is both
simple and accurate enough at the sample sizes used here.
"""
from __future__ import annotations

import numpy as np

from .nsga_common import non_dominated


def hypervolume(F: np.ndarray, ref: np.ndarray = None,
                n_mc: int = 5_000, seed: int = 0) -> float:
    """Monte-Carlo hypervolume of the dominated region below ``ref``.

    Returns the fraction of the box [0, ref] dominated by at least one point of
    the (non-dominated) front F, i.e. HV normalised to [0, 1].
    """
    if F.shape[0] == 0:
        return 0.0
    m = F.shape[1]
    if ref is None:
        ref = np.ones(m)
    # restrict to points inside the box and non-dominated
    inside = np.all(F <= ref, axis=1)
    F = F[inside]
    if F.shape[0] == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low=0.0, high=ref, size=(n_mc, m))
    # a sample is dominated if some front point is <= it in all objectives
    dominated = np.zeros(n_mc, dtype=bool)
    for p in F:
        dominated |= np.all(p <= samples, axis=1)
    vol_box = float(np.prod(ref))
    return float(dominated.mean() * vol_box)


def igd(F: np.ndarray, reference_front: np.ndarray) -> float:
    """Inverted generational distance from ``reference_front`` to F (lower better)."""
    if F.shape[0] == 0 or reference_front.shape[0] == 0:
        return float("inf")
    dists = []
    for r in reference_front:
        d = np.sqrt(np.sum((F - r) ** 2, axis=1))
        dists.append(d.min())
    return float(np.mean(dists))


def build_reference_front(fronts: list) -> np.ndarray:
    """Union of several methods' fronts, reduced to its non-dominated points."""
    allF = np.vstack([f for f in fronts if f.shape[0] > 0])
    dummyX = np.zeros((allF.shape[0], 1))
    _, ndF = non_dominated(dummyX, allF)
    return ndF
