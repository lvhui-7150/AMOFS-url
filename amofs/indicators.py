"""Quality indicators: hypervolume (exact when pymoo is installed, otherwise
scrambled-Sobol quasi-Monte-Carlo with a fixed seed) and IGD.

All objectives are minimised and lie in [0, 1]; the HV reference point is
(1, ..., 1), so HV is the fraction of the unit box dominated by the front.
"""
from __future__ import annotations

import numpy as np

from .nsga_common import non_dominated_mask

try:  # exact hypervolume
    from pymoo.indicators.hv import HV as _PymooHV  # type: ignore

    _HAVE_PYMOO = True
except Exception:  # pragma: no cover
    _HAVE_PYMOO = False

try:
    from scipy.stats import qmc as _qmc  # type: ignore
except Exception:  # pragma: no cover
    _qmc = None

_QMC_CACHE: dict = {}


def _qmc_points(m: int, n_pts: int, seed: int) -> np.ndarray:
    key = (m, n_pts, seed)
    if key not in _QMC_CACHE:
        if _qmc is not None:
            sampler = _qmc.Sobol(d=m, scramble=True, seed=seed)
            pts = sampler.random_base2(int(np.ceil(np.log2(n_pts))))
        else:
            pts = np.random.default_rng(seed).random((n_pts, m))
        _QMC_CACHE[key] = pts
    return _QMC_CACHE[key]


def hypervolume(F: np.ndarray, exact: bool = True, n_qmc: int = 2 ** 15, seed: int = 0) -> float:
    """HV of front ``F`` w.r.t. reference point 1.

    ``exact=False`` is used inside the search loop (ANCF progress signal) where
    a fast, consistent estimate is enough; the reported indicator always uses
    ``exact=True`` (pymoo exact algorithm, or a 2^17-point QMC estimate when
    pymoo is unavailable).
    """
    if F.shape[0] == 0:
        return 0.0
    F = F[np.all(F <= 1.0, axis=1)]
    if F.shape[0] == 0:
        return 0.0
    F = F[non_dominated_mask(F)]
    m = F.shape[1]
    if exact and _HAVE_PYMOO:
        return float(_PymooHV(ref_point=np.ones(m))(F))
    pts = _qmc_points(m, 2 ** 17 if exact else n_qmc, seed)
    dominated = np.zeros(pts.shape[0], dtype=bool)
    for s in range(0, F.shape[0], 64):
        blk = F[s:s + 64]
        dominated |= np.any(np.all(blk[:, None, :] <= pts[None, :, :], axis=2), axis=0)
    return float(dominated.mean())


def igd(F: np.ndarray, reference_front: np.ndarray) -> float:
    if F.shape[0] == 0 or reference_front.shape[0] == 0:
        return float("inf")
    d = np.sqrt(((reference_front[:, None, :] - F[None, :, :]) ** 2).sum(2))
    return float(d.min(axis=1).mean())


def build_reference_front(fronts) -> np.ndarray:
    allF = np.vstack([f for f in fronts if f.shape[0] > 0])
    allF = np.unique(allF, axis=0)
    return allF[non_dominated_mask(allF)]


def hv_backend() -> str:
    return "pymoo-exact" if _HAVE_PYMOO else "sobol-qmc-2^17"
