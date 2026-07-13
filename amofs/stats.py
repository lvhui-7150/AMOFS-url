"""Statistical testing: Wilcoxon signed-rank + Benjamini-Hochberg + effect sizes.

Implements the procedure described in the manuscript: for each metric, AMOFS is
compared against each baseline over the R paired runs with a two-sided Wilcoxon
signed-rank test; p-values across baselines are corrected for the false
discovery rate with Benjamini-Hochberg at level 0.05; effect size is reported as
the rank-biserial correlation and the Vargha-Delaney A12.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for paired samples x, y."""
    d = x - y
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total else 0.0


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """Probability that a random x exceeds a random y (0.5 = no effect)."""
    nx, ny = len(x), len(y)
    greater = sum(1.0 for xi in x for yj in y if xi > yj)
    equal = sum(0.5 for xi in x for yj in y if xi == yj)
    return float((greater + equal) / (nx * ny))


def benjamini_hochberg(pvals: List[float], alpha: float = 0.05):
    """Return BH-adjusted p-values and reject flags."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * n / (rank + 1)
        prev = min(prev, val)
        adj[idx] = prev
    reject = adj <= alpha
    return adj, reject


def compare_against_baselines(amofs_runs: np.ndarray,
                              baseline_runs: Dict[str, np.ndarray],
                              higher_is_better: bool = True,
                              alpha: float = 0.05) -> Dict[str, dict]:
    """Compare AMOFS vs each baseline on one metric across paired runs.

    ``amofs_runs`` and each entry of ``baseline_runs`` are length-R arrays.
    Returns a per-baseline dict with raw p, adjusted p, rank-biserial r, A12.
    """
    names = list(baseline_runs.keys())
    raw_p, rrb, a12 = [], [], []
    for name in names:
        b = baseline_runs[name]
        try:
            stat, p = wilcoxon(amofs_runs, b, zero_method="wilcox",
                               alternative="two-sided")
        except ValueError:
            p = 1.0
        raw_p.append(float(p))
        rrb.append(rank_biserial(amofs_runs, b))
        a12.append(vargha_delaney_a12(amofs_runs, b))
    adj_p, reject = benjamini_hochberg(raw_p, alpha)

    out = {}
    for i, name in enumerate(names):
        out[name] = dict(p_raw=raw_p[i], p_adj=float(adj_p[i]),
                         reject=bool(reject[i]), rank_biserial=rrb[i],
                         a12=a12[i])
    return out