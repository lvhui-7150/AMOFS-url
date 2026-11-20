"""Paired statistics: Wilcoxon signed-rank, Benjamini-Hochberg FDR control,
matched-pairs rank-biserial correlation, Vargha-Delaney A12, Friedman test."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    r_plus, r_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    tot = r_plus + r_minus
    return float((r_plus - r_minus) / tot) if tot else 0.0


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def benjamini_hochberg(pvals: List[float], alpha: float = 0.05):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * n / (rank + 1))
        adj[idx] = prev
    return adj, adj <= alpha


def compare_against_baselines(amofs: np.ndarray, baselines: Dict[str, np.ndarray],
                              higher_is_better: bool = True, alpha: float = 0.05) -> Dict[str, dict]:
    names = list(baselines)
    raw, rrb, a12 = [], [], []
    for n in names:
        b = np.asarray(baselines[n], dtype=float)
        a = np.asarray(amofs, dtype=float)
        if not higher_is_better:
            a, b = -a, -b
        try:
            p = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:  # all differences zero
            p = 1.0
        raw.append(p)
        rrb.append(rank_biserial(a, b))
        a12.append(vargha_delaney_a12(a, b))
    adj, rej = benjamini_hochberg(raw, alpha)
    return {n: dict(p_raw=raw[i], p_adj=float(adj[i]), reject=bool(rej[i]),
                    rank_biserial=rrb[i], a12=a12[i],
                    mean_diff=float(np.mean(np.asarray(amofs) - np.asarray(baselines[n]))))
            for i, n in enumerate(names)}


def friedman(metric_by_method: Dict[str, np.ndarray]) -> dict:
    arrays = [np.asarray(v, dtype=float) for v in metric_by_method.values()]
    if len(arrays) < 3:
        return dict(statistic=float("nan"), p=float("nan"))
    try:
        res = friedmanchisquare(*arrays)
        return dict(statistic=float(res.statistic), p=float(res.pvalue))
    except ValueError:
        return dict(statistic=float("nan"), p=float("nan"))
