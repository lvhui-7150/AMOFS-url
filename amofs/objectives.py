"""The four AMOFS objectives (all minimised, all in [0, 1]).

    f_err  = 1 - balanced accuracy of a k-NN probe on the validation split
    f_card = |x| / n
    f_rob  = 1 - log(1 + MEC(x)) / log(1 + MEC_max),
             MEC(x) = min_{i in x} d_i / (g_i + delta)          (cheap-edit surrogate)
    f_stab = 1 - mean_t Jaccard(x, xhat^(t))                    (temporal instability)

The :class:`Evaluator` pre-computes everything that does not depend on the
candidate subset so that one evaluation is a few milliseconds.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.preprocessing import StandardScaler

from .data import Dataset, Split, temporal_windows
from .probe import KnnProbe


def standardised_mean_difference(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|mu+ - mu-| / pooled std, squashed to [0, 1) by v / (1 + v)."""
    pos, neg = X[y == 1], X[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros(X.shape[1])
    v = np.abs(pos.mean(0) - neg.mean(0)) / np.sqrt(0.5 * (pos.var(0) + neg.var(0)) + 1e-12)
    return v / (1.0 + v)


def _stratified_subsample(idx: np.ndarray, y: np.ndarray, n_max: int, rng) -> np.ndarray:
    if len(idx) <= n_max:
        return idx
    out = []
    for cls in (0, 1):
        cls_idx = idx[y[idx] == cls]
        take = int(round(n_max * len(cls_idx) / len(idx)))
        take = max(1, min(take, len(cls_idx)))
        out.append(rng.choice(cls_idx, size=take, replace=False))
    return np.sort(np.concatenate(out))


class Evaluator:
    def __init__(self, ds: Dataset, split: Split, k: int = 5, probe_train: int = 2000,
                 probe_val: int = 1000, n_windows: int = 8, mec_delta: float = 0.05,
                 seed: int = 0):
        self.ds, self.split, self.k, self.delta = ds, split, k, mec_delta
        self.costs = ds.costs
        self.n = ds.n_features
        self._cache: Dict[bytes, np.ndarray] = {}
        rng = np.random.default_rng(seed)

        self.scaler = StandardScaler().fit(ds.X[split.tr])
        self.Xs = self.scaler.transform(ds.X)            # standardised, all rows
        self.y = ds.y

        # probe subsample (fixed per seed)
        self.p_tr = _stratified_subsample(split.tr, ds.y, probe_train, rng)
        self.p_va = _stratified_subsample(split.va, ds.y, probe_val, rng)
        self.Xp_tr, self.yp_tr = self.Xs[self.p_tr], ds.y[self.p_tr]
        self.Xp_va, self.yp_va = self.Xs[self.p_va], ds.y[self.p_va]
        # exact k-NN probe; AMOFS_PROBE=gpu|cpu|auto (see probe.py)
        self.probe = KnnProbe(self.Xp_tr, self.yp_tr, self.Xp_va, self.yp_va, k=self.k)

        # class separation g_i and MEC normaliser (training split only)
        self.g = standardised_mean_difference(self.Xs[split.tr], ds.y[split.tr])
        self.ratio = self.costs / (self.g + self.delta)
        self.mec_max = float(np.max(self.ratio))

        # per-window reference subsets from the development period
        self.window_refs: List[np.ndarray] = []
        for w in temporal_windows(split.dev, n_windows):
            if len(np.unique(ds.y[w])) < 2:
                continue
            g_w = standardised_mean_difference(self.Xs[w], ds.y[w])
            self.window_refs.append(np.flatnonzero(g_w >= np.median(g_w)))
        self._ref_masks = np.zeros((len(self.window_refs), self.n), dtype=bool)
        for r, ref in enumerate(self.window_refs):
            self._ref_masks[r, ref] = True

    # ------------------------------------------------------------------
    def f_err(self, mask: np.ndarray) -> float:
        return self.probe.f_err(mask)

    def f_card(self, mask: np.ndarray) -> float:
        return float(mask.sum()) / self.n

    def mec(self, mask: np.ndarray) -> float:
        sel = np.flatnonzero(mask)
        return float(np.min(self.ratio[sel])) if sel.size else 0.0

    def f_rob(self, mask: np.ndarray) -> float:
        m = self.mec(mask)
        return float(1.0 - np.log1p(m) / np.log1p(self.mec_max))

    def f_stab(self, mask: np.ndarray) -> float:
        if not self.window_refs:
            return 0.0
        m = mask.astype(bool)
        inter = (self._ref_masks & m).sum(1)
        union = (self._ref_masks | m).sum(1)
        jacc = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        return float(1.0 - jacc.mean())

    def evaluate(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=np.int8)
        key = mask.tobytes()
        v = self._cache.get(key)
        if v is None:
            v = np.array([self.f_err(mask), self.f_card(mask), self.f_rob(mask), self.f_stab(mask)])
            self._cache[key] = v
        return v.copy()

    def evaluate_population(self, masks: np.ndarray) -> np.ndarray:
        return np.vstack([self.evaluate(m) for m in masks])

    @property
    def n_evaluations(self) -> int:
        return len(self._cache)


class ObjectiveDropEvaluator:
    """Zero out selected objectives during search (ablation of f_rob / f_stab)."""

    def __init__(self, base: Evaluator, disabled):
        self.base, self.disabled = base, tuple(disabled)
        self.n = base.n

    def evaluate(self, mask):
        v = self.base.evaluate(mask)
        v[list(self.disabled)] = 0.0
        return v

    def evaluate_population(self, masks):
        return np.vstack([self.evaluate(m) for m in masks])

    def __getattr__(self, item):  # delegate everything else
        if item in ("base", "disabled", "n"):
            raise AttributeError(item)
        return getattr(self.base, item)
