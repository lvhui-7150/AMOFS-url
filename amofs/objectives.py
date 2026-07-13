"""The four AMOFS objectives, all minimised and bounded in [0, 1].

Implements, in the notation of the manuscript:
    f_err  (Eq. classification error)   -- 1 - balanced accuracy of a kNN probe
    f_card (Eq. cardinality)            -- fraction of features selected
    f_rob  (Eq. robustness / MEC)       -- 1 - normalised minimum evasion cost
    f_stab (Eq. temporal instability)   -- 1 - mean Jaccard overlap vs per-window optima

An :class:`Evaluator` precomputes everything that does not depend on the
candidate subset (standardisation, per-window reference subsets) so that a
single subset evaluation is cheap inside the evolutionary loop.
"""
from __future__ import annotations

from typing import List, Optional
from typing import Dict

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from .data import Dataset, make_windows


def _balanced_accuracy_knn(Xtr, ytr, Xva, yva, mask, k) -> float:
    sel = np.flatnonzero(mask)
    if sel.size == 0:
        return 0.5  # no features -> chance-level balanced accuracy
    kk = min(k, len(ytr))
    Xtr_sel = Xtr[:, sel]
    Xva_sel = Xva[:, sel]
    tr_norm = np.sum(Xtr_sel * Xtr_sel, axis=1)
    va_norm = np.sum(Xva_sel * Xva_sel, axis=1)
    dist = va_norm[:, None] + tr_norm[None, :] - 2.0 * Xva_sel @ Xtr_sel.T
    nn = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    votes = ytr[nn].mean(axis=1)
    pred = (votes >= 0.5).astype(int)
    return float(balanced_accuracy_score(yva, pred))


def _abs_standardised_mean_diff(X, y) -> np.ndarray:
    """Per-feature |standardised mean difference| between classes in [0, ~].

    Equivalent to the point-biserial separation; used as the discriminative
    weight g_i in the MEC surrogate. Returns one value per feature.
    """
    pos = X[y == 1]
    neg = X[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros(X.shape[1])
    mean_diff = np.abs(pos.mean(axis=0) - neg.mean(axis=0))
    pooled_std = np.sqrt(0.5 * (pos.var(axis=0) + neg.var(axis=0)) + 1e-12)
    g = mean_diff / pooled_std
    # squash into [0, 1] so the MEC ratio is well-scaled
    return g / (1.0 + g)


class Evaluator:
    """Evaluates the four-objective vector F(x) for binary masks x."""

    def __init__(self, ds: Dataset, X_tr, y_tr, X_va, y_va,
                 k: int = 5, n_windows: int = 5, mec_delta: float = 1e-3):
        self.ds = ds
        self.k = k
        self.delta = mec_delta
        self.costs = ds.costs
        self._cache: Dict[bytes, np.ndarray] = {}

        # standardise on the training split, apply to validation
        self.scaler = StandardScaler().fit(X_tr)
        self.X_tr = self.scaler.transform(X_tr)
        self.X_va = self.scaler.transform(X_va)
        self.y_tr = y_tr
        self.y_va = y_va

        # discriminative weights g_i (subset-independent first-order surrogate)
        self.g = _abs_standardised_mean_diff(self.X_tr, self.y_tr)

        # MEC normaliser: max_i d_i / delta  (Eq. obj-rob)
        self.mec_max = float(np.max(self.costs) / self.delta)

        # malicious validation rows used in the MEC expectation
        self.mal_idx = np.flatnonzero(self.y_va == 1)

        # per-window reference subsets for the stability objective:
        # select features whose per-window discriminative weight is above median
        self.window_refs = self._per_window_reference_subsets(ds, n_windows)

    # -- per-objective -------------------------------------------------------

    def f_err(self, mask: np.ndarray) -> float:
        bacc = _balanced_accuracy_knn(self.X_tr, self.y_tr,
                                      self.X_va, self.y_va, mask, self.k)
        return 1.0 - bacc

    def f_card(self, mask: np.ndarray) -> float:
        return float(mask.mean())

    def f_rob(self, mask: np.ndarray) -> float:
        sel = np.flatnonzero(mask)
        if sel.size == 0:
            return 1.0  # nothing selected -> trivially evadable -> worst
        # cheapest single-feature evasion per malicious URL: min_i d_i/(g_i+delta)
        ratio = self.costs[sel] / (self.g[sel] + self.delta)  # (|sel|,)
        cheapest = float(np.min(ratio))
        # MEC(x) is the mean over malicious URLs; with a subset-level surrogate
        # the per-URL min collapses to the subset min, so MEC = cheapest.
        mec = cheapest
        return float(1.0 - mec / self.mec_max)

    def f_stab(self, mask: np.ndarray) -> float:
        if not self.window_refs:
            return 0.0
        supp = set(np.flatnonzero(mask).tolist())
        jacc = []
        for ref in self.window_refs:
            rset = set(ref.tolist())
            union = supp | rset
            if not union:
                jacc.append(1.0)
            else:
                jacc.append(len(supp & rset) / len(union))
        return float(1.0 - np.mean(jacc))

    def evaluate(self, mask: np.ndarray) -> np.ndarray:
        """Return F(x) = (f_err, f_card, f_rob, f_stab)."""
        key = np.asarray(mask, dtype=np.int8).tobytes()
        if key not in self._cache:
            self._cache[key] = np.array([self.f_err(mask), self.f_card(mask),
                                         self.f_rob(mask), self.f_stab(mask)], dtype=float)
        return self._cache[key].copy()

    def evaluate_population(self, masks: np.ndarray) -> np.ndarray:
        return np.vstack([self.evaluate(m) for m in masks])

    # -- helpers -------------------------------------------------------------

    def _per_window_reference_subsets(self, ds: Dataset,
                                      n_windows: int) -> List[np.ndarray]:
        refs: List[np.ndarray] = []
        windows = make_windows(ds, n_windows)
        Xall = self.scaler.transform(ds.X)
        for w in windows:
            if len(w) < 4:
                continue
            g_w = _abs_standardised_mean_diff(Xall[w], ds.y[w])
            thresh = np.median(g_w)
            ref = np.flatnonzero(g_w >= thresh)
            refs.append(ref)
        return refs
