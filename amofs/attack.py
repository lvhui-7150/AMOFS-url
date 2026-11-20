"""Evasion attacks against the deployed detector (threat model, Section 3).

Both attacks edit *selected* features of a malicious URL towards the benign
class (feature value := benign training median) and pay the modification cost
d_i of every edited feature; an attack succeeds when the deployed detector
labels the edited vector benign within budget B.

transfer   Black-box.  The attacker knows the feature pipeline and the
           selected subset but not the model.  A surrogate (random forest) is
           trained on the attacker's own half of the training data; features
           are edited greedily in increasing order of d_i / g_i until the
           surrogate flips or the budget is exhausted; the result is replayed
           against the deployed detector.
direct     Adaptive, query-based (worst case for the defender).  At every step
           the attacker queries the deployed detector for all affordable
           single-feature edits and applies the one with the largest decrease
           in malicious probability per unit cost.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def train_detector(X_tr: np.ndarray, y_tr: np.ndarray, seed: int, max_iter: int = 200):
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1,
                                         random_state=seed, early_stopping=False)
    clf.fit(X_tr, y_tr)
    return clf


class EvasionAttacker:
    def __init__(self, Xs_tr: np.ndarray, y_tr: np.ndarray, Xs_te: np.ndarray, y_te: np.ndarray,
                 mask: np.ndarray, costs: np.ndarray, seed: int = 0,
                 max_samples: int = 1000, hgb_max_iter: int = 200):
        self.rng = np.random.default_rng(seed)
        self.sel = np.flatnonzero(mask)
        self.costs = costs[self.sel]
        Xtr, Xte = Xs_tr[:, self.sel], Xs_te[:, self.sel]
        self.deployed = train_detector(Xtr, y_tr, seed, hgb_max_iter)

        # attacker's mimicry target: benign median per selected feature
        self.target = np.median(Xtr[y_tr == 0], axis=0)

        # malicious test samples that are actually detected (attack only these)
        mal = Xte[y_te == 1]
        if mal.shape[0] > max_samples:
            mal = mal[self.rng.choice(mal.shape[0], max_samples, replace=False)]
        detected = self.deployed.predict(mal) == 1 if mal.shape[0] else np.zeros(0, dtype=bool)
        self.X0 = mal[detected]
        self.n_mal = int(mal.shape[0])
        self.clean_detection = float(detected.mean()) if mal.shape[0] else 0.0

        # surrogate on the attacker's own half of the training data
        half = self.rng.permutation(Xtr.shape[0])[: Xtr.shape[0] // 2]
        self.surrogate = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1,
                                                min_samples_leaf=2).fit(Xtr[half], y_tr[half])
        pos, neg = Xtr[half][y_tr[half] == 1], Xtr[half][y_tr[half] == 0]
        g = np.abs(pos.mean(0) - neg.mean(0)) / np.sqrt(0.5 * (pos.var(0) + neg.var(0)) + 1e-12)
        self.order = np.argsort(self.costs / (g + 1e-6), kind="stable")

    # ------------------------------------------------------------------
    def _rate(self, flipped_count: int) -> float:
        """Evasion rate over all attacked malicious samples (undetected ones count as evaded)."""
        undetected = self.n_mal - self.X0.shape[0]
        return 100.0 * (flipped_count + undetected) / max(self.n_mal, 1)

    def transfer(self, budget: float) -> float:
        if self.X0.shape[0] == 0:
            return self._rate(0)
        X = self.X0.copy()
        spent = np.zeros(X.shape[0])
        active = np.ones(X.shape[0], dtype=bool)
        for j in self.order:
            can = active & (spent + self.costs[j] <= budget)
            if not can.any():
                continue
            X[can, j] = self.target[j]
            spent[can] += self.costs[j]
            flipped = self.surrogate.predict(X[can]) == 0
            idx = np.flatnonzero(can)[flipped]
            active[idx] = False
            if not active.any():
                break
        return self._rate(int((self.deployed.predict(X) == 0).sum()))

    def direct(self, budget: float) -> float:
        if self.X0.shape[0] == 0:
            return self._rate(0)
        X = self.X0.copy()
        n, k = X.shape
        spent = np.zeros(n)
        edited = np.zeros((n, k), dtype=bool)
        done = self.deployed.predict(X) == 0
        for _ in range(k):
            active = np.flatnonzero(~done)
            if active.size == 0:
                break
            p_now = self.deployed.predict_proba(X[active])[:, 1]
            # candidate matrix: every affordable, not-yet-edited single edit
            cand = np.repeat(X[active][:, None, :], k, axis=1)            # (a, k, k)
            for j in range(k):
                cand[:, j, j] = self.target[j]
            p_new = self.deployed.predict_proba(cand.reshape(-1, k))[:, 1].reshape(active.size, k)
            gain = (p_now[:, None] - p_new) / self.costs[None, :]
            afford = (spent[active][:, None] + self.costs[None, :] <= budget) & ~edited[active]
            gain = np.where(afford, gain, -np.inf)
            best = np.argmax(gain, axis=1)
            ok = np.isfinite(gain[np.arange(active.size), best])
            if not ok.any():
                break
            rows = active[ok]
            cols = best[ok]
            X[rows, cols] = self.target[cols]
            spent[rows] += self.costs[cols]
            edited[rows, cols] = True
            done[active[~ok]] = True   # nothing affordable left; stop them
            done[rows] = self.deployed.predict(X[rows]) == 0
        return self._rate(int((self.deployed.predict(X) == 0).sum()))

    def curves(self, budgets: Sequence[float]) -> Dict[str, list]:
        return {"transfer": [self.transfer(b) for b in budgets],
                "direct": [self.direct(b) for b in budgets],
                "clean_detection": self.clean_detection}


def auec(rates: Sequence[float], budgets: Sequence[float]) -> float:
    """Normalised area under the evasion-rate curve (rates in %, result in [0, 1])."""
    r = np.asarray(rates, dtype=float) / 100.0
    b = np.asarray(budgets, dtype=float)
    if b[-1] <= b[0]:
        return float(r.mean())
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(r, b) / (b[-1] - b[0]))
