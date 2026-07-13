"""Black-box transfer evasion attack (threat model, Section 3).

The attacker trains a surrogate detector on an independent split, searches for
the cheapest feature edits (greedy on d_i / g_i, then local search) that flip
the surrogate subject to a total budget B, and replays the perturbed sample
against the deployed detector. The evasion rate at budget B is the fraction of
test malicious samples misclassified as benign after attack.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def train_deployed_detector(X_tr, y_tr, mask):
    sel = np.flatnonzero(mask)
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(X_tr[:, sel], y_tr)
    return clf, sel


def _benign_target(X_tr, y_tr, sel):
    """Mean benign feature vector on selected coords (attack target direction)."""
    benign = X_tr[y_tr == 0][:, sel]
    return benign.mean(axis=0)


def evasion_rate(X_tr, y_tr, X_te, y_te, mask, costs,
                 budget: float, surrogate_split: float = 0.5,
                 seed: int = 0) -> float:
    """Fraction of malicious test samples flipped to benign within ``budget``."""
    rng = np.random.default_rng(seed)
    sel = np.flatnonzero(mask)
    if sel.size == 0:
        return 1.0

    # deployed detector
    deployed, _ = train_deployed_detector(X_tr, y_tr, mask)

    # surrogate trained on an independent half of the training data
    n = X_tr.shape[0]
    perm = rng.permutation(n)
    cut = int(n * surrogate_split)
    s_idx = perm[:cut]
    surro = KNeighborsClassifier(n_neighbors=5)
    surro.fit(X_tr[s_idx][:, sel], y_tr[s_idx])

    benign_dir = _benign_target(X_tr, y_tr, sel)
    g = np.abs(X_tr[y_tr == 1][:, sel].mean(axis=0)
               - X_tr[y_tr == 0][:, sel].mean(axis=0)) + 1e-6
    csel = costs[sel]
    order = np.argsort(csel / g)  # cheapest, most-effective edits first

    mal = X_te[y_te == 1]
    if mal.shape[0] == 0:
        return 0.0

    flipped = 0
    for x in mal:
        xv = x[sel].copy()
        spent = 0.0
        # greedy edits towards the benign centroid, cheapest-effective first
        for j in order:
            if spent + csel[j] > budget:
                continue
            xv[j] = benign_dir[j]
            spent += csel[j]
            if surro.predict(xv.reshape(1, -1))[0] == 0:
                break
        # replay against deployed detector
        if deployed.predict(xv.reshape(1, -1))[0] == 0:
            flipped += 1
    return 100.0 * flipped / mal.shape[0]


def evasion_curve(X_tr, y_tr, X_te, y_te, mask, costs,
                  budgets, seed: int = 0) -> np.ndarray:
    return np.array([evasion_rate(X_tr, y_tr, X_te, y_te, mask, costs, b,
                                  seed=seed) for b in budgets])