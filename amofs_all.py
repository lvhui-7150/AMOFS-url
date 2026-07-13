"""
AMOFS: Adversary-aware Multi-Objective Feature Selection
========================================================

Single-file reference implementation. All functionality from
Usage:
    python amofs_all.py --dataset data/processed/urlhaus_majestic.csv --name URLhausMajestic




"""

from __future__ import annotations

import argparse
import csv
import io
import ipaddress
import json
import math
import os
import re
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wilcoxon

__version__ = "0.1.0"


# --- source: config.py ---

@dataclass
class AMOFSParams:
    """AMOFS-specific parameters (ANCF + AGDM)."""
    alpha: float = 2.0          # ANCF generation-decay shape
    beta: float = 1.5           # ANCF hypervolume-progress shape
    pc_min: float = 0.6
    pc_max: float = 0.95
    eta_min: float = 0.02       # lower bound on per-bit mutation scale (>0)
    eta_max: float = 0.4
    lam: float = 0.5            # AGDM mixing weight between s_i and c_i
    eps: float = 0.05           # AGDM probability clip (0 < eps < 1/2)
    use_ancf: bool = True       # ablation switch: adaptive HV-aware schedule
    use_agdm: bool = True       # ablation switch: archive-guided mutation


@dataclass
class ExperimentConfig:
    """Top-level configuration for one experimental run."""
    # search budget
    pop_size: int = 100
    generations: int = 200
    n_runs: int = 30            # independent seeds (paper uses R=30)
    ablation_runs: int = 30     # paper ablation also reports mean over R runs
    base_seed: int = 20240101

    # objectives
    knn_k: int = 5              # probe classifier neighbours
    n_windows: int = 5          # temporal windows T for the stability objective
    mec_delta: float = 0.05     # guard term delta in the MEC surrogate

    # evasion attack budgets (cost units); five increasing levels
    attack_budgets: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

    # data split
    test_size: float = 0.20
    val_size: float = 0.20      # taken from the non-test remainder

    amofs: AMOFSParams = field(default_factory=AMOFSParams)




# --- source: features.py ---

SUSPICIOUS_TOKENS = (
    "login", "verify", "update", "secure", "account", "bank", "free",
    "bonus", "paypal", "signin", "wp-admin", "download", "invoice",
    "confirm", "password", "wallet", "crypto", "gift", "prize",
)
SHORTENER_TOKENS = (
    "bit.ly", "goo.gl", "tinyurl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "bitly.com",
)
EXECUTABLE_SUFFIXES = (
    ".exe", ".scr", ".dll", ".bat", ".cmd", ".js", ".jar", ".zip",
    ".rar", ".7z", ".apk", ".msi", ".bin", ".sh",
)


def _normalise_url(raw_url: str) -> str:
    url = str(raw_url).strip()
    if not url:
        return "http://"
    if "://" not in url:
        url = "http://" + url
    return url


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _is_ip_address(host: str) -> int:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return 1
    except ValueError:
        return 0


def _has_valid_port(parsed) -> int:
    try:
        return int(parsed.port is not None)
    except ValueError:
        return 0


def _longest_run(text: str) -> int:
    if not text:
        return 0
    longest = current = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
            prev = ch
    return longest


def _tokens(parts: Iterable[str]) -> List[str]:
    joined = " ".join(part for part in parts if part)
    return [token for token in re.split(r"[^A-Za-z0-9]+", joined) if token]


def extract_url_features(raw_url: str) -> Dict[str, float]:
    """Return deterministic numeric features for one URL."""
    url = _normalise_url(raw_url)
    parsed = urlparse(url)
    decoded_url = unquote(url)

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    host_parts = [part for part in host.split(".") if part]
    subdomain_count = max(0, len(host_parts) - 2) if not _is_ip_address(host) else 0
    all_tokens = _tokens([host, path, query])
    token_lengths = [len(token) for token in all_tokens]

    length = len(url)
    alpha = sum(ch.isalpha() for ch in url)
    digit = sum(ch.isdigit() for ch in url)
    special = sum(not ch.isalnum() for ch in url)
    hex_tokens = sum(1 for token in all_tokens if len(token) >= 8 and re.fullmatch(r"[0-9a-fA-F]+", token))
    digit_tokens = sum(1 for token in all_tokens if any(ch.isdigit() for ch in token))
    params = parse_qsl(query, keep_blank_values=True)

    lower_url = decoded_url.lower()
    suspicious_count = sum(1 for token in SUSPICIOUS_TOKENS if token in lower_url)
    shortener = int(any(token in host for token in SHORTENER_TOKENS))
    executable = int(any(path.lower().endswith(suffix) for suffix in EXECUTABLE_SUFFIXES))

    return {
        "url_len": float(length),
        "url_entropy": _entropy(url),
        "url_digit_ratio": digit / max(length, 1),
        "url_alpha_ratio": alpha / max(length, 1),
        "url_special_count": float(special),
        "dot_count": float(url.count(".")),
        "slash_count": float(url.count("/")),
        "hyphen_count": float(url.count("-")),
        "at_count": float(url.count("@")),
        "question_count": float(url.count("?")),
        "equal_count": float(url.count("=")),
        "amp_count": float(url.count("&")),
        "percent_count": float(url.count("%")),
        "encoded_char_count": float(len(re.findall(r"%[0-9a-fA-F]{2}", url))),
        "host_len": float(len(host)),
        "host_entropy": _entropy(host),
        "host_dot_count": float(host.count(".")),
        "subdomain_count": float(subdomain_count),
        "tld_len": float(len(tld)),
        "domain_token_count": float(len(host_parts)),
        "has_ip_host": float(_is_ip_address(host)),
        "has_https": float(parsed.scheme.lower() == "https"),
        "has_port": float(_has_valid_port(parsed) if host else 0),
        "punycode_host": float("xn--" in host),
        "www_prefix": float(host.startswith("www.")),
        "path_len": float(len(path)),
        "path_entropy": _entropy(path),
        "path_depth": float(len([part for part in path.split("/") if part])),
        "query_len": float(len(query)),
        "query_param_count": float(len(params)),
        "suspicious_token_count": float(suspicious_count),
        "shortener_token": float(shortener),
        "executable_suffix": float(executable),
        "longest_token_len": float(max(token_lengths) if token_lengths else 0),
        "avg_token_len": float(sum(token_lengths) / len(token_lengths) if token_lengths else 0),
        "repeated_char_run": float(_longest_run(url)),
        "digit_token_count": float(digit_tokens),
        "hexadecimal_token_count": float(hex_tokens),
    }


def feature_names() -> List[str]:
    """Return feature names in stable extraction order."""
    return list(extract_url_features("http://example.com/path?a=1").keys())

# --- source: data.py ---

# ----------------------------------------------------------------------------
# Cost model (threat model, Section 3)
# ----------------------------------------------------------------------------
# Ordinal modification-cost scale. Lexical features the attacker fully controls
# are cheap; host-configuration features are moderate; registration/reputation
# features are expensive or practically immutable.
COST_CHEAP = 1.0
COST_MODERATE = 3.0
COST_EXPENSIVE = 8.0

# Substring heuristics mapping common URL-feature names to a cost tier. Extend
# this for your concrete feature pipeline. Unknown features default to moderate.
_CHEAP_HINTS = ("len", "count", "ratio", "entropy", "digit", "letter",
                "token", "dot", "slash", "hyphen", "query", "param", "path")
_EXPENSIVE_HINTS = ("age", "whois", "registration", "asn", "reputation",
                    "rank", "ssl", "cert", "dns_ttl", "alexa")
_MODERATE_HINTS = ("subdomain", "host", "ttl", "ip", "port", "redirect")

_EXPLICIT_COSTS = {
    "has_ip_host": COST_EXPENSIVE,
    "host_entropy": COST_EXPENSIVE,
    "domain_token_count": COST_EXPENSIVE,
    "tld_len": COST_EXPENSIVE,
    "punycode_host": COST_EXPENSIVE,
    "shortener_token": COST_EXPENSIVE,
    "host_len": COST_MODERATE,
    "host_dot_count": COST_MODERATE,
    "subdomain_count": COST_MODERATE,
    "has_https": COST_MODERATE,
    "has_port": COST_MODERATE,
    "path_depth": COST_CHEAP,
    "path_len": COST_CHEAP,
    "query_len": COST_CHEAP,
    "query_param_count": COST_CHEAP,
}


def assign_costs(feature_names: List[str]) -> np.ndarray:
    """Return a cost vector d_i (one per feature) on the ordinal scale."""
    costs = np.empty(len(feature_names), dtype=float)
    for i, name in enumerate(feature_names):
        low = name.lower()
        if low in _EXPLICIT_COSTS:
            costs[i] = _EXPLICIT_COSTS[low]
        elif any(h in low for h in _EXPENSIVE_HINTS):
            costs[i] = COST_EXPENSIVE
        elif any(h in low for h in _MODERATE_HINTS):
            costs[i] = COST_MODERATE
        elif any(h in low for h in _CHEAP_HINTS):
            costs[i] = COST_CHEAP
        else:
            costs[i] = COST_MODERATE
    return costs


@dataclass
class Dataset:
    """A loaded, split-ready dataset."""
    name: str
    X: np.ndarray                 # (n_samples, n_features), standardised later
    y: np.ndarray                 # (n_samples,) in {0,1}
    feature_names: List[str]
    costs: np.ndarray             # (n_features,) modification costs d_i
    timestamps: Optional[np.ndarray] = None  # (n_samples,) sortable, or None

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def load_dataset(csv_path: str, name: str,
                 label_col: str = "label",
                 timestamp_col: str = "timestamp") -> Dataset:
    """Load a feature CSV into a :class:`Dataset`."""
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"{csv_path}: missing required '{label_col}' column")

    ts = None
    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce").astype("int64").to_numpy()
        df = df.drop(columns=[timestamp_col])

    y = df[label_col].astype(int).to_numpy()
    df = df.drop(columns=[label_col])

    # keep only numeric feature columns
    df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    feature_names = list(df.columns)
    X = df.to_numpy(dtype=float)
    costs = assign_costs(feature_names)
    return Dataset(name=name, X=X, y=y, feature_names=feature_names,
                   costs=costs, timestamps=ts)




























































def make_windows(ds: Dataset, n_windows: int) -> List[np.ndarray]:
    """Split sample indices into ``n_windows`` equal-frequency temporal windows.

    Uses ``timestamps`` if present, otherwise row order. Returns a list of index
    arrays, oldest window first.
    """
    n = len(ds.y)
    if ds.timestamps is not None:
        order = np.argsort(ds.timestamps, kind="stable")
    else:
        order = np.arange(n)
    return [w for w in np.array_split(order, n_windows)]

# --- source: indicators.py ---

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

# --- source: nsga_common.py ---

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff a (weakly) dominates b and is strictly better in >=1 objective."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> List[np.ndarray]:
    """Return a list of fronts; each front is an array of row indices into F."""
    n = F.shape[0]
    S = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return [np.array(f, dtype=int) for f in fronts]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Crowding distance for a set of objective vectors (NSGA-II)."""
    n, m = F.shape
    if n == 0:
        return np.array([])
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j])
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = F[order[0], j], F[order[-1], j]
        span = fmax - fmin
        if span <= 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (F[order[k + 1], j] - F[order[k - 1], j]) / span
    return dist


def normalised_crowding(F: np.ndarray) -> np.ndarray:
    """Crowding distance squashed into [0, 1] (used by AGDM)."""
    cd = crowding_distance(F)
    finite = cd[np.isfinite(cd)]
    hi = finite.max() if finite.size else 1.0
    cd = np.where(np.isinf(cd), 1.0, cd / (hi + 1e-12))
    return cd


def environmental_selection(X: np.ndarray, F: np.ndarray,
                            n_keep: int) -> Tuple[np.ndarray, np.ndarray]:
    """NSGA-II environmental selection: fill by fronts, break ties by crowding."""
    fronts = fast_non_dominated_sort(F)
    chosen: List[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= n_keep:
            chosen.extend(front.tolist())
        else:
            cd = crowding_distance(F[front])
            order = front[np.argsort(-cd)]
            need = n_keep - len(chosen)
            chosen.extend(order[:need].tolist())
            break
    idx = np.array(chosen, dtype=int)
    return X[idx], F[idx]


def non_dominated(X: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the non-dominated subset (first front) of (X, F)."""
    if F.shape[0] == 0:
        return X, F
    fronts = fast_non_dominated_sort(F)
    idx = fronts[0]
    return X[idx], F[idx]


def archive_update(arch_X: np.ndarray, arch_F: np.ndarray,
                   new_X: np.ndarray, new_F: np.ndarray,
                   cap: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Merge new solutions into the archive, keep non-dominated, dedupe, cap."""
    if arch_X.size == 0:
        X, F = new_X, new_F
    else:
        X = np.vstack([arch_X, new_X])
        F = np.vstack([arch_F, new_F])
    # dedupe identical masks
    _, uniq = np.unique(X, axis=0, return_index=True)
    X, F = X[np.sort(uniq)], F[np.sort(uniq)]
    X, F = non_dominated(X, F)
    if X.shape[0] > cap:  # keep the most diverse by crowding
        cd = crowding_distance(F)
        keep = np.argsort(-cd)[:cap]
        X, F = X[keep], F[keep]
    return X, F

# --- source: objectives.py ---

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

# --- source: amofs.py ---

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

# --- source: baselines.py ---

def _init_pop(n, d, rng, p=0.3):
    P = (rng.random((n, d)) < p).astype(np.int8)
    P[P.sum(axis=1) == 0, 0] = 1
    return P


def _bitflip(child, rate, rng):
    flip = rng.random(child.shape[0]) < rate
    out = child.copy()
    out[flip] ^= 1
    if out.sum() == 0:
        out[rng.integers(0, child.shape[0])] = 1
    return out


def _tournament(F, rng):
    i, j = rng.integers(0, F.shape[0], size=2)
    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
        return int(i)
    return int(j)


def _uniform_xover(p1, p2, pc, rng):
    if rng.random() > pc:
        return p1.copy()
    swap = rng.random(p1.shape[0]) < 0.5
    return np.where(swap, p1, p2).astype(np.int8)


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------
def run_nsga2(ev, d, pop_size, generations, seed=0, pc=0.9):
    rng = np.random.default_rng(seed)
    P = _init_pop(pop_size, d, rng)
    FP = ev.evaluate_population(P)
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), P, FP)
    hv_log = []
    rate = 1.0 / d
    for _ in range(generations):
        Q = np.empty_like(P)
        for j in range(pop_size):
            i1, i2 = _tournament(FP, rng), _tournament(FP, rng)
            child = _uniform_xover(P[i1], P[i2], pc, rng)
            Q[j] = _bitflip(child, rate, rng)
        FQ = ev.evaluate_population(Q)
        XU, FU = np.vstack([P, Q]), np.vstack([FP, FQ])
        P, FP = environmental_selection(XU, FU, pop_size)
        aX, aF = archive_update(aX, aF, P, FP)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


# ---------------------------------------------------------------------------
# SPEA2 (strength-based fitness, archive truncation)
# ---------------------------------------------------------------------------
def _spea2_fitness(F):
    n = F.shape[0]
    dom = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                dom[i, j] = True
    strength = dom.sum(axis=1)
    raw = np.array([strength[dom[:, j]].sum() for j in range(n)], dtype=float)
    # density via k-th nearest neighbour
    k = int(np.sqrt(n)) or 1
    dens = np.zeros(n)
    for i in range(n):
        dists = np.sort(np.sqrt(((F - F[i]) ** 2).sum(axis=1)))
        kth = dists[min(k, n - 1)]
        dens[i] = 1.0 / (kth + 2.0)
    return raw + dens


def run_spea2(ev, d, pop_size, generations, seed=0, pc=0.9):
    rng = np.random.default_rng(seed)
    P = _init_pop(pop_size, d, rng)
    FP = ev.evaluate_population(P)
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), P, FP)
    hv_log = []
    rate = 1.0 / d
    for _ in range(generations):
        fit = _spea2_fitness(FP)
        Q = np.empty_like(P)
        for j in range(pop_size):
            i1, i2 = rng.integers(0, pop_size, 2)
            parent_a = i1 if fit[i1] < fit[i2] else i2
            i3, i4 = rng.integers(0, pop_size, 2)
            parent_b = i3 if fit[i3] < fit[i4] else i4
            child = _uniform_xover(P[parent_a], P[parent_b], pc, rng)
            Q[j] = _bitflip(child, rate, rng)
        FQ = ev.evaluate_population(Q)
        XU, FU = np.vstack([P, Q]), np.vstack([FP, FQ])
        P, FP = environmental_selection(XU, FU, pop_size)
        aX, aF = archive_update(aX, aF, P, FP)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


# ---------------------------------------------------------------------------
# MOEA/D (Tchebycheff decomposition)
# ---------------------------------------------------------------------------
def run_moead(ev, d, pop_size, generations, seed=0, n_neighbors=10):
    rng = np.random.default_rng(seed)
    m = 4
    # weight vectors on the simplex (random; adequate for m=4)
    W = rng.random((pop_size, m))
    W /= W.sum(axis=1, keepdims=True)
    W = np.clip(W, 1e-3, None)
    # neighbourhoods
    dist = np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2).sum(axis=2))
    B = np.argsort(dist, axis=1)[:, :n_neighbors]

    P = _init_pop(pop_size, d, rng)
    FP = ev.evaluate_population(P)
    z = FP.min(axis=0)
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), P, FP)
    hv_log = []
    rate = 1.0 / d

    def tcheby(fv, w):
        return np.max(w * np.abs(fv - z))

    for _ in range(generations):
        for i in range(pop_size):
            k, l = rng.choice(B[i], size=2, replace=True)
            child = _uniform_xover(P[k], P[l], 0.9, rng)
            child = _bitflip(child, rate, rng)
            fc = ev.evaluate(child)
            z = np.minimum(z, fc)
            for nb in B[i]:
                if tcheby(fc, W[nb]) <= tcheby(FP[nb], W[nb]):
                    P[nb] = child
                    FP[nb] = fc
        aX, aF = archive_update(aX, aF, P, FP)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


# ---------------------------------------------------------------------------
# Single-population baselines on a tuned scalarisation
# ---------------------------------------------------------------------------
def _tune_weights(ev: Evaluator, d, rng, n_trials=20):
    """Pick scalarisation weights maximising validation BACC - lambda*card."""
    best_w, best_score = None, -np.inf
    for _ in range(n_trials):
        w = rng.random(4)
        w /= w.sum()
        mask = (rng.random(d) < 0.5).astype(np.int8)
        f = ev.evaluate(mask)
        score = -np.dot(w, f)
        if score > best_score:
            best_score, best_w = score, w
    return best_w


def run_bpso(ev, d, pop_size, generations, seed=0):
    rng = np.random.default_rng(seed)
    w = _tune_weights(ev, d, rng)
    # binary PSO with sigmoid transfer
    Xpos = rng.random((pop_size, d))
    V = rng.normal(scale=1.0, size=(pop_size, d))
    Xbin = (Xpos > 0.5).astype(np.int8)
    Xbin[Xbin.sum(axis=1) == 0, 0] = 1
    F = ev.evaluate_population(Xbin)
    scal = F @ w
    pbest, pbest_f = Xbin.copy(), scal.copy()
    g = int(np.argmin(scal))
    gbest = Xbin[g].copy()
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), Xbin, F)
    hv_log = []
    c1 = c2 = 1.5
    for _ in range(generations):
        r1, r2 = rng.random((pop_size, d)), rng.random((pop_size, d))
        V = 0.7 * V + c1 * r1 * (pbest - Xbin) + c2 * r2 * (gbest - Xbin)
        prob = 1.0 / (1.0 + np.exp(-V))
        Xbin = (rng.random((pop_size, d)) < prob).astype(np.int8)
        Xbin[Xbin.sum(axis=1) == 0, 0] = 1
        F = ev.evaluate_population(Xbin)
        scal = F @ w
        improved = scal < pbest_f
        pbest[improved], pbest_f[improved] = Xbin[improved], scal[improved]
        g = int(np.argmin(pbest_f))
        gbest = pbest[g].copy()
        aX, aF = archive_update(aX, aF, Xbin, F)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


def run_bgwo(ev, d, pop_size, generations, seed=0):
    rng = np.random.default_rng(seed)
    w = _tune_weights(ev, d, rng)
    X = _init_pop(pop_size, d, rng)
    F = ev.evaluate_population(X)
    scal = F @ w
    order = np.argsort(scal)
    alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    aX, aF = archive_update(np.empty((0, d), np.int8), np.empty((0, 4)), X, F)
    hv_log = []
    for t in range(generations):
        a = 2 - 2 * t / generations
        for i in range(pop_size):
            new = np.zeros(d, dtype=np.int8)
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(d), rng.random(d)
                A = 2 * a * r1 - a
                Dist = np.abs(2 * r2 * leader - X[i])
                cand = leader - A * Dist
                # binary transfer
                prob = 1.0 / (1.0 + np.exp(-cand))
                new = new | (rng.random(d) < prob).astype(np.int8)
            if new.sum() == 0:
                new[rng.integers(0, d)] = 1
            X[i] = new
        F = ev.evaluate_population(X)
        scal = F @ w
        order = np.argsort(scal)
        alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
        aX, aF = archive_update(aX, aF, X, F)
        hv_log.append(hypervolume(aF))
    return aX, aF, hv_log


BASELINES: dict = {
    "NSGA-II": run_nsga2,
    "MOEA/D": run_moead,
    "SPEA2": run_spea2,
    "BPSO": run_bpso,
    "BGWO": run_bgwo,
}

# --- source: attack.py ---

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

# --- source: stats.py ---

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

# --- source: py ---

r"""Emit LaTeX table fragments that drop into the manuscript.

Each emitter writes a standalone tabular body that can be ``\input`` into the
corresponding table in ``sections/07-experiments.tex``, replacing ``\TBD``
cells. The best value per column is bolded automatically.
"""



ROW_END = " " + chr(92) * 2


def _fmt(x, prec=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{prec}f}"


def _fmt_pm(mean, std, prec=3):
    return f"{_fmt(mean, prec)}\\,$\\pm$\\,{_fmt(std, prec)}"


def _bold_best(values, higher_is_better):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if np.all(np.isnan(arr)):
        return [False] * len(values)
    best = np.nanmax(arr) if higher_is_better else np.nanmin(arr)
    return [abs(v - best) < 1e-9 if not np.isnan(v) else False for v in arr]


def _method_label(method: str) -> str:
    return f"\\textbf{{{method}}}" if method == "AMOFS" else method


def _write(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body + "\n")
    print(f"[export] wrote {path}")


def main_comparison_table(methods: List[str], stats: Dict[str, dict], out_path: str):
    """Paper-compatible main table: HV, IGD, BACC, F1."""
    cols: Sequence[Tuple[str, bool, int]] = (
        ("hv", True, 3), ("igd", False, 3), ("bacc", True, 3), ("f1", True, 3),
    )
    bold = {key: _bold_best([stats[m][f"{key}_mean"] for m in methods], hib)
            for key, hib, _ in cols}
    lines = []
    for i, method in enumerate(methods):
        cells = []
        for key, _, prec in cols:
            cell = _fmt_pm(stats[method][f"{key}_mean"], stats[method][f"{key}_std"], prec)
            if bold[key][i]:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{_method_label(method)} & " + " & ".join(cells) + ROW_END)
    _write(out_path, "\n".join(lines))


def extended_metrics_table(methods: List[str], stats: Dict[str, dict], out_path: str):
    """Additional downstream table: AUC, stability, feature count, latency."""
    cols: Sequence[Tuple[str, bool, int]] = (
        ("auc", True, 3),
        ("stability", True, 3),
        ("features", False, 1),
        ("latency_ms", False, 3),
    )
    bold = {key: _bold_best([stats[m][f"{key}_mean"] for m in methods], hib)
            for key, hib, _ in cols}
    lines = []
    for i, method in enumerate(methods):
        cells = []
        for key, _, prec in cols:
            cell = _fmt_pm(stats[method][f"{key}_mean"], stats[method][f"{key}_std"], prec)
            if bold[key][i]:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{_method_label(method)} & " + " & ".join(cells) + ROW_END)
    _write(out_path, "\n".join(lines))


def evasion_table(methods: List[str], evasion: Dict[str, np.ndarray], out_path: str):
    """evasion[method] = array over budgets (mean evasion %, lower better)."""
    n_b = len(next(iter(evasion.values())))
    bold = [_bold_best([evasion[m][b] for m in methods], higher_is_better=False)
            for b in range(n_b)]
    lines = []
    for i, method in enumerate(methods):
        cells = []
        for b in range(n_b):
            cell = _fmt(float(evasion[method][b]), prec=1)
            if bold[b][i]:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{_method_label(method)} & " + " & ".join(cells) + ROW_END)
    _write(out_path, "\n".join(lines))


def ablation_table(rows: Dict[str, dict], order: List[str], out_path: str):
    """rows[name] = {hv_mean/std, igd_mean/std, evasion3_mean/std}."""
    lines = []
    for name in order:
        row = rows[name]
        lines.append(
            f"{name} & {_fmt_pm(row['hv_mean'], row['hv_std'])} & "
            f"{_fmt_pm(row['igd_mean'], row['igd_std'])} & "
            f"{_fmt_pm(row['evasion3_mean'], row['evasion3_std'], 1)}" + ROW_END)
    _write(out_path, "\n".join(lines))


def stats_table(metric: str, comparison: Dict[str, dict], out_path: str):
    """Emit a significance table: baseline, adjusted p, rank-biserial, A12."""
    lines = []
    for name, c in comparison.items():
        lines.append(f"{name} & {c['p_adj']:.3g} & {c['rank_biserial']:.3f} & "
                     f"{c['a12']:.3f}" + ROW_END)
    _write(out_path, "\n".join(lines))


def dataset_table(dataset_summary: dict, out_path: str):
    lines = [
        f"{dataset_summary['name']} & {dataset_summary['snapshot']} & "
        f"{dataset_summary['n_urls']} & {dataset_summary['malicious_pct']:.1f} & "
        f"{dataset_summary['n_features']} & {dataset_summary['windows']}" + ROW_END,
    ]
    _write(out_path, "\n".join(lines))


def hyperparameter_table(selected: dict, out_path: str):
    rows = [
        ("population $m$", "$\\{50,100,200\\}$", selected["pop_size"]),
        ("budget $T_{\\max}$", "$\\{100,200,400\\}$", selected["generations"]),
        ("shape $\\alpha,\\beta$", "$[1,3]$", f"{selected['alpha']}, {selected['beta']}"),
        ("crossover $p_c^{\\min},p_c^{\\max}$", "$[0.6,0.95]$", f"{selected['pc_min']}, {selected['pc_max']}"),
        ("mutation $\\eta^{\\min},\\eta^{\\max}$", "$[0.01,0.5]$", f"{selected['eta_min']}, {selected['eta_max']}"),
        ("mixing $\\lambda$", "$[0,1]$", selected["lam"]),
        ("clip $\\varepsilon$", "$[0.01,0.1]$", selected["eps"]),
    ]
    lines = [f"{name} & {rng} & {value}" + ROW_END for name, rng, value in rows]
    _write(out_path, "\n".join(lines))





























































































































































































































































# --- source: run.py ---

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


class ObjectiveDropEvaluator:
    """Wrapper that disables selected objectives during search only."""

    def __init__(self, base: Evaluator, disabled: Iterable[int]):
        self.base = base
        self.disabled = tuple(disabled)
        self.scaler = base.scaler

    def evaluate(self, mask: np.ndarray) -> np.ndarray:
        values = self.base.evaluate(mask)
        for idx in self.disabled:
            values[idx] = 0.0
        return values

    def evaluate_population(self, masks: np.ndarray) -> np.ndarray:
        return np.vstack([self.evaluate(mask) for mask in masks])


def _split(ds: Dataset, cfg: ExperimentConfig, seed: int):
    Xtr, Xte, ytr, yte = train_test_split(
        ds.X, ds.y, test_size=cfg.test_size, random_state=seed, stratify=ds.y)
    val_frac = cfg.val_size / (1 - cfg.test_size)
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr, ytr, test_size=val_frac, random_state=seed, stratify=ytr)
    return Xtr, Xva, Xte, ytr, yva, yte


def _knee_subset(aX: np.ndarray, aF: np.ndarray, n_features: int) -> np.ndarray:
    if aX.shape[0] == 0:
        return np.ones(n_features, dtype=np.int8)
    scores = aF.sum(axis=1)
    return aX[int(np.argmin(scores))].astype(np.int8)


def _downstream_scores(Xtr, ytr, Xte, yte, mask, seed: int) -> dict:
    sel = np.flatnonzero(mask)
    if sel.size == 0:
        return dict(bacc=0.5, f1=0.0, auc=0.5, latency_ms=0.0)
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(Xtr[:, sel], ytr)
    start = time.perf_counter()
    pred = clf.predict(Xte[:, sel])
    elapsed = time.perf_counter() - start
    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(Xte[:, sel])[:, 1]
    else:
        score = pred
    try:
        auc = float(roc_auc_score(yte, score))
    except ValueError:
        auc = 0.5
    return dict(
        bacc=float(balanced_accuracy_score(yte, pred)),
        f1=float(f1_score(yte, pred, zero_division=0)),
        auc=auc,
        latency_ms=float(1000.0 * elapsed / max(len(yte), 1)),
    )


def _standardised_test(ev: Evaluator, Xtr, Xte):
    return ev.scaler.transform(Xtr), ev.scaler.transform(Xte)


def _dataset_summary(ds: Dataset, cfg: ExperimentConfig) -> dict:
    snapshot = "n/a"
    if ds.timestamps is not None and len(ds.timestamps):
        valid = ds.timestamps[np.isfinite(ds.timestamps.astype(float))]
        if valid.size:
            ts = pd.to_datetime(int(valid.max()), unit="ns", errors="coerce")
            if not pd.isna(ts):
                snapshot = str(ts.date())
    mal = int(ds.y.sum())
    return dict(
        name=ds.name,
        snapshot=snapshot,
        n_urls=int(len(ds.y)),
        malicious_pct=float(100.0 * mal / max(len(ds.y), 1)),
        n_features=int(ds.n_features),
        windows=int(cfg.n_windows),
    )


def _selected_hyperparameters(cfg: ExperimentConfig) -> dict:
    values = asdict(cfg.amofs)
    values.update(pop_size=cfg.pop_size, generations=cfg.generations)
    return values


def _evaluate_archive(ev: Evaluator, aX: np.ndarray) -> np.ndarray:
    if aX.shape[0] == 0:
        return np.empty((0, 4), dtype=float)
    return ev.evaluate_population(aX)


def run_experiment(ds: Dataset, cfg: ExperimentConfig, run_ablation: bool = True):
    methods = list(BASELINES.keys()) + ["AMOFS"]
    metric_keys = ("hv", "igd", "bacc", "f1", "auc", "stability", "features", "latency_ms")
    store = {m: {key: [] for key in metric_keys} | {"evasion": []} for m in methods}
    hv_logs = {m: [] for m in methods}
    amofs_selfreq_logs: List[np.ndarray] = []

    for run_idx in range(cfg.n_runs):
        seed = cfg.base_seed + run_idx
        Xtr, Xva, Xte, ytr, yva, yte = _split(ds, cfg, seed)
        ev = Evaluator(ds, Xtr, ytr, Xva, yva, k=cfg.knn_k,
                       n_windows=cfg.n_windows, mec_delta=cfg.mec_delta)

        run_fronts: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for name, fn in BASELINES.items():
            aX, aF, log = fn(ev, ds.n_features, cfg.pop_size,
                             cfg.generations, seed=seed)
            run_fronts[name] = (aX, aF)
            hv_logs[name].append(log)

        res = run_amofs(ev, ds.n_features, cfg.pop_size, cfg.generations,
                        cfg.amofs, seed=seed)
        run_fronts["AMOFS"] = (res.arch_X, res.arch_F)
        hv_logs["AMOFS"].append(res.hv_log)
        if res.sel_freq_log:
            amofs_selfreq_logs.append(np.vstack(res.sel_freq_log))

        ref = build_reference_front([aF for _, aF in run_fronts.values()])
        Xtr_s, Xte_s = _standardised_test(ev, Xtr, Xte)

        for name, (aX, aF) in run_fronts.items():
            knee = _knee_subset(aX, aF, ds.n_features)
            downstream = _downstream_scores(Xtr_s, ytr, Xte_s, yte, knee, seed)
            curve = evasion_curve(Xtr_s, ytr, Xte_s, yte, knee, ds.costs,
                                  cfg.attack_budgets, seed=seed)
            store[name]["hv"].append(hypervolume(aF))
            store[name]["igd"].append(igd(aF, ref))
            store[name]["bacc"].append(downstream["bacc"])
            store[name]["f1"].append(downstream["f1"])
            store[name]["auc"].append(downstream["auc"])
            store[name]["latency_ms"].append(downstream["latency_ms"])
            store[name]["features"].append(float(knee.sum()))
            store[name]["stability"].append(float(1.0 - ev.f_stab(knee)))
            store[name]["evasion"].append(curve)
        print(f"[run {run_idx + 1}/{cfg.n_runs}] done (seed={seed})")

    return _aggregate_and_export(ds, cfg, methods, store, hv_logs,
                                 amofs_selfreq_logs, run_ablation)


def _aggregate_and_export(ds, cfg, methods, store, hv_logs, amofs_selfreq_logs, run_ablation):
    agg: Dict[str, dict] = {}
    evasion_mean: Dict[str, np.ndarray] = {}
    raw_runs: Dict[str, dict] = {}
    for method in methods:
        agg[method] = {}
        raw_runs[method] = {}
        for key in ("hv", "igd", "bacc", "f1", "auc", "stability", "features", "latency_ms"):
            vals = np.array(store[method][key], dtype=float)
            raw_runs[method][key] = vals.tolist()
            agg[method][f"{key}_mean"] = float(np.nanmean(vals))
            agg[method][f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        evasion_arr = np.vstack(store[method]["evasion"])
        raw_runs[method]["evasion"] = evasion_arr.tolist()
        evasion_mean[method] = np.nanmean(evasion_arr, axis=0)

    name = ds.name
    table_dir = os.path.join(RESULTS, "tables")
    figure_dir = os.path.join(RESULTS, "figures")

    main_comparison_table(methods, agg, os.path.join(table_dir, f"main_{name}.tex"))
    extended_metrics_table(methods, agg, os.path.join(table_dir, f"extended_{name}.tex"))
    evasion_table(methods, evasion_mean, os.path.join(table_dir, f"evasion_{name}.tex"))
    dataset_table(_dataset_summary(ds, cfg), os.path.join(table_dir, f"dataset_{name}.tex"))
    hyperparameter_table(_selected_hyperparameters(cfg), os.path.join(table_dir, f"hyper_{name}.tex"))

    amofs_hv = np.array(store["AMOFS"]["hv"], dtype=float)
    baseline_hv = {m: np.array(store[m]["hv"], dtype=float) for m in methods if m != "AMOFS"}
    comp = compare_against_baselines(amofs_hv, baseline_hv, higher_is_better=True)
    stats_table("HV", comp, os.path.join(table_dir, f"stats_hv_{name}.tex"))

    ablation = {}
    if run_ablation:
        ablation = _ablation(ds, cfg)
        order = ["Full AMOFS", "w/o ANCF", "w/o AGDM", "w/o ANCF \\& AGDM", "w/o $f_{\\mathrm{rob}}$ objective"]
        ablation_table(ablation, order, os.path.join(table_dir, f"ablation_{name}.tex"))

    _plot_comprehensive(methods, agg, store, hv_logs, evasion_mean,
                        amofs_selfreq_logs, figure_dir, name)

    summary = dict(
        dataset=_dataset_summary(ds, cfg),
        config=asdict(cfg),
        metrics=agg,
        raw_runs=raw_runs,
        evasion={m: evasion_mean[m].tolist() for m in methods},
        hv_significance=comp,
        ablation=ablation,
        outputs=dict(tables=table_dir, figures=figure_dir),
    )
    os.makedirs(RESULTS, exist_ok=True)
    summary_path = os.path.join(RESULTS, f"{name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[export] wrote {summary_path}")
    return summary


def _run_amofs_variant(ev: Evaluator, cfg: ExperimentConfig, params, seed: int,
                       disabled_objectives: Tuple[int, ...] = ()):
    search_ev = ObjectiveDropEvaluator(ev, disabled_objectives) if disabled_objectives else ev
    res = run_amofs(search_ev, ev.ds.n_features, cfg.pop_size, cfg.generations, params, seed=seed)
    full_F = _evaluate_archive(ev, res.arch_X)
    return res.arch_X, full_F


def _ablation(ds: Dataset, cfg: ExperimentConfig) -> Dict[str, dict]:
    rows = {
        "Full AMOFS": {"hv": [], "igd": [], "evasion3": []},
        "w/o ANCF": {"hv": [], "igd": [], "evasion3": []},
        "w/o AGDM": {"hv": [], "igd": [], "evasion3": []},
        "w/o ANCF \\& AGDM": {"hv": [], "igd": [], "evasion3": []},
        "w/o $f_{\\mathrm{rob}}$ objective": {"hv": [], "igd": [], "evasion3": []},
    }

    variants = []
    full = deepcopy(cfg.amofs)
    variants.append(("Full AMOFS", full, ()))
    no_ancf = deepcopy(cfg.amofs); no_ancf.use_ancf = False
    variants.append(("w/o ANCF", no_ancf, ()))
    no_agdm = deepcopy(cfg.amofs); no_agdm.use_agdm = False
    variants.append(("w/o AGDM", no_agdm, ()))
    no_both = deepcopy(cfg.amofs); no_both.use_ancf = False; no_both.use_agdm = False
    variants.append(("w/o ANCF \\& AGDM", no_both, ()))
    no_rob = deepcopy(cfg.amofs)
    variants.append(("w/o $f_{\\mathrm{rob}}$ objective", no_rob, (2,)))

    for run_idx in range(cfg.ablation_runs):
        seed = cfg.base_seed + 10_000 + run_idx
        Xtr, Xva, Xte, ytr, yva, yte = _split(ds, cfg, seed)
        ev = Evaluator(ds, Xtr, ytr, Xva, yva, k=cfg.knn_k,
                       n_windows=cfg.n_windows, mec_delta=cfg.mec_delta)
        Xtr_s, Xte_s = _standardised_test(ev, Xtr, Xte)
        fronts = {}
        for label, params, disabled in variants:
            aX, aF = _run_amofs_variant(ev, cfg, params, seed, disabled)
            fronts[label] = (aX, aF)
        ref = build_reference_front([aF for _, aF in fronts.values()])
        for label, (aX, aF) in fronts.items():
            knee = _knee_subset(aX, aF, ds.n_features)
            curve = evasion_curve(Xtr_s, ytr, Xte_s, yte, knee, ds.costs,
                                  cfg.attack_budgets, seed=seed)
            mid = min(2, len(curve) - 1)
            rows[label]["hv"].append(hypervolume(aF))
            rows[label]["igd"].append(igd(aF, ref))
            rows[label]["evasion3"].append(float(curve[mid]))
        print(f"[ablation {run_idx + 1}/{cfg.ablation_runs}] done (seed={seed})")

    out = {}
    for label, metrics in rows.items():
        out[label] = {}
        for key, values in metrics.items():
            vals = np.array(values, dtype=float)
            out[label][f"{key}_mean"] = float(np.nanmean(vals))
            out[label][f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out





def _plot_comprehensive(methods, agg, store, hv_logs, evasion_mean,
                    selfreq_logs, figure_dir, name):
    """Generate a publication-quality multi-panel figure.

    Panel A: HV convergence (cubic-spline smoothed)
    Panel B: Evasion-robustness curves
    Panel C: AGDM guidance evidence
    Panel D: Grouped-bar metric comparison
    Panel E: Feature count comparison
    Panel F: Selection stability
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from scipy.interpolate import make_interp_spline
    import os

    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })

    palette = {
        "AMOFS": "#E74C3C",
        "NSGA-II": "#3498DB",
        "MOEA/D": "#2ECC71",
        "SPEA2": "#F39C12",
        "BPSO": "#9B59B6",
        "BGWO": "#1ABC9C",
    }
    dashes = {
        "AMOFS": (),
        "NSGA-II": (4, 1.5),
        "MOEA/D": (2, 1),
        "SPEA2": (5, 2, 1, 2),
        "BPSO": (3, 1.5, 1, 1.5),
        "BGWO": (1, 1),
    }

    os.makedirs(figure_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    (ax_hv, ax_evasion, ax_agdm,
     ax_metrics, ax_features, ax_stability) = axes.flat

    # ========== Panel A: HV Convergence (smoothed) ==========
    for method, logs in hv_logs.items():
        good = [np.asarray(log, dtype=float) for log in logs if len(log)]
        if not good:
            continue
        min_len = min(len(log) for log in good)
        arr = np.vstack([log[:min_len] for log in good])
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ci = 1.96 * se
        gens = np.arange(1, min_len + 1)

        # cubic-spline smoothing
        if min_len >= 6:
            k = min(3, min_len - 1)
            spl = make_interp_spline(gens, mean, k=k)
            gens_s = np.linspace(gens[0], gens[-1], min_len * 3)
            mean_s = spl(gens_s)
        else:
            gens_s, mean_s = gens, mean

        color = palette.get(method, "#555555")
        dash = dashes.get(method, ())
        lw = 1.8 if method == "AMOFS" else 1.1
        ls = "-" if not dash else (0, dash)
        ax_hv.plot(gens_s, mean_s, color=color, linewidth=lw,
                   linestyle=ls, label=method,
                   zorder=5 if method == "AMOFS" else 3)
        if method == "AMOFS" and min_len >= 4:
            ax_hv.fill_between(gens, mean - ci, mean + ci, color=color,
                               alpha=0.15, edgecolor="none")

    ax_hv.set_xlabel("Generation")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("A  Convergence Dynamics", loc="left", fontweight="bold")
    ax_hv.legend(frameon=True, facecolor="white", edgecolor="#cccccc",
                 framealpha=0.85, fontsize=7, loc="lower right")

    # ========== Panel B: Evasion Curve ==========
    first_curve = next(iter(evasion_mean.values()))
    budgets = np.linspace(0, 1, len(first_curve))
    for method in methods:
        curve = evasion_mean.get(method)
        if curve is None or len(curve) != len(budgets):
            continue
        color = palette.get(method, "#555555")
        dash = dashes.get(method, ())
        lw = 1.8 if method == "AMOFS" else 1.1
        ls = "-" if not dash else (0, dash)

        if len(budgets) >= 6:
            k = min(3, len(budgets) - 1)
            spl = make_interp_spline(budgets, np.asarray(curve), k=k)
            b_s = np.linspace(budgets[0], budgets[-1], 80)
            c_s = spl(b_s)
        else:
            b_s, c_s = budgets, np.asarray(curve)

        ax_evasion.plot(b_s, c_s, color=color, linewidth=lw,
                        linestyle=ls, label=method,
                        zorder=5 if method == "AMOFS" else 3)

    ax_evasion.set_xlabel("Attack Budget (perturbation ratio)")
    ax_evasion.set_ylabel("Evasion Rate")
    ax_evasion.set_title("B  Evasion Robustness", loc="left", fontweight="bold")
    ax_evasion.legend(frameon=True, facecolor="white", edgecolor="#cccccc",
                      framealpha=0.85, fontsize=7, loc="upper left")
    ax_evasion.set_ylim(-0.02, 1.02)

    # ========== Panel C: AGDM Guidance ==========
    if selfreq_logs:
        min_len = min(log.shape[0] for log in selfreq_logs)
        arr = np.stack([log[:min_len] for log in selfreq_logs], axis=0)
        final_freq = arr[:, -1, :].mean(axis=0)
        recovered = final_freq >= np.median(final_freq)
        series = arr[:, :, recovered].mean(axis=2)
        mean = series.mean(axis=0)
        se = series.std(axis=0, ddof=1) / np.sqrt(series.shape[0])
        ci = 1.96 * se
        gens = np.arange(1, min_len + 1)
        gamma = 0.0
        if np.any(recovered):
            gamma = float(np.min(final_freq[recovered]))

        if min_len >= 6:
            k = min(3, min_len - 1)
            spl = make_interp_spline(gens, mean, k=k)
            gens_s = np.linspace(gens[0], gens[-1], min_len * 3)
            mean_s = spl(gens_s)
        else:
            gens_s, mean_s = gens, mean

        ax_agdm.plot(gens_s, mean_s, color="#2C3E50", linewidth=1.8,
                     label="AGDM-guided features")
        ax_agdm.fill_between(gens, mean - ci, mean + ci, color="#2C3E50",
                             alpha=0.12, edgecolor="none")
        ax_agdm.axhline(gamma, color="#7F8C8D", linestyle="--", linewidth=1.0)
        ax_agdm.text(gens[-1] * 0.98, gamma + 0.005, f"threshold {gamma:.3f}",
                     fontsize=7.5, color="#7F8C8D", va="bottom", ha="right")
        ax_agdm.set_ylim(-0.02, 1.02)
    ax_agdm.set_xlabel("Generation")
    ax_agdm.set_ylabel("Mean Selection Frequency")
    ax_agdm.set_title("C  AGDM Guidance", loc="left", fontweight="bold")

    # ========== Panel D: Metrics bar chart ==========
    metric_labels = ["HV", "IGD\n(inv.)", "BAcc", "F1", "AUC", "Stab."]
    metric_keys = ["hv", "igd", "bacc", "f1", "auc", "stability"]
    invert_metrics = {"igd": True}

    x = np.arange(len(metric_labels))
    width = 0.12
    for i, method in enumerate(methods):
        vals = []
        for key in metric_keys:
            v = agg[method].get(f"{key}_mean", 0)
            if invert_metrics.get(key, False):
                v = 1.0 / (1.0 + v) if v > 0 else 1.0
            vals.append(v)
        color = palette.get(method, "#555555")
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax_metrics.bar(x + offset, vals, width, color=color,
                       label=method, alpha=0.88, edgecolor="white",
                       linewidth=0.3, zorder=3)

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_labels, fontsize=7.5)
    ax_metrics.set_ylabel("Normalised Score (higher \u2192 better)")
    ax_metrics.set_title("D  Overall Metrics", loc="left", fontweight="bold")
    ax_metrics.set_ylim(0, 1.08)
    ax_metrics.legend(frameon=True, facecolor="white", edgecolor="#cccccc",
                      framealpha=0.85, fontsize=6.5, ncol=3,
                      loc="upper right")

    # ========== Panel E: Feature count ==========
    y_pos = np.arange(len(methods))
    mean_feat = []
    std_feat = []
    colors_feat = []
    for method in methods:
        vals = np.array(store[method]["features"], dtype=float)
        mean_feat.append(np.nanmean(vals))
        std = np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0
        std_feat.append(std)
        colors_feat.append(palette.get(method, "#555555"))

    ax_features.barh(y_pos, mean_feat, xerr=std_feat, color=colors_feat,
                     alpha=0.88, edgecolor="white", linewidth=0.3,
                     height=0.55, zorder=3, capsize=2)
    ax_features.set_yticks(y_pos)
    ax_features.set_yticklabels(methods, fontsize=8)
    ax_features.set_xlabel("Number of Selected Features")
    ax_features.set_title("E  Feature Count", loc="left", fontweight="bold")
    ax_features.margins(y=0.15)

    # ========== Panel F: Stability ==========
    stab_means = []
    stab_errs = []
    for method in methods:
        vals = np.array(store[method]["stability"], dtype=float)
        stab_means.append(np.nanmean(vals))
        std = np.nanstd(vals, ddof=1) if len(vals) > 1 else 0
        stab_errs.append(std)
    colors_stab = [palette.get(m, "#555555") for m in methods]

    ax_stability.bar(methods, stab_means, yerr=stab_errs, color=colors_stab,
                     alpha=0.88, edgecolor="white", linewidth=0.3,
                     width=0.55, zorder=3, capsize=3)
    ax_stability.set_xticks(np.arange(len(methods)))
    ax_stability.set_xticklabels(methods, fontsize=7.5, rotation=25, ha="right")
    ax_stability.set_ylabel("Selection Stability (\u2191 better)")
    ax_stability.set_title("F  Selection Stability", loc="left", fontweight="bold")
    ax_stability.set_ylim(0, 1.08)

    fig.tight_layout(pad=1.5, h_pad=2.5, w_pad=1.8)
    out = os.path.join(figure_dir, f"comprehensive_{name}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[export] wrote {out}")

    # ----- also save a clean 2-panel version for quick reference -----
    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 3.5))
    for method, logs in hv_logs.items():
        good = [np.asarray(log, dtype=float) for log in logs if len(log)]
        if not good:
            continue
        min_len = min(len(log) for log in good)
        arr = np.vstack([log[:min_len] for log in good])
        mean = arr.mean(axis=0)
        ci = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
        gens = np.arange(1, min_len + 1)
        color = palette.get(method, "#555555")
        dash = dashes.get(method, ())
        lw = 1.8 if method == "AMOFS" else 1.1
        ls = "-" if not dash else (0, dash)

        if min_len >= 6:
            k = min(3, min_len - 1)
            spl = make_interp_spline(gens, mean, k=k)
            gens_s = np.linspace(gens[0], gens[-1], min_len * 3)
            mean_s = spl(gens_s)
        else:
            gens_s, mean_s = gens, mean

        ax2[0].plot(gens_s, mean_s, color=color, linewidth=lw,
                    linestyle=ls, label=method)
        if method == "AMOFS" and min_len >= 4:
            ax2[0].fill_between(gens, mean - ci, mean + ci, color=color,
                                alpha=0.15, edgecolor="none")
    ax2[0].set_xlabel("Generation")
    ax2[0].set_ylabel("Hypervolume")
    ax2[0].set_title("Convergence", fontweight="bold")
    ax2[0].legend(fontsize=7.5, framealpha=0.85, edgecolor="#cccccc")

    if selfreq_logs:
        min_len = min(log.shape[0] for log in selfreq_logs)
        arr = np.stack([log[:min_len] for log in selfreq_logs], axis=0)
        final_freq = arr[:, -1, :].mean(axis=0)
        recovered = final_freq >= np.median(final_freq)
        series = arr[:, :, recovered].mean(axis=2)
        mean = series.mean(axis=0)
        ci = 1.96 * series.std(axis=0, ddof=1) / np.sqrt(series.shape[0]) if series.shape[0] > 1 else np.zeros_like(mean)
        gens = np.arange(1, min_len + 1)
        gamma = 0.0
        if np.any(recovered):
            gamma = float(np.min(final_freq[recovered]))

        if min_len >= 6:
            k = min(3, min_len - 1)
            spl = make_interp_spline(gens, mean, k=k)
            gens_s = np.linspace(gens[0], gens[-1], min_len * 3)
            mean_s = spl(gens_s)
        else:
            gens_s, mean_s = gens, mean
        ax2[1].plot(gens_s, mean_s, color="#2C3E50", linewidth=1.6)
        ax2[1].fill_between(gens, mean - ci, mean + ci, color="#2C3E50",
                            alpha=0.12, edgecolor="none")
        ax2[1].axhline(gamma, color="grey", linestyle="--", linewidth=1.0)
        ax2[1].text(gens[-1] * 0.98, gamma + 0.005, f"threshold {gamma:.3f}",
                    fontsize=8, color="grey", va="bottom", ha="right")
    ax2[1].set_xlabel("Generation")
    ax2[1].set_ylabel("Mean Archive Frequency")
    ax2[1].set_title("AGDM Guidance Evidence", fontweight="bold")

    fig2.tight_layout()
    out2 = os.path.join(figure_dir, f"convergence_{name}.png")
    fig2.savefig(out2, dpi=300)
    plt.close(fig2)
    print(f"[export] wrote {out2}")

def run_main():
    dataset_path = "C:/Users/Administrator/Desktop/论文/恶意URL检测/code/data/processed/phishing_majestic.csv"
    dataset_name = "PhishingMajestic"

    cfg = ExperimentConfig()
    ds = load_dataset(dataset_path, dataset_name)
    run_experiment(ds, cfg, run_ablation=True)


# --- entry point ---
def main():
    """Entry point: run AMOFS on a real dataset."""
    run_main()

if __name__ == "__main__":
    main()