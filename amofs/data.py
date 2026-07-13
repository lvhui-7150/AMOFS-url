"""Dataset loading, feature pipeline, cost model, and temporal windows.

Real datasets
-------------
The paper uses three public corpora. Place their feature CSVs under ``data/``:

    data/iscx_url_2016.csv
    data/phishtank.csv
    data/kaggle_malicious_url.csv

Each CSV must contain numeric feature columns, a binary ``label`` column
(1 = malicious, 0 = benign), and optionally a ``timestamp`` column used to form
the temporal windows for the stability objective. ``load_dataset`` reads a CSV
in this format; if ``timestamp`` is absent, windows are formed by row order.

A feature ``cost`` (the modification cost d_i from the threat model) is assigned
per feature group by ``assign_costs``; override the mapping for your own
pipeline. The cheap/moderate/expensive ordinal scale is documented inline.

Synthetic data
--------------
``make_synthetic`` produces a small, internally consistent dataset for
*pipeline verification only* (smoke tests). It is NOT a substitute for the real
corpora and must never be reported as a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def make_synthetic(n_samples: int = 1500, n_features: int = 30,
                    n_informative: int = 8, seed: int = 0,
                    name: str = "synthetic") -> Dataset:
    """Generate a small, internally consistent synthetic dataset.

    The construction is deliberately transparent so that the *expected*
    behaviour of AMOFS is interpretable on it: a handful of informative
    features carry the class signal, and we deliberately make the cheapest
    informative features the most discriminative, so that a robustness-aware
    selector should prefer slightly costlier-but-still-informative features.
    FOR PIPELINE VERIFICATION ONLY -- not a measurement.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples)

    X = rng.normal(size=(n_samples, n_features))
    feature_names = []
    costs = np.empty(n_features, dtype=float)

    # informative features: shift mean by class; assign mixed costs
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)
    for rank, idx in enumerate(sorted(informative_idx)):
        # strongest signal on the cheapest features (adversarially inconvenient)
        strength = 1.4 - 0.12 * rank
        X[:, idx] += strength * (2 * y - 1)
        if rank < n_informative // 2:
            costs[idx] = COST_CHEAP
            feature_names.append(f"lex_informative_{rank}_len")
        else:
            costs[idx] = COST_EXPENSIVE
            feature_names.append(f"host_informative_{rank}_age")
    # noise features
    info_set = set(int(i) for i in informative_idx)
    for idx in range(n_features):
        if idx in info_set:
            continue
        costs[idx] = COST_MODERATE
        feature_names.append(f"noise_subdomain_{idx}")

    # reorder names to align with column order
    ordered_names = [None] * n_features
    name_iter = iter(feature_names)
    # rebuild names deterministically by column to keep alignment simple
    ordered_names = []
    for idx in range(n_features):
        if idx in info_set:
            if costs[idx] == COST_CHEAP:
                ordered_names.append(f"lex_informative_{idx}_len")
            else:
                ordered_names.append(f"host_informative_{idx}_age")
        else:
            ordered_names.append(f"noise_subdomain_{idx}")

    ts = np.sort(rng.integers(0, 10_000, size=n_samples))
    return Dataset(name=name, X=X, y=y, feature_names=ordered_names,
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
