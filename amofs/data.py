"""Dataset loading, chronological splitting, and temporal windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import FEATURE_COSTS, FEATURE_NAMES, FEATURE_TIERS, TIER_COST


@dataclass
class Dataset:
    name: str
    X: np.ndarray                 # (n, d) raw (unstandardised) features
    y: np.ndarray                 # (n,) in {0, 1}
    t: np.ndarray                 # (n,) int64 ns timestamps (sorted ascending)
    feature_names: List[str]
    costs: np.ndarray             # (d,) modification cost d_i
    tiers: List[str]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


@dataclass
class Split:
    """Chronological development / out-of-time split."""
    tr: np.ndarray      # training indices (development period)
    va: np.ndarray      # validation indices (development period)
    te: np.ndarray      # out-of-time test indices (latest period)
    dev: np.ndarray     # all development indices (tr ∪ va), time ordered
    cutoff_ns: int


def _costs_for(names: List[str]) -> Tuple[np.ndarray, List[str]]:
    costs, tiers = [], []
    for n in names:
        if n in FEATURE_COSTS:
            costs.append(FEATURE_COSTS[n]); tiers.append(FEATURE_TIERS[n])
        else:  # unknown column from a user CSV -> moderate tier
            costs.append(TIER_COST["moderate"]); tiers.append("moderate")
    return np.asarray(costs, dtype=float), tiers


def load_dataset(csv_path: str, name: str, label_col: str = "label",
                 time_col: str = "timestamp") -> Dataset:
    df = pd.read_csv(csv_path)
    if label_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"{csv_path}: need '{label_col}' and '{time_col}' columns")
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    keep = ts.notna().to_numpy()
    df, ts = df.loc[keep].reset_index(drop=True), ts[keep]
    ns = ts.dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype("int64")
    order = np.argsort(ns, kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    t = ns[order]
    y = df[label_col].astype(int).to_numpy()
    feat_df = df.drop(columns=[c for c in (label_col, time_col, "url") if c in df.columns])
    feat_df = feat_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.fillna(0.0)
    # drop constant columns (carry no information, break standardisation)
    nunique = feat_df.nunique()
    feat_df = feat_df.loc[:, nunique > 1]
    names = list(feat_df.columns)
    costs, tiers = _costs_for(names)
    return Dataset(name=name, X=feat_df.to_numpy(dtype=float), y=y, t=t,
                   feature_names=names, costs=costs, tiers=tiers)


def chronological_split(ds: Dataset, dev_fraction: float, val_fraction: float,
                        seed: int) -> Split:
    """First ``dev_fraction`` of time -> development (train/val), rest -> OOT test."""
    n = len(ds.y)
    n_dev = int(round(dev_fraction * n))
    dev = np.arange(n_dev)
    te = np.arange(n_dev, n)
    tr, va = train_test_split(dev, test_size=val_fraction, random_state=seed, stratify=ds.y[dev])
    return Split(tr=np.sort(tr), va=np.sort(va), te=te, dev=dev, cutoff_ns=int(ds.t[n_dev - 1]))


def temporal_windows(idx_sorted: np.ndarray, n_windows: int) -> List[np.ndarray]:
    """Equal-frequency windows over an already time-ordered index array."""
    return [w for w in np.array_split(idx_sorted, n_windows) if len(w) >= 4]


# ---------------------------------------------------------------------------
# synthetic data (pipeline verification only; never a measurement)
# ---------------------------------------------------------------------------
def make_synthetic(n_samples: int = 1500, n_features: int = 30, n_informative: int = 8,
                   seed: int = 0, name: str = "synthetic") -> Dataset:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples)
    X = rng.normal(size=(n_samples, n_features))
    tiers = ["moderate"] * n_features
    info = sorted(rng.choice(n_features, size=n_informative, replace=False).tolist())
    for rank, idx in enumerate(info):
        X[:, idx] += (1.4 - 0.12 * rank) * (2 * y - 1)
        tiers[idx] = "cheap" if rank < n_informative // 2 else "expensive"
    # mild drift: informative features lose signal over time
    drift = np.linspace(0, 0.6, n_samples)
    for idx in info[: n_informative // 2]:
        X[:, idx] -= drift * (2 * y - 1)
    costs = np.array([TIER_COST[t] for t in tiers])
    names = [f"f{i}_{tiers[i]}" for i in range(n_features)]
    t = np.sort(rng.integers(1_600_000_000, 1_700_000_000, size=n_samples)).astype("int64") * 10**9
    return Dataset(name=name, X=X, y=y, t=t, feature_names=names, costs=costs, tiers=tiers)
