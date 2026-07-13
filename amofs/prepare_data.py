"""Prepare public URL datasets for the AMOFS experiments.

The script downloads reproducible, no-login public URL feeds and converts raw
URLs into numeric feature CSVs accepted by :func:`amofs.data.load_dataset`.
Generated CSVs contain a binary ``label`` column, a ``timestamp`` column, and
stable numeric feature columns.

Examples
--------
    python -m amofs.prepare_data --dataset urlhaus_majestic --limit-per-class 2500
    python -m amofs.prepare_data --all --limit-per-class 2500
"""
from __future__ import annotations

import argparse
import csv
import io
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Tuple
from urllib.request import Request, urlopen

import pandas as pd

from .features import extract_url_features, feature_names


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

URLHAUS_RECENT = "https://urlhaus.abuse.ch/downloads/csv_recent/"
MAJESTIC_MILLION = "https://downloads.majestic.com/majestic_million.csv"
GITHUB_FAIZAN = (
    "https://raw.githubusercontent.com/faizann24/"
    "Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv"
)
PHISHING_ACTIVE = (
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/"
    "master/phishing-links-ACTIVE.txt"
)


def _fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "AMOFS-reproducibility-script/1.0"})
    chunks = []
    with urlopen(req, timeout=180) as resp:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", "replace")


def _cache_text(url: str, filename: str, refresh: bool = False) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    if refresh or not os.path.exists(path):
        text = _fetch_text(url)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _cache_majestic_subset(limit: int, refresh: bool = False) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"majestic_top_{limit}.csv")
    if not refresh and os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    req = Request(MAJESTIC_MILLION, headers={"User-Agent": "AMOFS-reproducibility-script/1.0"})
    lines = []
    with urlopen(req, timeout=180) as resp:
        for i, raw_line in enumerate(resp):
            if i > limit:
                break
            lines.append(raw_line.decode("utf-8", "replace"))
    text = "".join(lines)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return text


def _balanced(records: List[Tuple[str, int, str]], limit_per_class: int) -> List[Tuple[str, int, str]]:
    benign = [row for row in records if row[1] == 0][:limit_per_class]
    malicious = [row for row in records if row[1] == 1][:limit_per_class]
    return benign + malicious


def _rows_from_urlhaus(refresh: bool, limit: int) -> List[Tuple[str, int, str]]:
    text = _cache_text(URLHAUS_RECENT, "urlhaus_recent.csv", refresh)
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            rec = next(csv.reader([line]))
        except csv.Error:
            continue
        if len(rec) < 3 or rec[0] == "id":
            continue
        ts = rec[1]
        url = rec[2]
        rows.append((url, 1, ts))
        if len(rows) >= limit:
            break
    return rows


def _rows_from_majestic(refresh: bool, limit: int) -> List[Tuple[str, int, str]]:
    text = _cache_majestic_subset(limit, refresh)
    df = pd.read_csv(io.StringIO(text), usecols=["GlobalRank", "Domain"])
    now = datetime.now(timezone.utc)
    rows = []
    for i, row in df.head(limit).iterrows():
        domain = str(row["Domain"]).strip()
        if not domain or domain == "nan":
            continue
        # Spread timestamps deterministically to create temporal windows.
        ts = now - timedelta(minutes=int(row["GlobalRank"]))
        rows.append((f"https://{domain}/", 0, ts.strftime("%Y-%m-%d %H:%M:%S")))
    return rows


def _rows_from_faizan(refresh: bool, limit_per_class: int) -> List[Tuple[str, int, str]]:
    os.makedirs(RAW_DIR, exist_ok=True)
    cache_path = os.path.join(RAW_DIR, f"faizan_subset_{limit_per_class}.csv")
    label_map = {"good": 0, "benign": 0, "bad": 1, "malicious": 1}
    now = datetime.now(timezone.utc)
    rows = []
    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                ts = now - timedelta(minutes=idx)
                rows.append((row["url"], int(row["label"]), ts.strftime("%Y-%m-%d %H:%M:%S")))
        return rows

    req = Request(GITHUB_FAIZAN, headers={"User-Agent": "AMOFS-reproducibility-script/1.0"})
    counts = {0: 0, 1: 0}
    with urlopen(req, timeout=180) as resp:
        text_stream = io.TextIOWrapper(resp, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text_stream)
        if "url" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError("Faizan dataset must contain url,label columns")
        for idx, row in enumerate(reader):
            label = str(row["label"]).strip().lower()
            if label not in label_map:
                continue
            y = label_map[label]
            if counts[y] >= limit_per_class:
                continue
            ts = now - timedelta(minutes=idx)
            rows.append((str(row["url"]), y, ts.strftime("%Y-%m-%d %H:%M:%S")))
            counts[y] += 1
            if counts[0] >= limit_per_class and counts[1] >= limit_per_class:
                break
    if counts[0] == 0 or counts[1] == 0:
        raise RuntimeError(f"Faizan stream produced imbalanced labels: {counts}")
    with open(cache_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label"])
        writer.writeheader()
        for url, label, _ in rows:
            writer.writerow({"url": url, "label": label})
    return rows


def build_urlhaus_majestic(refresh: bool, limit_per_class: int) -> List[Tuple[str, int, str]]:
    malicious = _rows_from_urlhaus(refresh, limit_per_class)
    benign = _rows_from_majestic(refresh, limit_per_class)
    return benign + malicious


def _rows_from_phishing_database(refresh: bool, limit: int) -> List[Tuple[str, int, str]]:
    os.makedirs(RAW_DIR, exist_ok=True)
    cache_path = os.path.join(RAW_DIR, f"phishing_active_subset_{limit}.txt")
    rows = []
    now = datetime.now(timezone.utc)
    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                url = line.strip()
                if url:
                    ts = now - timedelta(minutes=idx)
                    rows.append((url, 1, ts.strftime("%Y-%m-%d %H:%M:%S")))
        return rows[:limit]

    req = Request(PHISHING_ACTIVE, headers={"User-Agent": "AMOFS-reproducibility-script/1.0"})
    with urlopen(req, timeout=180) as resp:
        text_stream = io.TextIOWrapper(resp, encoding="utf-8", errors="replace", newline="")
        for idx, line in enumerate(text_stream):
            url = line.strip()
            if not url or url.startswith("#") or "." not in url:
                continue
            ts = now - timedelta(minutes=idx)
            rows.append((url, 1, ts.strftime("%Y-%m-%d %H:%M:%S")))
            if len(rows) >= limit:
                break
    with open(cache_path, "w", encoding="utf-8", newline="") as f:
        for url, _, _ in rows:
            f.write(url + "\n")
    return rows


def build_phishing_majestic(refresh: bool, limit_per_class: int) -> List[Tuple[str, int, str]]:
    malicious = _rows_from_phishing_database(refresh, limit_per_class)
    benign = _rows_from_majestic(refresh, limit_per_class)
    return benign + malicious


def build_faizan(refresh: bool, limit_per_class: int) -> List[Tuple[str, int, str]]:
    return _rows_from_faizan(refresh, limit_per_class)


DATASETS: Dict[str, Callable[[bool, int], List[Tuple[str, int, str]]]] = {
    "phishing_majestic": build_phishing_majestic,
    "urlhaus_majestic": build_urlhaus_majestic,
    "faizan_public": build_faizan,
}


def write_feature_csv(name: str, records: Iterable[Tuple[str, int, str]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    names = feature_names()
    out_path = os.path.join(out_dir, f"{name}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "timestamp"] + names)
        writer.writeheader()
        for url, label, timestamp in records:
            feats = extract_url_features(url)
            row = {"url": url, "label": int(label), "timestamp": timestamp}
            row.update(feats)
            writer.writerow(row)
    return out_path


def describe_csv(path: str) -> dict:
    df = pd.read_csv(path)
    n = len(df)
    mal = int(df["label"].sum())
    feature_cols = [c for c in df.columns if c not in {"url", "label", "timestamp"}]
    return {
        "path": path,
        "rows": n,
        "malicious": mal,
        "malicious_pct": 100.0 * mal / max(n, 1),
        "n_features": len(feature_cols),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="dataset to prepare")
    parser.add_argument("--all", action="store_true", help="prepare every supported public dataset")
    parser.add_argument("--limit-per-class", type=int, default=2500)
    parser.add_argument("--refresh", action="store_true", help="redownload raw source files")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        raise SystemExit("choose --dataset NAME or --all")

    selected = sorted(DATASETS) if args.all else [args.dataset]
    for name in selected:
        rows = DATASETS[name](args.refresh, args.limit_per_class)
        if not rows:
            raise RuntimeError(f"{name}: no rows produced")
        path = write_feature_csv(name, rows, PROCESSED_DIR)
        desc = describe_csv(path)
        print(
            f"[prepare] {name}: {desc['rows']} rows, "
            f"{desc['malicious_pct']:.1f}% malicious, "
            f"{desc['n_features']} features -> {path}"
        )


if __name__ == "__main__":
    main()
