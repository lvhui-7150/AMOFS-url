"""Build timestamped malicious-URL corpora for the AMOFS experiments.

Each corpus pairs a *malicious* feed that carries a first-seen timestamp with a
*benign* source that also carries a real creation timestamp, so that temporal
windows, the drift-stability objective and the chronological out-of-time test
are all defined on genuine time information.

Malicious sources (pick with ``--malicious``)
    urlhaus    abuse.ch URLhaus (malware distribution URLs, ``dateadded``)
    phishtank  PhishTank verified online phish (``submission_time``)
    phishstats PhishStats public API (phishing, ``date``) - fallback if you
               cannot obtain a PhishTank key
    threatfox  abuse.ch ThreatFox URL IOCs (``first_seen_utc``)
    csv        any CSV you already have (``--mal-csv``, with url/time columns)

Benign source
    hn         Hacker News story links via the public Algolia API; every link
               has ``created_at_i``.  Benign URLs are sampled so that their
               time distribution matches the malicious feed window by window.
    csv        your own benign CSV with url/time columns (``--ben-csv``)

Outputs ``data/processed/<name>.csv`` with columns
    url, label (1 = malicious), timestamp (ISO-8601 UTC), <67 features>
and ``data/processed/<name>_meta.json`` describing provenance.

Examples
--------
    python -m amofs.prepare_data --name urlhaus_hn   --malicious urlhaus   --max-malicious 15000
    python -m amofs.prepare_data --name phishtank_hn --malicious phishtank --phishtank-key KEY --max-malicious 15000
    python -m amofs.prepare_data --name threatfox_hn --malicious threatfox --abusech-key KEY --max-malicious 15000

abuse.ch downloads may require a free Auth-Key (https://auth.abuse.ch/); pass
it with ``--abusech-key`` or set ``ABUSECH_AUTH_KEY``.  If a download is blocked
in your network, download the file manually and point ``--mal-file`` at it.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, extract_url_features

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

URLHAUS_ONLINE = "https://urlhaus.abuse.ch/downloads/csv_online/"
URLHAUS_RECENT = "https://urlhaus.abuse.ch/downloads/csv_recent/"
URLHAUS_FULL = "https://urlhaus.abuse.ch/downloads/csv/"
THREATFOX_URLS = "https://threatfox.abuse.ch/export/csv/urls/full/"
PHISHTANK_PUBLIC = "http://data.phishtank.com/data/online-valid.csv"
PHISHTANK_KEYED = "http://data.phishtank.com/data/{key}/online-valid.csv"
PHISHSTATS_API = "https://phishstats.info:2096/api/phishing"
HN_ALGOLIA = "https://hn.algolia.com/api/v1/search_by_date"

UA = "AMOFS-reproducibility/2.0 (+research use)"

Record = Tuple[str, int, datetime]  # (url, label, utc timestamp)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 300,
         retries: int = 3) -> bytes:
    hdrs = {"User-Agent": UA}
    if headers:
        hdrs.update(headers)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers=hdrs)
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (HTTPError, URLError, TimeoutError) as e:  # pragma: no cover
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last_err}")


def _cached_download(url: str, filename: str, refresh: bool,
                     headers: Optional[Dict[str, str]] = None) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    if refresh or not os.path.exists(path):
        print(f"[download] {url}")
        data = _get(url, headers=headers)
        with open(path, "wb") as f:
            f.write(data)
    return path


def _read_text_maybe_zip(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(4)
    if head.startswith(b"PK"):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            with zf.open(names[0]) as member:
                return member.read().decode("utf-8", "replace")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _parse_time(value: str) -> Optional[datetime]:
    v = str(value).strip()
    if not v:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(v, fmt)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        ts = pd.to_datetime(v, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _is_http_url(u: str) -> bool:
    u = u.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


# ---------------------------------------------------------------------------
# Malicious sources
# ---------------------------------------------------------------------------
def _abusech_headers(key: Optional[str]) -> Dict[str, str]:
    key = key or os.environ.get("ABUSECH_AUTH_KEY", "")
    return {"Auth-Key": key} if key else {}


def load_urlhaus(variant: str, refresh: bool, key: Optional[str],
                 manual_file: Optional[str]) -> List[Record]:
    if manual_file:
        path = manual_file
    else:
        url = {"online": URLHAUS_ONLINE, "recent": URLHAUS_RECENT, "full": URLHAUS_FULL}[variant]
        path = _cached_download(url, f"urlhaus_{variant}.bin", refresh, _abusech_headers(key))
    text = _read_text_maybe_zip(path)
    out: List[Record] = []
    for row in csv.reader(line for line in text.splitlines() if line and not line.startswith("#")):
        if len(row) < 3 or row[0] == "id":
            continue
        ts, u = _parse_time(row[1]), row[2].strip()
        if ts is None or not _is_http_url(u):
            continue
        out.append((u, 1, ts))
    return out


def load_threatfox(refresh: bool, key: Optional[str], manual_file: Optional[str]) -> List[Record]:
    path = manual_file or _cached_download(THREATFOX_URLS, "threatfox_urls.bin", refresh,
                                           _abusech_headers(key))
    text = _read_text_maybe_zip(path)
    out: List[Record] = []
    for row in csv.reader((line for line in text.splitlines() if line and not line.startswith("#")),
                          skipinitialspace=True):
        if len(row) < 4 or row[0] == "first_seen_utc":
            continue
        ts, u, ioc_type = _parse_time(row[0]), row[2].strip().strip('"'), row[3].strip().strip('"')
        if ioc_type != "url" or ts is None or not _is_http_url(u):
            continue
        out.append((u, 1, ts))
    return out


def load_phishtank(refresh: bool, key: Optional[str], manual_file: Optional[str]) -> List[Record]:
    if manual_file:
        path = manual_file
    else:
        url = PHISHTANK_KEYED.format(key=key) if key else PHISHTANK_PUBLIC
        path = _cached_download(url, "phishtank_online_valid.bin", refresh,
                                {"User-Agent": "phishtank/amofs-research"})
    text = _read_text_maybe_zip(path)
    df = pd.read_csv(io.StringIO(text))
    time_col = "submission_time" if "submission_time" in df.columns else "verification_time"
    out: List[Record] = []
    for u, t in zip(df["url"].astype(str), df[time_col].astype(str)):
        ts = _parse_time(t)
        if ts is not None and _is_http_url(u):
            out.append((u.strip(), 1, ts))
    return out


def load_phishstats(max_rows: int, refresh: bool) -> List[Record]:
    """Page through the PhishStats API (100 rows/page) until ``max_rows``."""
    os.makedirs(RAW_DIR, exist_ok=True)
    cache = os.path.join(RAW_DIR, f"phishstats_{max_rows}.json")
    if not refresh and os.path.exists(cache):
        rows = json.load(open(cache, encoding="utf-8"))
    else:
        rows = []
        page = 1
        while len(rows) < max_rows:
            q = urlencode({"_size": 100, "_p": page, "_sort": "-date"})
            data = json.loads(_get(f"{PHISHSTATS_API}?{q}").decode("utf-8", "replace"))
            if not data:
                break
            rows.extend({"url": d.get("url"), "date": d.get("date")} for d in data)
            page += 1
            time.sleep(0.5)
            if page % 20 == 0:
                print(f"[phishstats] {len(rows)} rows")
        json.dump(rows, open(cache, "w", encoding="utf-8"))
    out: List[Record] = []
    for r in rows:
        ts = _parse_time(str(r.get("date", "")))
        u = str(r.get("url", "")).strip()
        if ts is not None and _is_http_url(u):
            out.append((u, 1, ts))
    return out


def load_csv_records(path: str, label: int, url_col: str, time_col: str) -> List[Record]:
    df = pd.read_csv(path)
    out: List[Record] = []
    for u, t in zip(df[url_col].astype(str), df[time_col].astype(str)):
        ts = _parse_time(t)
        if ts is not None and _is_http_url(u):
            out.append((u.strip(), label, ts))
    return out


# ---------------------------------------------------------------------------
# Benign source: Hacker News (Algolia), time-matched to the malicious feed
# ---------------------------------------------------------------------------
def _hn_fetch_window(lo: int, hi: int, need: int, exclude: set) -> List[Record]:
    """Fetch up to ``need`` story URLs with lo <= created_at_i < hi."""
    out: List[Record] = []
    cur_hi = hi
    while len(out) < need and cur_hi > lo:
        q = urlencode({
            "tags": "story",
            "numericFilters": f"created_at_i>={lo},created_at_i<{cur_hi}",
            "hitsPerPage": 1000,
        })
        try:
            payload = json.loads(_get(f"{HN_ALGOLIA}?{q}", timeout=60).decode("utf-8", "replace"))
        except RuntimeError as e:  # pragma: no cover
            print(f"[hn] window fetch failed ({e}); continuing")
            break
        hits = payload.get("hits", [])
        if not hits:
            break
        min_ts = cur_hi
        for h in hits:
            ts_i = int(h.get("created_at_i", 0))
            min_ts = min(min_ts, ts_i)
            u = (h.get("url") or "").strip()
            if not _is_http_url(u) or u in exclude:
                continue
            exclude.add(u)
            out.append((u, 0, datetime.fromtimestamp(ts_i, tz=timezone.utc)))
            if len(out) >= need:
                break
        if min_ts >= cur_hi:
            break
        cur_hi = min_ts  # slide the window backwards
        time.sleep(0.25)  # stay far below the 10k req/h limit
    return out


def benign_from_hn(mal_times: Sequence[datetime], ratio: float, n_bins: int,
                   refresh: bool, cache_name: str) -> List[Record]:
    os.makedirs(RAW_DIR, exist_ok=True)
    cache = os.path.join(RAW_DIR, f"hn_benign_{cache_name}.csv")
    if not refresh and os.path.exists(cache):
        df = pd.read_csv(cache)
        out = [(str(u), 0, _parse_time(t)) for u, t in zip(df["url"], df["timestamp"])]
        return [r for r in out if r[2] is not None]

    ts = np.array(sorted(int(t.timestamp()) for t in mal_times))
    edges = np.quantile(ts, np.linspace(0, 1, n_bins + 1)).astype(int)
    edges[-1] += 1
    out: List[Record] = []
    seen: set = set()
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        n_mal_bin = int(np.sum((ts >= lo) & (ts < hi)))
        need = int(math.ceil(n_mal_bin * ratio))
        got = _hn_fetch_window(lo, hi, need, seen)
        out.extend(got)
        print(f"[hn] bin {b + 1}/{n_bins} {datetime.fromtimestamp(lo, tz=timezone.utc).date()}"
              f"..{datetime.fromtimestamp(hi, tz=timezone.utc).date()}: need {need}, got {len(got)}")
    pd.DataFrame({"url": [r[0] for r in out],
                  "timestamp": [r[2].isoformat() for r in out]}).to_csv(cache, index=False)
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def _dedupe(records: Iterable[Record]) -> List[Record]:
    seen: set = set()
    out: List[Record] = []
    for u, y, t in records:
        key = u.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append((key, y, t))
    return out


def _subsample_uniform_in_time(records: List[Record], n_max: int, seed: int) -> List[Record]:
    """Keep at most ``n_max`` records while preserving the full time span."""
    if len(records) <= n_max:
        return records
    records = sorted(records, key=lambda r: r[2])
    idx = np.linspace(0, len(records) - 1, n_max).round().astype(int)
    idx = np.unique(idx)
    rng = np.random.default_rng(seed)
    if len(idx) < n_max:  # pad with random extra indices
        extra = rng.choice(np.setdiff1d(np.arange(len(records)), idx), n_max - len(idx), replace=False)
        idx = np.sort(np.concatenate([idx, extra]))
    return [records[i] for i in idx]


def write_feature_csv(name: str, records: List[Record], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.csv")
    records = sorted(records, key=lambda r: r[2])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "label", "timestamp"] + FEATURE_NAMES)
        for i, (u, y, t) in enumerate(records):
            try:
                feats = extract_url_features(u)
            except Exception:
                continue
            w.writerow([u, int(y), t.isoformat()] + [f"{feats[n]:.6g}" for n in FEATURE_NAMES])
            if (i + 1) % 10000 == 0:
                print(f"[features] {i + 1}/{len(records)}")
    return out_path


def describe(path: str) -> dict:
    df = pd.read_csv(path, usecols=["label", "timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return dict(path=path, rows=int(len(df)), malicious=int(df["label"].sum()),
                malicious_pct=float(100 * df["label"].mean()),
                n_features=len(FEATURE_NAMES),
                first_seen=str(ts.min().date()), last_seen=str(ts.max().date()))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="output corpus name")
    p.add_argument("--malicious", required=True,
                   choices=["urlhaus", "phishtank", "phishstats", "threatfox", "csv"])
    p.add_argument("--urlhaus-variant", default="online", choices=["online", "recent", "full"])
    p.add_argument("--mal-file", help="manually downloaded feed file (skips download)")
    p.add_argument("--mal-csv", help="generic malicious CSV (with --malicious csv)")
    p.add_argument("--mal-url-col", default="url")
    p.add_argument("--mal-time-col", default="timestamp")
    p.add_argument("--abusech-key", default=None)
    p.add_argument("--phishtank-key", default=None)
    p.add_argument("--benign", default="hn", choices=["hn", "csv"])
    p.add_argument("--ben-csv")
    p.add_argument("--ben-url-col", default="url")
    p.add_argument("--ben-time-col", default="timestamp")
    p.add_argument("--max-malicious", type=int, default=15000)
    p.add_argument("--benign-ratio", type=float, default=2.0,
                   help="benign:malicious ratio (2.0 -> 33%% malicious)")
    p.add_argument("--align-bins", type=int, default=24,
                   help="time bins used to match benign to malicious timestamps")
    p.add_argument("--min-date", default=None, help="drop malicious records before YYYY-MM-DD")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    # --- malicious ---------------------------------------------------------
    if args.malicious == "urlhaus":
        mal = load_urlhaus(args.urlhaus_variant, args.refresh, args.abusech_key, args.mal_file)
    elif args.malicious == "threatfox":
        mal = load_threatfox(args.refresh, args.abusech_key, args.mal_file)
    elif args.malicious == "phishtank":
        mal = load_phishtank(args.refresh, args.phishtank_key, args.mal_file)
    elif args.malicious == "phishstats":
        mal = load_phishstats(args.max_malicious * 2, args.refresh)
    else:
        if not args.mal_csv:
            raise SystemExit("--malicious csv requires --mal-csv")
        mal = load_csv_records(args.mal_csv, 1, args.mal_url_col, args.mal_time_col)
    mal = _dedupe(mal)
    if args.min_date:
        cutoff = _parse_time(args.min_date)
        mal = [r for r in mal if r[2] >= cutoff]
    print(f"[malicious] {len(mal)} unique timestamped URLs from {args.malicious}")
    if not mal:
        raise SystemExit("no malicious records; check the feed / key / manual file")
    mal = _subsample_uniform_in_time(mal, args.max_malicious, args.seed)
    print(f"[malicious] kept {len(mal)} ({mal[0][2].date()} .. {mal[-1][2].date()})")

    # --- benign ------------------------------------------------------------
    if args.benign == "hn":
        ben = benign_from_hn([r[2] for r in mal], args.benign_ratio, args.align_bins,
                             args.refresh, args.name)
    else:
        if not args.ben_csv:
            raise SystemExit("--benign csv requires --ben-csv")
        ben = load_csv_records(args.ben_csv, 0, args.ben_url_col, args.ben_time_col)
    ben = _dedupe(ben)
    mal_urls = {r[0] for r in mal}
    ben = [r for r in ben if r[0] not in mal_urls]
    print(f"[benign] {len(ben)} unique timestamped URLs")

    records = mal + ben
    path = write_feature_csv(args.name, records, PROCESSED_DIR)
    meta = describe(path)
    meta.update(dict(
        malicious_source=args.malicious, benign_source=args.benign,
        urlhaus_variant=args.urlhaus_variant if args.malicious == "urlhaus" else None,
        benign_ratio=args.benign_ratio, align_bins=args.align_bins,
        retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        note=("Benign HN links are sampled bin-by-bin to match the malicious first-seen "
              "distribution; exact-URL duplicates removed; hosts shared across labels kept."),
    ))
    with open(os.path.join(PROCESSED_DIR, f"{args.name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] {args.name}: {meta['rows']} rows, {meta['malicious_pct']:.1f}% malicious, "
          f"{meta['n_features']} features, {meta['first_seen']}..{meta['last_seen']} -> {path}")


if __name__ == "__main__":
    main()
