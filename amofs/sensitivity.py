"""Sensitivity analyses for AMOFS (paired with NSGA-II):

    delta   : MEC stabiliser delta in {0.01, 0.05, 0.10}
    windows : number of temporal windows T in {4, 8, 12}
    lambda  : AGDM mixing weight in {0.3, 0.6, 0.9}
    cost    : cost-model misspecification -- the *search* uses perturbed costs
              (20 % of features re-tiered at random, or log-normal noise with
              sigma = 0.5), the *attack* uses the true cost model.

    python -m amofs.sensitivity --dataset data/processed/urlhaus_hn.csv --tag UH --n-runs 10
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import replace

import numpy as np
from joblib import Parallel, delayed

from .amofs import run_amofs
from .baselines import run_nsga2
from .config import PROFILES
from .data import Dataset, chronological_split, load_dataset
from .indicators import build_reference_front, hypervolume, igd
from .objectives import Evaluator
from .run import RESULTS, _mean_std, _score_subset, operating_point

SETTINGS = (
    [("delta", d) for d in (0.01, 0.05, 0.10)]
    + [("windows", t) for t in (4, 8, 12)]
    + [("lambda", l) for l in (0.3, 0.6, 0.9)]
    + [("cost", "retier20"), ("cost", "lognormal05")]
)


def _perturbed_costs(costs: np.ndarray, kind: str, rng) -> np.ndarray:
    c = costs.copy()
    if kind == "retier20":
        tiers = np.array([1.0, 3.0, 8.0])
        idx = rng.choice(len(c), size=max(1, int(0.2 * len(c))), replace=False)
        c[idx] = rng.choice(tiers, size=len(idx))
    elif kind == "lognormal05":
        c = c * np.exp(rng.normal(0.0, 0.5, size=len(c)))
    return c


def run_setting(ds: Dataset, cfg, seed: int, kind: str, value) -> dict:
    split = chronological_split(ds, cfg.dev_fraction, cfg.val_fraction, seed)
    params = deepcopy(cfg.amofs)
    delta, windows = cfg.mec_delta, cfg.n_windows
    search_ds = ds
    if kind == "delta":
        delta = float(value)
    elif kind == "windows":
        windows = int(value)
    elif kind == "lambda":
        params.lam = float(value)
    elif kind == "cost":
        rng = np.random.default_rng(seed + 777)
        search_ds = replace(ds, costs=_perturbed_costs(ds.costs, str(value), rng))
    kw = dict(k=cfg.knn_k, probe_train=cfg.probe_train, probe_val=cfg.probe_val, seed=seed)
    ev_search = Evaluator(search_ds, split, n_windows=windows, mec_delta=delta, **kw)
    ev_true = Evaluator(ds, split, n_windows=cfg.n_windows, mec_delta=cfg.mec_delta, **kw)
    d = ds.n_features
    res = run_amofs(ev_search, d, cfg.pop_size, cfg.generations, params, seed=seed, archive_cap=cfg.archive_cap)
    nX, _, _ = run_nsga2(ev_search, d, cfg.pop_size, cfg.generations, seed=seed, archive_cap=cfg.archive_cap)
    fronts = {"AMOFS": (res.arch_X, ev_true.evaluate_population(res.arch_X)),
              "NSGA-II": (nX, ev_true.evaluate_population(nX))}
    ref = build_reference_front([F for _, F in fronts.values()])
    out = dict(seed=seed, kind=kind, value=value, methods={})
    for name, (aX, aF) in fronts.items():
        bal, _ = operating_point(aX, aF, ref)
        m = dict(hv=hypervolume(aF, exact=True), igd=igd(aF, ref))
        m.update(_score_subset(ev_true, bal, cfg, seed))
        out["methods"][name] = m
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", default="DS")
    p.add_argument("--profile", default="fast", choices=sorted(PROFILES))
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--n-jobs", type=int)
    args = p.parse_args(argv)

    cfg = PROFILES[args.profile]()
    if args.n_jobs:
        cfg.n_jobs = args.n_jobs
    ds = load_dataset(args.dataset, os.path.splitext(os.path.basename(args.dataset))[0])
    n_jobs = cfg.n_jobs if cfg.n_jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    seeds = [cfg.base_seed + 20_000 + i for i in range(args.n_runs)]
    jobs = [(s, k, v) for (k, v) in SETTINGS for s in seeds]
    raw = Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_setting)(ds, cfg, s, k, v) for s, k, v in jobs)

    table = {}
    for k, v in SETTINGS:
        rows = [r for r in raw if r["kind"] == k and r["value"] == v]
        table[f"{k}={v}"] = {m: {met: _mean_std([r["methods"][m][met] for r in rows])
                                 for met in ("hv", "igd", "bacc", "features", "auec_direct", "temporal_stability")}
                             for m in ("AMOFS", "NSGA-II")}
    out_dir = os.path.join(RESULTS, args.tag)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    with open(os.path.join(out_dir, "sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(dict(settings=table, n_runs=args.n_runs, raw=raw), f, indent=1, default=float)

    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Setting & Method & HV$\\uparrow$ & BACC$\\uparrow$ & $|\\mathbf{x}|$ & AUEC$_{\\mathrm{dir}}\\downarrow$ \\\\",
             "\\midrule"]
    for setting, by_m in table.items():
        for i, m in enumerate(("AMOFS", "NSGA-II")):
            r = by_m[m]
            lab = setting.replace("delta=", "$\\delta=$").replace("windows=", "$T=$").replace("lambda=", "$\\lambda=$") \
                .replace("cost=retier20", "costs: 20\\% re-tiered").replace("cost=lognormal05", "costs: log-normal noise") \
                if i == 0 else ""
            lines.append(f"{lab} & {m} & {r['hv']['mean']:.3f}$\\pm${r['hv']['std']:.3f} & "
                         f"{r['bacc']['mean']:.3f} & {r['features']['mean']:.1f} & "
                         f"{r['auec_direct']['mean']:.3f}$\\pm${r['auec_direct']['std']:.3f} \\\\")
        lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out_dir, "tables", f"sensitivity_{args.tag}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[sensitivity] written to {out_dir}/sensitivity.json and tables/sensitivity_{args.tag}.tex")


if __name__ == "__main__":
    main()
