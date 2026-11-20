"""End-to-end experiment driver (paired design: every method sees the same
seed, split, evaluator and attack samples).

    python -m amofs.run --smoke
    python -m amofs.run --dataset data/processed/urlhaus_hn.csv --tag UH --display "URLhaus--HN" --profile fast
    python -m amofs.run --dataset ... --tag UH --aggregate-only      # re-export tables from saved seeds

Outputs under results/<tag>/:
    seeds/seed_<s>.json        raw per-seed results (resumable)
    ablation/seed_<s>.json     raw per-seed ablation results
    summary.json               aggregated numbers used by export/make_figures
    tables/*.tex               LaTeX fragments (\\input{} them)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from . import export
from .amofs import run_amofs
from .attack import EvasionAttacker, auec
from .baselines import BASELINES, METHOD_ORDER
from .config import PROFILES, ExperimentConfig
from .data import Dataset, chronological_split, load_dataset, make_synthetic
from .indicators import build_reference_front, hv_backend, hypervolume, igd
from .objectives import Evaluator, ObjectiveDropEvaluator
from .stats import compare_against_baselines, friedman

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")

ABLATION_VARIANTS = [
    ("Full AMOFS", dict(), ()),
    ("w/o ANCF", dict(use_ancf=False), ()),
    ("w/o AGDM", dict(use_agdm=False), ()),
    ("w/o ANCF \\& AGDM", dict(use_ancf=False, use_agdm=False), ()),
    ("w/o $f_{\\mathrm{rob}}$", dict(), (2,)),
    ("w/o $f_{\\mathrm{stab}}$", dict(), (3,)),
    ("w/o $f_{\\mathrm{rob}}$, $f_{\\mathrm{stab}}$", dict(), (2, 3)),
]


# ---------------------------------------------------------------------------
def operating_point(aX: np.ndarray, aF: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Balanced operating point: argmin of the sum of range-normalised objectives.

    Normalisation uses the union reference front of the run so that every
    method is scored on the same scale.  Also returns the accuracy-first point.
    """
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    score = ((aF - lo) / span).sum(axis=1)
    balanced = aX[int(np.argmin(score))]
    # accuracy-first: minimal f_err, ties broken by fewer features
    order = np.lexsort((aF[:, 1], aF[:, 0]))
    return balanced.astype(np.int8), aX[order[0]].astype(np.int8)


def _detector_metrics(clf, Xte_sel, yte) -> dict:
    pred = clf.predict(Xte_sel)
    try:
        auc = float(roc_auc_score(yte, clf.predict_proba(Xte_sel)[:, 1]))
    except ValueError:
        auc = float("nan")
    return dict(bacc=float(balanced_accuracy_score(yte, pred)),
                f1=float(f1_score(yte, pred, zero_division=0)), auc=auc)


def _score_subset(ev: Evaluator, mask: np.ndarray, cfg: ExperimentConfig, seed: int) -> dict:
    sp = ev.split
    atk = EvasionAttacker(ev.Xs[sp.tr], ev.y[sp.tr], ev.Xs[sp.te], ev.y[sp.te], mask, ev.costs,
                          seed=seed, max_samples=cfg.attack_max_samples, hgb_max_iter=cfg.hgb_max_iter)
    out = _detector_metrics(atk.deployed, ev.Xs[sp.te][:, atk.sel], ev.y[sp.te])
    curves = atk.curves(cfg.attack_budgets)
    out.update(evasion_transfer=curves["transfer"], evasion_direct=curves["direct"],
               auec_transfer=auec(curves["transfer"], cfg.attack_budgets),
               auec_direct=auec(curves["direct"], cfg.attack_budgets),
               clean_detection=curves["clean_detection"],
               features=float(mask.sum()), mec=ev.mec(mask),
               temporal_stability=1.0 - ev.f_stab(mask), f_vec=ev.evaluate(mask).tolist())
    return out


def run_one_seed(ds: Dataset, cfg: ExperimentConfig, seed: int, methods: Sequence[str]) -> dict:
    split = chronological_split(ds, cfg.dev_fraction, cfg.val_fraction, seed)
    ev = Evaluator(ds, split, k=cfg.knn_k, probe_train=cfg.probe_train, probe_val=cfg.probe_val,
                   n_windows=cfg.n_windows, mec_delta=cfg.mec_delta, seed=seed)
    d = ds.n_features
    fronts: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    logs: Dict[str, list] = {}
    runtime: Dict[str, float] = {}
    amofs_res = None
    for name in methods:
        t0 = time.perf_counter()
        if name == "AMOFS":
            amofs_res = run_amofs(ev, d, cfg.pop_size, cfg.generations, cfg.amofs, seed=seed,
                                  archive_cap=cfg.archive_cap)
            fronts[name] = (amofs_res.arch_X, amofs_res.arch_F)
            logs[name] = amofs_res.hv_log
        else:
            aX, aF, hv_log = BASELINES[name](ev, d, cfg.pop_size, cfg.generations, seed=seed,
                                             archive_cap=cfg.archive_cap)
            fronts[name] = (aX, aF)
            logs[name] = hv_log
        runtime[name] = time.perf_counter() - t0

    ref = build_reference_front([F for _, F in fronts.values()])
    out: dict = dict(seed=seed, n_train=int(len(split.tr)), n_val=int(len(split.va)),
                     n_test=int(len(split.te)), dev_cutoff_ns=split.cutoff_ns,
                     n_evaluations=ev.n_evaluations, methods={}, hv_logs=logs)
    for name, (aX, aF) in fronts.items():
        bal, acc = operating_point(aX, aF, ref)
        m = dict(hv=hypervolume(aF, exact=True), igd=igd(aF, ref), archive_size=int(aX.shape[0]),
                 runtime_s=runtime[name], mask=bal.tolist(), mask_acc=acc.tolist())
        m.update(_score_subset(ev, bal, cfg, seed))
        acc_scores = _score_subset(ev, acc, cfg, seed)
        m.update(bacc_acc=acc_scores["bacc"], features_acc=acc_scores["features"],
                 auec_direct_acc=acc_scores["auec_direct"])
        out["methods"][name] = m
    if amofs_res is not None:
        sf = np.vstack(amofs_res.sel_freq_log)                       # (gens, d)
        final = sf[-1]
        S = final >= 0.5
        gamma_t = np.minimum((2 * sf[:, S] - 1).min(axis=1) if S.any() else np.zeros(sf.shape[0]),
                             (1 - 2 * sf[:, ~S]).min(axis=1) if (~S).any() else np.zeros(sf.shape[0]))
        out["amofs"] = dict(gamma_hat=float(gamma_t[-1]), gamma_hat_traj=gamma_t.tolist(),
                            a_log=amofs_res.a_log, eta_log=amofs_res.eta_log,
                            archive_size_log=amofs_res.archive_size_log,
                            truncations=amofs_res.truncations, final_sel_freq=final.tolist(),
                            consensus_support=np.flatnonzero(S).tolist())
    return out


def run_ablation_seed(ds: Dataset, cfg: ExperimentConfig, seed: int) -> dict:
    split = chronological_split(ds, cfg.dev_fraction, cfg.val_fraction, seed)
    ev = Evaluator(ds, split, k=cfg.knn_k, probe_train=cfg.probe_train, probe_val=cfg.probe_val,
                   n_windows=cfg.n_windows, mec_delta=cfg.mec_delta, seed=seed)
    fronts = {}
    for label, overrides, disabled in ABLATION_VARIANTS:
        params = deepcopy(cfg.amofs)
        for k, v in overrides.items():
            setattr(params, k, v)
        search_ev = ObjectiveDropEvaluator(ev, disabled) if disabled else ev
        res = run_amofs(search_ev, ds.n_features, cfg.pop_size, cfg.generations, params, seed=seed,
                        archive_cap=cfg.archive_cap)
        aF = ev.evaluate_population(res.arch_X)                   # re-score on all four objectives
        fronts[label] = (res.arch_X, aF)
    ref = build_reference_front([F for _, F in fronts.values()])
    out = dict(seed=seed, variants={})
    for label, (aX, aF) in fronts.items():
        bal, _ = operating_point(aX, aF, ref)
        m = dict(hv=hypervolume(aF, exact=True), igd=igd(aF, ref))
        m.update(_score_subset(ev, bal, cfg, seed))
        out["variants"][label] = m
    return out


# ---------------------------------------------------------------------------
def _mean_std(values: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return dict(mean=float(np.nanmean(v)) if v.size else float("nan"),
                std=float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0, values=v.tolist())


def _curve_stats(curves: Sequence[Sequence[float]]) -> dict:
    arr = np.asarray(curves, dtype=float)
    return dict(mean=arr.mean(axis=0).tolist(),
                std=(arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])).tolist())


def _pairwise_jaccard(masks: Sequence[Sequence[int]]) -> float:
    M = np.asarray(masks, dtype=bool)
    if M.shape[0] < 2:
        return float("nan")
    vals = []
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            u = (M[i] | M[j]).sum()
            vals.append((M[i] & M[j]).sum() / u if u else 1.0)
    return float(np.mean(vals))


def _mean_ci(logs: Sequence[Sequence[float]]) -> dict:
    good = [np.asarray(l, dtype=float) for l in logs if len(l) > 1]
    if not good:
        return {}
    L = min(len(l) for l in good)
    arr = np.vstack([l[:L] for l in good])
    ci = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(L)
    return dict(mean=arr.mean(axis=0).tolist(), ci=ci.tolist())


def aggregate(ds: Dataset, cfg: ExperimentConfig, seeds: List[dict], abl: List[dict],
              tag: str, display: str) -> dict:
    methods = [m for m in METHOD_ORDER if m in seeds[0]["methods"]]
    scalar_keys = ("hv", "igd", "archive_size", "runtime_s", "bacc", "f1", "auc", "features", "mec",
                   "temporal_stability", "auec_transfer", "auec_direct", "clean_detection",
                   "bacc_acc", "features_acc", "auec_direct_acc")
    metrics, run_stab, tier_share = {}, {}, {}
    tiers = np.asarray(ds.tiers)
    for m in methods:
        rows = [s["methods"][m] for s in seeds]
        metrics[m] = {k: _mean_std([r[k] for r in rows]) for k in scalar_keys}
        metrics[m]["evasion_transfer"] = _curve_stats([r["evasion_transfer"] for r in rows])
        metrics[m]["evasion_direct"] = _curve_stats([r["evasion_direct"] for r in rows])
        masks = [r["mask"] for r in rows]
        run_stab[m] = _pairwise_jaccard(masks)
        sel = np.asarray(masks, dtype=bool)
        counts = sel.sum(axis=1).astype(float)
        tier_share[m] = {t: float(np.mean([(sel[i] & (tiers == t)).sum() / c if c else 0.0
                                           for i, c in enumerate(counts)]))
                         for t in ("cheap", "moderate", "expensive")}
    tests, fried = {}, {}
    for metric, higher in (("hv", True), ("igd", False), ("bacc", True), ("auc", True),
                           ("auec_direct", False), ("auec_transfer", False), ("features", False)):
        base = {m: np.asarray(metrics[m][metric]["values"]) for m in methods if m != "AMOFS"}
        tests[metric] = compare_against_baselines(np.asarray(metrics["AMOFS"][metric]["values"]), base, higher)
        fried[metric] = friedman({m: metrics[m][metric]["values"] for m in methods})

    convergence = {m: _mean_ci([s["hv_logs"][m] for s in seeds]) for m in methods}
    diag = {}
    if "amofs" in seeds[0]:
        diag["gamma_hat"] = _mean_std([s["amofs"]["gamma_hat"] for s in seeds])
        diag["gamma_hat_traj"] = _mean_ci([s["amofs"]["gamma_hat_traj"] for s in seeds])
        diag["a_mean"] = _mean_ci([s["amofs"]["a_log"] for s in seeds]).get("mean", [])
        diag["eta_mean"] = _mean_ci([s["amofs"]["eta_log"] for s in seeds]).get("mean", [])
        diag["archive_size_mean"] = _mean_ci([s["amofs"]["archive_size_log"] for s in seeds]).get("mean", [])
        diag["truncations_total"] = int(sum(s["amofs"]["truncations"] for s in seeds))
        freq = np.mean([s["amofs"]["final_sel_freq"] for s in seeds], axis=0)
        diag["final_sel_freq_mean"] = freq.tolist()
        diag["feature_names"] = ds.feature_names
        if convergence.get("NSGA-II") and convergence.get("AMOFS"):
            target = convergence["NSGA-II"]["mean"][-1]
            hit = [i + 1 for i, v in enumerate(convergence["AMOFS"]["mean"]) if v >= target]
            diag["gens_to_match_nsga2"] = hit[0] if hit else None

    ablation = {}
    if abl:
        for label, _, _ in ABLATION_VARIANTS:
            rows = [a["variants"][label] for a in abl if label in a["variants"]]
            if rows:
                ablation[label] = {k: _mean_std([r[k] for r in rows])
                                   for k in ("hv", "igd", "bacc", "features", "auec_direct",
                                             "auec_transfer", "temporal_stability", "mec")}
    s0 = seeds[0]
    first = np.datetime64(int(ds.t[0]), "ns").astype("datetime64[D]")
    last = np.datetime64(int(ds.t[-1]), "ns").astype("datetime64[D]")
    cutoff = np.datetime64(int(s0["dev_cutoff_ns"]), "ns").astype("datetime64[D]")
    n_dev = s0["n_train"] + s0["n_val"]
    dataset = dict(name=ds.name, tag=tag, display_name=display, n_urls=int(len(ds.y)),
                   malicious_pct=float(100 * ds.y.mean()), n_features=ds.n_features, windows=cfg.n_windows,
                   first_seen=str(first), last_seen=str(last), dev_cutoff=str(cutoff),
                   n_dev=int(n_dev), n_train=s0["n_train"], n_val=s0["n_val"], n_test=s0["n_test"],
                   test_malicious_pct=float(100 * ds.y[n_dev:].mean()),
                   tier_counts={t: int((tiers == t).sum()) for t in ("cheap", "moderate", "expensive")})
    return dict(dataset=dataset, config=cfg.to_dict(), hv_backend=hv_backend(), methods=methods,
                n_seeds=len(seeds), n_ablation_seeds=len(abl), metrics=metrics, run_stability=run_stab,
                tier_share=tier_share, tests=tests, friedman=fried, convergence=convergence,
                amofs_diagnostics=diag, ablation=ablation,
                operating_masks={m: [s["methods"][m]["mask"] for s in seeds] for m in methods})


# ---------------------------------------------------------------------------
def _load_or_run(path: str, fn, resume: bool, *args):
    if resume and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    res = fn(*args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, default=float)
    return res


def run_experiment(ds: Dataset, cfg: ExperimentConfig, tag: str, display: str, methods: Sequence[str],
                   run_ablation: bool = True, resume: bool = True, aggregate_only: bool = False) -> dict:
    out_dir = os.path.join(RESULTS, tag)
    seed_dir, abl_dir = os.path.join(out_dir, "seeds"), os.path.join(out_dir, "ablation")
    n_jobs = cfg.n_jobs if cfg.n_jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    seeds_list = [cfg.base_seed + i for i in range(cfg.n_runs)]
    abl_list = [cfg.base_seed + 10_000 + i for i in range(cfg.ablation_runs)] if run_ablation else []

    if aggregate_only:
        seed_res = [json.load(open(os.path.join(seed_dir, f"seed_{s}.json"), encoding="utf-8"))
                    for s in seeds_list if os.path.exists(os.path.join(seed_dir, f"seed_{s}.json"))]
        abl_res = [json.load(open(os.path.join(abl_dir, f"seed_{s}.json"), encoding="utf-8"))
                   for s in abl_list if os.path.exists(os.path.join(abl_dir, f"seed_{s}.json"))]
    else:
        print(f"[run] {ds.name}: n={len(ds.y)}, d={ds.n_features}, {cfg.n_runs} seeds x {len(methods)} methods, "
              f"pop={cfg.pop_size}, gens={cfg.generations}, workers={n_jobs}, HV backend={hv_backend()}")
        t0 = time.time()
        seed_res = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_load_or_run)(os.path.join(seed_dir, f"seed_{s}.json"), run_one_seed, resume,
                                  ds, cfg, s, list(methods))
            for s in seeds_list)
        print(f"[run] main comparison done in {(time.time() - t0) / 60:.1f} min")
        abl_res = []
        if run_ablation:
            t0 = time.time()
            abl_res = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_load_or_run)(os.path.join(abl_dir, f"seed_{s}.json"), run_ablation_seed, resume,
                                      ds, cfg, s)
                for s in abl_list)
            print(f"[run] ablation done in {(time.time() - t0) / 60:.1f} min")

    summary = aggregate(ds, cfg, seed_res, abl_res, tag, display) if seed_res else None
    if summary is None:
        raise SystemExit(f"no seed results found under {seed_dir}; run without --aggregate-only first")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    export.write_all(summary, os.path.join(out_dir, "tables"))
    print(f"[export] {out_dir}/summary.json and tables/ written")
    _print_brief(summary)
    return summary


def _print_brief(summary: dict) -> None:
    print(f"\n{summary['dataset']['display_name']}  (seeds={summary['n_seeds']})")
    print(f"{'method':14s} {'HV':>7s} {'IGD':>7s} {'BACC':>7s} {'|x|':>6s} {'AUEC-dir':>9s} {'AUEC-tr':>8s}")
    for m in summary["methods"]:
        M = summary["metrics"][m]
        print(f"{m:14s} {M['hv']['mean']:7.3f} {M['igd']['mean']:7.3f} {M['bacc']['mean']:7.3f} "
              f"{M['features']['mean']:6.1f} {M['auec_direct']['mean']:9.3f} {M['auec_transfer']['mean']:8.3f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", help="processed feature CSV")
    p.add_argument("--tag", default="DS", help="short identifier used in file and macro names (letters only)")
    p.add_argument("--display", default=None, help="corpus name shown in tables")
    p.add_argument("--profile", default="fast", choices=sorted(PROFILES))
    p.add_argument("--smoke", action="store_true", help="synthetic data, tiny budget, pipeline check")
    p.add_argument("--methods", default=",".join(METHOD_ORDER))
    p.add_argument("--n-runs", type=int)
    p.add_argument("--ablation-runs", type=int)
    p.add_argument("--pop-size", type=int)
    p.add_argument("--generations", type=int)
    p.add_argument("--n-windows", type=int)
    p.add_argument("--mec-delta", type=float)
    p.add_argument("--n-jobs", type=int)
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        cfg = ExperimentConfig.smoke()
        ds = make_synthetic(n_samples=800, n_features=20, seed=0, name="smoke")
        tag, display = "SMOKE", "Synthetic smoke test"
    else:
        if not args.dataset:
            raise SystemExit("provide --dataset PATH or --smoke")
        cfg = PROFILES[args.profile]()
        ds = load_dataset(args.dataset, os.path.splitext(os.path.basename(args.dataset))[0])
        tag, display = args.tag, args.display or ds.name
    for attr in ("n_runs", "ablation_runs", "pop_size", "generations", "n_windows", "mec_delta", "n_jobs"):
        v = getattr(args, attr)
        if v is not None:
            setattr(cfg, attr, v)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "AMOFS" not in methods:
        methods.append("AMOFS")
    run_experiment(ds, cfg, tag, display, methods, run_ablation=not args.no_ablation,
                   resume=not args.no_resume, aggregate_only=args.aggregate_only)


if __name__ == "__main__":
    main()
