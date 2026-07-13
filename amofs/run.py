"""End-to-end experiment driver.

Usage
-----
    python -m amofs.run --smoke
    python -m amofs.prepare_data --dataset urlhaus_majestic --limit-per-class 2500
    python -m amofs.run --dataset data/processed/urlhaus_majestic.csv --name URLhausMajestic

Outputs (under results/):
    tables/main_<name>.tex          main comparison (HV/IGD/BACC/F1)
    tables/extended_<name>.tex      AUC/stability/features/latency
    tables/evasion_<name>.tex       evasion versus attacker budget
    tables/ablation_<name>.tex      AMOFS ablation
    tables/dataset_<name>.tex       dataset row for the manuscript
    tables/hyper_<name>.tex         selected hyperparameters
    tables/stats_hv_<name>.tex      AMOFS significance versus baselines on HV
    figures/convergence_<name>.png  HV convergence and AGDM evidence
    results/<name>_summary.json     raw aggregated numbers
"""
from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import export
from .amofs import run_amofs
from .attack import evasion_curve
from .baselines import BASELINES
from .config import ExperimentConfig
from .data import Dataset, load_dataset, make_synthetic
from .indicators import build_reference_front, hypervolume, igd
from .objectives import Evaluator
from .stats import compare_against_baselines

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


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

    export.main_comparison_table(methods, agg, os.path.join(table_dir, f"main_{name}.tex"))
    export.extended_metrics_table(methods, agg, os.path.join(table_dir, f"extended_{name}.tex"))
    export.evasion_table(methods, evasion_mean, os.path.join(table_dir, f"evasion_{name}.tex"))
    export.dataset_table(_dataset_summary(ds, cfg), os.path.join(table_dir, f"dataset_{name}.tex"))
    export.hyperparameter_table(_selected_hyperparameters(cfg), os.path.join(table_dir, f"hyper_{name}.tex"))

    amofs_hv = np.array(store["AMOFS"]["hv"], dtype=float)
    baseline_hv = {m: np.array(store[m]["hv"], dtype=float) for m in methods if m != "AMOFS"}
    comp = compare_against_baselines(amofs_hv, baseline_hv, higher_is_better=True)
    export.stats_table("HV", comp, os.path.join(table_dir, f"stats_hv_{name}.tex"))

    ablation = {}
    if run_ablation:
        ablation = _ablation(ds, cfg)
        order = ["Full AMOFS", "w/o ANCF", "w/o AGDM", "w/o ANCF \\& AGDM", "w/o $f_{\\mathrm{rob}}$ objective"]
        export.ablation_table(ablation, order, os.path.join(table_dir, f"ablation_{name}.tex"))

    _plot_convergence(hv_logs, amofs_selfreq_logs, figure_dir, name)

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


def _plot_convergence(hv_logs, selfreq_logs, figure_dir, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figure_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for method, logs in hv_logs.items():
        good = [np.asarray(log, dtype=float) for log in logs if len(log)]
        if not good:
            continue
        min_len = min(len(log) for log in good)
        arr = np.vstack([log[:min_len] for log in good])
        mean = arr.mean(axis=0)
        ci = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
        gens = np.arange(1, min_len + 1)
        ax[0].plot(gens, mean, label=method, linewidth=1.4 if method == "AMOFS" else 1.0)
        if method == "AMOFS":
            ax[0].fill_between(gens, mean - ci, mean + ci, alpha=0.2)
    ax[0].set_xlabel("generation")
    ax[0].set_ylabel("archive hypervolume")
    ax[0].set_title("Convergence")
    ax[0].legend(fontsize=8)

    if selfreq_logs:
        min_len = min(log.shape[0] for log in selfreq_logs)
        arr = np.stack([log[:min_len] for log in selfreq_logs], axis=0)
        final_freq = arr[:, -1, :].mean(axis=0)
        recovered = final_freq >= np.median(final_freq)
        series = arr[:, :, recovered].mean(axis=2)
        mean = series.mean(axis=0)
        ci = 1.96 * series.std(axis=0, ddof=1) / np.sqrt(series.shape[0]) if series.shape[0] > 1 else np.zeros_like(mean)
        gens = np.arange(1, min_len + 1)
        gamma = float(np.min(final_freq[recovered])) if np.any(recovered) else 0.0
        ax[1].plot(gens, mean, color="tab:blue")
        ax[1].fill_between(gens, mean - ci, mean + ci, color="tab:blue", alpha=0.2)
        ax[1].axhline(gamma, color="grey", linestyle="--", linewidth=1.0)
        ax[1].text(gens[-1], gamma, f" gamma={gamma:.2f}", va="bottom", fontsize=8)
    ax[1].set_xlabel("generation")
    ax[1].set_ylabel("mean archive frequency")
    ax[1].set_title("AGDM guidance evidence")

    fig.tight_layout()
    out = os.path.join(figure_dir, f"convergence_{name}.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[export] wrote {out}")


def _apply_cli_overrides(cfg: ExperimentConfig, args) -> ExperimentConfig:
    if args.n_runs is not None:
        cfg.n_runs = args.n_runs
    if args.ablation_runs is not None:
        cfg.ablation_runs = args.ablation_runs
    if args.pop_size is not None:
        cfg.pop_size = args.pop_size
    if args.generations is not None:
        cfg.generations = args.generations
    if args.knn_k is not None:
        cfg.knn_k = args.knn_k
    if args.n_windows is not None:
        cfg.n_windows = args.n_windows
    if args.mec_delta is not None:
        cfg.mec_delta = args.mec_delta
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="path to a feature CSV")
    parser.add_argument("--name", default="dataset")
    parser.add_argument("--synthetic", action="store_true", help="run full config on synthetic data")
    parser.add_argument("--smoke", action="store_true", help="fast synthetic pipeline check")
    parser.add_argument("--n-runs", type=int)
    parser.add_argument("--ablation-runs", type=int)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--knn-k", type=int)
    parser.add_argument("--n-windows", type=int)
    parser.add_argument("--mec-delta", type=float)
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        cfg = ExperimentConfig.smoke()
        ds = make_synthetic(n_samples=500, n_features=20, seed=0, name="smoke")
    elif args.synthetic:
        cfg = ExperimentConfig()
        ds = make_synthetic(name="synthetic")
    else:
        if not args.dataset:
            raise SystemExit("provide --dataset PATH or use --smoke/--synthetic")
        cfg = ExperimentConfig()
        ds = load_dataset(args.dataset, args.name)

    cfg = _apply_cli_overrides(cfg, args)
    run_experiment(ds, cfg, run_ablation=not args.no_ablation)


if __name__ == "__main__":
    main()
