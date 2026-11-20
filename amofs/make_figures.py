"""Figures and cross-dataset LaTeX artefacts generated from real results.

    python -m amofs.make_figures --tags UH PT TF --out ../figures

Writes to --out:
    fig_convergence.pdf     HV vs generation (mean, 95% CI), gamma-hat trajectory, ANCF schedule
    fig_evasion.pdf         evasion rate vs budget, transfer (top) and direct (bottom), per corpus
    fig_overview.pdf        HV / BACC / |x| / AUEC per method and corpus
    fig_tiers.pdf           cost-tier composition of the selected subsets
    fig_features.pdf        AMOFS archive selection frequency per feature (first corpus)
and to results/:
    macros.tex, tables/datasets.tex
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import export  # noqa: E402
from .run import RESULTS  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})

COLORS = {"MI-rank": "#999999", "RF-rank": "#666666", "BPSO": "#984EA3", "BGWO": "#FF7F00",
          "NSGA-II-2obj": "#A6CEE3", "MOEA/D": "#377EB8", "SPEA2": "#4DAF4A", "NSGA-II": "#E41A1C",
          "AMOFS": "#000000"}
TIER_COLORS = {"cheap": "#fdae61", "moderate": "#abd9e9", "expensive": "#2c7bb6"}
EVOL = ["BPSO", "BGWO", "NSGA-II-2obj", "MOEA/D", "SPEA2", "NSGA-II", "AMOFS"]


def _load(tag: str) -> dict:
    with open(os.path.join(RESULTS, tag, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def _style(m):
    return dict(color=COLORS.get(m, "#333"), linewidth=1.6 if m == "AMOFS" else 1.0, zorder=10 if m == "AMOFS" else 2)


def fig_convergence(summaries: List[dict], out: str) -> None:
    s = summaries[0]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    ax = axes[0]
    for m in EVOL:
        c = s["convergence"].get(m)
        if not c:
            continue
        mean, ci = np.asarray(c["mean"]), np.asarray(c["ci"])
        g = np.arange(1, len(mean) + 1)
        ax.plot(g, mean, label=m, **_style(m))
        ax.fill_between(g, mean - ci, mean + ci, color=COLORS.get(m, "#333"), alpha=0.12, linewidth=0)
    ax.set_xlabel("generation"); ax.set_ylabel("archive hypervolume")
    ax.set_title(f"(a) Convergence, {s['dataset']['display_name']}", loc="left")
    ax.legend(ncol=2, frameon=False)
    ax = axes[1]
    for sm in summaries:
        d = sm["amofs_diagnostics"]
        if d.get("gamma_hat_traj"):
            mean, ci = np.asarray(d["gamma_hat_traj"]["mean"]), np.asarray(d["gamma_hat_traj"]["ci"])
            g = np.arange(1, len(mean) + 1)
            ax.plot(g, mean, label=sm["dataset"]["display_name"], linewidth=1.2)
            ax.fill_between(g, mean - ci, mean + ci, alpha=0.15, linewidth=0)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("generation"); ax.set_ylabel(r"$\hat\gamma_t$ w.r.t. final consensus")
    ax.set_title(r"(b) Archive-direction separation", loc="left"); ax.legend(frameon=False)
    ax = axes[2]
    d = s["amofs_diagnostics"]
    if d.get("a_mean"):
        g = np.arange(1, len(d["a_mean"]) + 1)
        ax.plot(g, d["a_mean"], color="black", label=r"$a(t)$")
        ax.plot(g, np.asarray(d["eta_mean"]) * (2.0 / max(s["config"]["amofs"]["eta_max"], 1e-9)),
                color="#E41A1C", linestyle="--", label=r"$\eta(t)$ (scaled)")
        ax2 = ax.twinx()
        ax2.plot(g, d["archive_size_mean"], color="#377EB8", linestyle=":", label="archive size")
        ax2.set_ylabel("archive size", color="#377EB8"); ax2.spines["top"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center right")
    ax.set_xlabel("generation"); ax.set_ylabel("convergence factor")
    ax.set_title("(c) ANCF schedule", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_convergence.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_evasion(summaries: List[dict], out: str) -> None:
    n = len(summaries)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 5.2), squeeze=False)
    for j, s in enumerate(summaries):
        budgets = np.asarray(s["config"]["attack_budgets"])
        for i, att in enumerate(("transfer", "direct")):
            ax = axes[i, j]
            for m in s["methods"]:
                cur = s["metrics"][m][f"evasion_{att}"]
                mean, sd = np.asarray(cur["mean"]), np.asarray(cur["std"])
                ci = 1.96 * sd / np.sqrt(s["n_seeds"])
                ax.plot(budgets, mean, marker="o", markersize=3, label=m, **_style(m))
                ax.fill_between(budgets, mean - ci, mean + ci, color=COLORS.get(m, "#333"), alpha=0.10, linewidth=0)
            ax.set_xlabel("attacker budget $B$ (cost units)")
            if j == 0:
                ax.set_ylabel(f"evasion rate (%), {att} attack")
            ax.set_title(f"({'ab'[i]}{j + 1}) {s['dataset']['display_name']}", loc="left")
            ax.set_xticks(budgets)
    axes[0, 0].legend(ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_evasion.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_overview(summaries: List[dict], out: str) -> None:
    metrics = [("hv", "HV $\\uparrow$"), ("bacc", "BACC (out-of-time) $\\uparrow$"),
               ("features", "selected features $\\downarrow$"), ("auec_direct", "AUEC, direct attack $\\downarrow$")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    methods = summaries[0]["methods"]
    width = 0.8 / len(methods)
    for ax, (met, lab) in zip(axes, metrics):
        x = np.arange(len(summaries))
        for i, m in enumerate(methods):
            means = [s["metrics"][m][met]["mean"] for s in summaries]
            stds = [s["metrics"][m][met]["std"] for s in summaries]
            ax.bar(x + i * width, means, width, yerr=stds, color=COLORS.get(m, "#333"), label=m,
                   edgecolor="white", linewidth=0.3, error_kw=dict(linewidth=0.5, capsize=1.5))
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([s["dataset"]["display_name"] for s in summaries], fontsize=7)
        ax.set_title(lab, loc="left")
        if met in ("hv", "bacc"):
            lo = min(s["metrics"][m][met]["mean"] - s["metrics"][m][met]["std"] for s in summaries for m in methods)
            ax.set_ylim(max(0, lo - 0.03), None)
    axes[0].legend(ncol=1, frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_overview.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_tiers(summaries: List[dict], out: str) -> None:
    fig, axes = plt.subplots(1, len(summaries), figsize=(3.4 * len(summaries), 2.8), squeeze=False)
    for ax, s in zip(axes[0], summaries):
        methods = s["methods"]
        bottom = np.zeros(len(methods))
        for tier in ("cheap", "moderate", "expensive"):
            vals = np.array([100 * s["tier_share"][m][tier] for m in methods])
            ax.bar(methods, vals, bottom=bottom, color=TIER_COLORS[tier], label=tier, edgecolor="white", linewidth=0.3)
            bottom += vals
        ax.set_ylabel("share of selected features (%)"); ax.set_title(s["dataset"]["display_name"], loc="left")
        ax.tick_params(axis="x", rotation=60)
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_tiers.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_features(summaries: List[dict], out: str, top: int = 25) -> None:
    s = summaries[0]
    d = s["amofs_diagnostics"]
    freq = np.asarray(d["final_sel_freq_mean"])
    names = d["feature_names"]
    from .features import FEATURE_TIERS
    order = np.argsort(-freq)[:top]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    colors = [TIER_COLORS.get(FEATURE_TIERS.get(names[i], "moderate"), "#999") for i in order]
    ax.bar(range(len(order)), freq[order], color=colors, edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([names[i] for i in order], rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("final archive selection frequency $s_i$")
    ax.set_title(f"AMOFS, {s['dataset']['display_name']}: top-{top} features by $s_i$ "
                 f"(colour = cost tier)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER_COLORS.values()]
    ax.legend(handles, list(TIER_COLORS), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_features.pdf"), bbox_inches="tight"); plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--out", default=os.path.join(RESULTS, "figures"))
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    summaries = [_load(t) for t in args.tags]
    fig_convergence(summaries, args.out)
    fig_evasion(summaries, args.out)
    fig_overview(summaries, args.out)
    fig_tiers(summaries, args.out)
    fig_features(summaries, args.out)
    export.macros(summaries, os.path.join(RESULTS, "macros.tex"))
    export.dataset_table(summaries, os.path.join(RESULTS, "tables", "datasets.tex"))
    print(f"[figures] written to {args.out}; macros -> {os.path.join(RESULTS, 'macros.tex')}")


if __name__ == "__main__":
    main()
