"""LaTeX emitters: table fragments and a macro file, generated from the
aggregated ``summary.json`` so that no experimental number is ever typed by
hand into the manuscript."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence

import numpy as np

_LATEX_METHOD = {"NSGA-II-2obj": "NSGA-II (2-obj.)", "MI-rank": "MI ranking", "RF-rank": "RF ranking"}


def _m(name: str) -> str:
    return _LATEX_METHOD.get(name, name)


def _fmt(mean: float, std: float | None = None, nd: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    s = f"{mean:.{nd}f}"
    if std is not None and not np.isnan(std):
        s += f"\\,$\\pm$\\,{std:.{nd}f}"
    return s


def _best(values: Dict[str, float], higher: bool) -> str:
    vals = {k: v for k, v in values.items() if v is not None and not np.isnan(v)}
    if not vals:
        return ""
    return max(vals, key=vals.get) if higher else min(vals, key=vals.get)


def _bold_if(s: str, cond: bool) -> str:
    return f"\\textbf{{{s}}}" if cond else s


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _stat(summary: dict, method: str, metric: str, key: str = "mean") -> float:
    try:
        return float(summary["metrics"][method][metric][key])
    except (KeyError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
def main_table(summary: dict, path: str) -> None:
    methods: List[str] = summary["methods"]
    cols = [("hv", "HV$\\uparrow$", True, 3), ("igd", "IGD$\\downarrow$", False, 3),
            ("bacc", "BACC$\\uparrow$", True, 3), ("f1", "$F_1\\uparrow$", True, 3),
            ("auc", "AUC$\\uparrow$", True, 3), ("features", "$|\\mathbf{x}|\\downarrow$", False, 1)]
    best = {m: _best({meth: _stat(summary, meth, m) for meth in methods}, hi) for m, _, hi, _ in cols}
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Method & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for meth in methods:
        cells = [_bold_if(_fmt(_stat(summary, meth, m), _stat(summary, meth, m, "std"), nd), best[m] == meth)
                 for m, _, _, nd in cols]
        lines.append(f"{_m(meth)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(path, "\n".join(lines) + "\n")


def evasion_table(summary: dict, attack: str, path: str) -> None:
    methods: List[str] = summary["methods"]
    budgets: Sequence[float] = summary["config"]["attack_budgets"]
    key = f"evasion_{attack}"
    auec_key = f"auec_{attack}"
    per_b = {meth: summary["metrics"][meth][key]["mean"] for meth in methods}
    best_b = [_best({meth: per_b[meth][j] for meth in methods}, False) for j in range(len(budgets))]
    best_auec = _best({meth: _stat(summary, meth, auec_key) for meth in methods}, False)
    lines = ["\\begin{tabular}{l" + "c" * (len(budgets) + 1) + "}", "\\toprule",
             "Method & " + " & ".join(f"$B={b:g}$" for b in budgets) + " & AUEC$\\downarrow$ \\\\", "\\midrule"]
    for meth in methods:
        cells = [_bold_if(f"{per_b[meth][j]:.1f}", best_b[j] == meth) for j in range(len(budgets))]
        cells.append(_bold_if(_fmt(_stat(summary, meth, auec_key), _stat(summary, meth, auec_key, "std")),
                              best_auec == meth))
        lines.append(f"{_m(meth)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(path, "\n".join(lines) + "\n")


def robustness_table(summary: dict, path: str) -> None:
    methods: List[str] = summary["methods"]
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "Method & MEC$\\uparrow$ & cheap (\\%) & moderate (\\%) & expensive (\\%) & "
             "$J_{\\mathrm{run}}\\uparrow$ & $1-f_{\\mathrm{stab}}\\uparrow$ \\\\", "\\midrule"]
    best_mec = _best({m: _stat(summary, m, "mec") for m in methods}, True)
    best_run = _best({m: summary["run_stability"].get(m, float("nan")) for m in methods}, True)
    best_tmp = _best({m: _stat(summary, m, "temporal_stability") for m in methods}, True)
    for meth in methods:
        share = summary["tier_share"][meth]
        cells = [_bold_if(_fmt(_stat(summary, meth, "mec"), _stat(summary, meth, "mec", "std"), 2), best_mec == meth),
                 f"{100 * share['cheap']:.0f}", f"{100 * share['moderate']:.0f}", f"{100 * share['expensive']:.0f}",
                 _bold_if(f"{summary['run_stability'][meth]:.2f}", best_run == meth),
                 _bold_if(_fmt(_stat(summary, meth, "temporal_stability"),
                               _stat(summary, meth, "temporal_stability", "std"), 2), best_tmp == meth)]
        lines.append(f"{_m(meth)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(path, "\n".join(lines) + "\n")


def stats_table(summary: dict, metrics: Sequence[str], path: str) -> None:
    labels = {"hv": "HV", "igd": "IGD", "bacc": "BACC", "auec_direct": "AUEC (direct)",
              "auec_transfer": "AUEC (transfer)", "features": "$|\\mathbf{x}|$"}
    baselines = [m for m in summary["methods"] if m != "AMOFS"]
    lines = ["\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule",
             "AMOFS vs. & " + " & ".join(labels.get(m, m) for m in metrics) + " \\\\", "\\midrule"]
    for b in baselines:
        cells = []
        for m in metrics:
            t = summary["tests"].get(m, {}).get(b)
            if t is None:
                cells.append("--"); continue
            p = t["p_adj"]
            ptxt = "$<\\!0.001$" if p < 0.001 else f"${p:.3f}$"
            cells.append(f"{ptxt} / {t['rank_biserial']:+.2f} / {t['a12']:.2f}")
        lines.append(f"{_m(b)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\multicolumn{" + str(len(metrics) + 1) + "}{l}{\\footnotesize Cells: "
              "BH-adjusted Wilcoxon $p$ / rank-biserial $r$ / Vargha--Delaney $\\hat A_{12}$.} \\\\",
              "\\end{tabular}"]
    _write(path, "\n".join(lines) + "\n")


def ablation_table(summary: dict, path: str) -> None:
    abl = summary.get("ablation", {})
    if not abl:
        return
    variants = list(abl)
    cols = [("hv", "HV$\\uparrow$", 3), ("igd", "IGD$\\downarrow$", 3), ("bacc", "BACC$\\uparrow$", 3),
            ("features", "$|\\mathbf{x}|$", 1), ("auec_direct", "AUEC$_{\\mathrm{dir}}\\downarrow$", 3),
            ("temporal_stability", "$1-f_{\\mathrm{stab}}\\uparrow$", 2)]
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Configuration & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for v in variants:
        cells = [_fmt(abl[v][m]["mean"], abl[v][m]["std"], nd) if m in abl[v] else "--" for m, _, nd in cols]
        lines.append(f"{v} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(path, "\n".join(lines) + "\n")


def dataset_table(summaries: Sequence[dict], path: str) -> None:
    lines = ["\\begin{tabular}{lccccccc}", "\\toprule",
             "Corpus & Period & \\#URLs & \\% mal. & $n$ & $T$ & Dev./OOT & \\% mal. (OOT) \\\\", "\\midrule"]
    for s in summaries:
        d = s["dataset"]
        lines.append(f"{d['display_name']} & {d['first_seen'][:7]}--{d['last_seen'][:7]} & "
                     f"{d['n_urls']:,} & {d['malicious_pct']:.1f} & {d['n_features']} & {d['windows']} & "
                     f"{d['n_dev']:,}/{d['n_test']:,} & {d['test_malicious_pct']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def _macro_name(*parts: str) -> str:
    out = ""
    for p in parts:
        p = p.replace("2obj", "twoobj").replace("F1", "Fone").replace("f1", "fone")
        out += re.sub(r"[^A-Za-z]", "", p)
    return out


def macros(summaries: Sequence[dict], path: str) -> None:
    """One \\newcommand per (dataset, method, metric) plus dataset-level facts."""
    lines = ["% auto-generated by amofs.export.macros -- do not edit by hand"]
    for s in summaries:
        ds = s["dataset"]["tag"]
        d = s["dataset"]
        for k, v in (("Urls", f"{d['n_urls']:,}"), ("MalPct", f"{d['malicious_pct']:.1f}"),
                     ("Nfeat", str(d["n_features"])), ("Windows", str(d["windows"])),
                     ("Ndev", f"{d['n_dev']:,}"), ("Ntest", f"{d['n_test']:,}"),
                     ("TestMalPct", f"{d['test_malicious_pct']:.1f}"),
                     ("First", d["first_seen"]), ("Last", d["last_seen"]), ("Cutoff", d["dev_cutoff"])):
            lines.append(f"\\newcommand{{\\{_macro_name('ds', ds, k)}}}{{{v}}}")
        for meth in s["methods"]:
            M = s["metrics"][meth]
            for metric, nd in (("hv", 3), ("igd", 3), ("bacc", 3), ("f1", 3), ("auc", 3), ("features", 1),
                               ("auec_transfer", 3), ("auec_direct", 3), ("mec", 2),
                               ("temporal_stability", 2), ("clean_detection", 3), ("runtime_s", 0)):
                if metric in M:
                    lines.append(f"\\newcommand{{\\{_macro_name(ds, meth, metric)}}}{{{M[metric]['mean']:.{nd}f}}}")
                    lines.append(f"\\newcommand{{\\{_macro_name(ds, meth, metric, 'sd')}}}{{{M[metric]['std']:.{nd}f}}}")
            for att in ("transfer", "direct"):
                for j, val in enumerate(M[f"evasion_{att}"]["mean"]):
                    lines.append(f"\\newcommand{{\\{_macro_name(ds, meth, 'ev', att, 'B' + chr(ord('a') + j))}}}{{{val:.1f}}}")
            lines.append(f"\\newcommand{{\\{_macro_name(ds, meth, 'runstab')}}}{{{s['run_stability'][meth]:.2f}}}")
            for tier, v in s["tier_share"][meth].items():
                lines.append(f"\\newcommand{{\\{_macro_name(ds, meth, tier, 'share')}}}{{{100 * v:.0f}}}")
        for metric, tests in s["tests"].items():
            for b, t in tests.items():
                lines.append(f"\\newcommand{{\\{_macro_name(ds, 'p', metric, b)}}}{{{t['p_adj']:.3f}}}")
                lines.append(f"\\newcommand{{\\{_macro_name(ds, 'r', metric, b)}}}{{{t['rank_biserial']:.2f}}}")
                lines.append(f"\\newcommand{{\\{_macro_name(ds, 'a', metric, b)}}}{{{t['a12']:.2f}}}")
        diag = s.get("amofs_diagnostics", {})
        if "gamma_hat" in diag:
            lines.append(f"\\newcommand{{\\{_macro_name(ds, 'gammahat')}}}{{{diag['gamma_hat']['mean']:.2f}}}")
            lines.append(f"\\newcommand{{\\{_macro_name(ds, 'gammahat', 'sd')}}}{{{diag['gamma_hat']['std']:.2f}}}")
        if "gens_to_match_nsga2" in diag:
            g = diag["gens_to_match_nsga2"]
            g_txt = str(g) if g is not None else "$>T_{\\max}$"
            lines.append(f"\\newcommand{{\\{_macro_name(ds, 'gensToMatch')}}}{{{g_txt}}}")
        for v, row in s.get("ablation", {}).items():
            for metric, nd in (("hv", 3), ("igd", 3), ("auec_direct", 3), ("temporal_stability", 2), ("bacc", 3)):
                if metric in row:
                    lines.append(f"\\newcommand{{\\{_macro_name(ds, 'abl', v, metric)}}}{{{row[metric]['mean']:.{nd}f}}}")
    _write(path, "\n".join(lines) + "\n")


def write_all(summary: dict, table_dir: str) -> None:
    tag = summary["dataset"]["tag"]
    main_table(summary, os.path.join(table_dir, f"main_{tag}.tex"))
    evasion_table(summary, "transfer", os.path.join(table_dir, f"evasion_transfer_{tag}.tex"))
    evasion_table(summary, "direct", os.path.join(table_dir, f"evasion_direct_{tag}.tex"))
    robustness_table(summary, os.path.join(table_dir, f"robustness_{tag}.tex"))
    stats_table(summary, ["hv", "igd", "bacc", "auec_direct", "features"], os.path.join(table_dir, f"stats_{tag}.tex"))
    ablation_table(summary, os.path.join(table_dir, f"ablation_{tag}.tex"))
