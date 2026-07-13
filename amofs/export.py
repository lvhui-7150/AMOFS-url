r"""Emit LaTeX table fragments that drop into the manuscript.

Each emitter writes a standalone tabular body that can be ``\input`` into the
corresponding table in ``sections/07-experiments.tex``, replacing ``\TBD``
cells. The best value per column is bolded automatically.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

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
