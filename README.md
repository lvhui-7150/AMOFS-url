# AMOFS — Adversary-aware Multi-Objective Feature Selection for malicious URL detection

Implementation and experimental materials for the manuscript
*Adversary-aware multi-objective feature selection with archive-guided search for malicious URL detection*.

This snapshot includes source code, raw and processed data, and stored per-seed outputs, summaries and tables. The stored outputs are archived runs; the corrected aggregate results used in the revised manuscript are maintained with the manuscript sources. No experiments were rerun for this repository update.

```
pip install -r requirements.txt
python -m amofs.run --smoke                                   # pipeline check on synthetic data (minutes)

# corpora with real timestamps (malicious feed + time-matched Hacker News benign links)
python -m amofs.prepare_data --name urlhaus_hn   --malicious urlhaus   --max-malicious 15000 [--abusech-key KEY]
python -m amofs.prepare_data --name phishtank_hn --malicious phishtank --max-malicious 15000 [--phishtank-key KEY]
python -m amofs.prepare_data --name threatfox_hn --malicious threatfox --max-malicious 15000 [--abusech-key KEY]

# main comparison + ablation (resumable, parallel over seeds)
python -m amofs.run --dataset data/processed/urlhaus_hn.csv   --tag UH --display "URLhaus--HN"
python -m amofs.run --dataset data/processed/phishtank_hn.csv --tag PT --display "PhishTank--HN"
python -m amofs.run --dataset data/processed/threatfox_hn.csv --tag TF --display "ThreatFox--HN"

python -m amofs.sensitivity --dataset data/processed/urlhaus_hn.csv --tag UH --n-runs 10
python -m amofs.make_figures --tags UH PT TF --out ../figures   # figures + results/macros.tex + datasets table
```

Outputs: `results/<TAG>/seeds/*.json` (raw), `results/<TAG>/summary.json`, `results/<TAG>/tables/*.tex`,
`results/macros.tex`, `figures/*.pdf`.

| Module | Content |
|---|---|
| `features.py` | 67 string-only URL features, each with a declared cost tier (cheap 1 / moderate 3 / expensive 8) |
| `prepare_data.py` | URLhaus / PhishTank / PhishStats / ThreatFox malicious feeds; Hacker News benign links via Algolia; time matching |
| `data.py` | loading, chronological development / out-of-time split, temporal windows |
| `objectives.py` | $f_{err}$ (k-NN probe), $f_{card}$, $f_{rob}$ (cheap-edit MEC), $f_{stab}$ (per-window reference subsets) |
| `amofs.py` | ANCF (progress-normalised schedule) + AGDM (archive-guided mutation) on NSGA-II selection, unbounded archive |
| `baselines.py` | NSGA-II, NSGA-II (2 obj.), MOEA/D, SPEA2, BPSO, BGWO (multi-weight), MI/RF rankings |
| `attack.py` | transfer attack (RF surrogate) and direct query-based attack under the cost model |
| `indicators.py`, `stats.py` | exact HV (pymoo) / IGD; Wilcoxon + BH + effect sizes + Friedman |
| `run.py`, `sensitivity.py`, `make_figures.py`, `export.py` | drivers and LaTeX/figure emitters |

Synthetic data (`--smoke`) exists only to verify the pipeline and must never be reported as a result.
