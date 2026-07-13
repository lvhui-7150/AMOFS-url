# AMOFS: Adversary-Aware Multi-Objective Feature Selection

**AMOFS** is an evolutionary feature selection framework designed for **malicious URL detection**. It simultaneously optimises four objectives — classification accuracy, feature cardinality, adversarial robustness, and temporal stability — using a novel combination of **Adaptive Normalised Convergence Factor (ANCF)** scheduling and **Archive-Guided Discrete Mutation (AGDM)**.

## Overview

| Aspect | Detail |
|--------|--------|
| **Problem** | Select a robust, stable, and economical subset of URL features for malware/phishing detection |
| **Approach** | Multi-objective evolutionary algorithm (MOEA) with adversary-aware mutation |
| **Objectives** | Classification error, feature cardinality, minimum evasion cost (MEC), temporal stability |
| **Baselines** | NSGA-II, MOEA/D, SPEA2, BPSO, BGWO |
| **Paper** | Accompanying manuscript: *"AMOFS: An Adversary-Aware Multi-Objective Evolutionary Feature Selection Framework with Convergence Guarantees for Malicious URL Detection"* |

## Project Structure

```
code/
├── amofs_all.py          # Single-file reference implementation (all modules merged)
├── requirements.txt      # Python dependencies
├── amofs/                # Package source
│   ├── __init__.py       # Package entry, module overview
│   ├── config.py         # ExperimentConfig & AMOFSParams dataclasses
│   ├── data.py           # Dataset loading, feature pipeline, cost model, synthetic data
│   ├── features.py       # URL feature extraction (38 lexical / syntactic / semantic features)
│   ├── objectives.py     # Four-objective evaluator (error, cardinality, MEC, stability)
│   ├── indicators.py     # Hypervolume (Monte-Carlo) and IGD metrics
│   ├── nsga_common.py    # Shared MOEA primitives: non-dominated sort, crowding distance, archive
│   ├── amofs.py          # AMOFS algorithm: ANCF + AGDM (Algorithm 1)
│   ├── baselines.py      # NSGA-II, MOEA/D, SPEA2, BPSO, BGWO implementations
│   ├── attack.py         # Black-box transfer evasion attack simulation
│   ├── stats.py          # Wilcoxon signed-rank + Benjamini-Hochberg + effect sizes
│   ├── export.py         # LaTeX table emitters for the manuscript
│   └── run.py            # End-to-end experiment driver
├── results/              # Generated outputs
│   ├── tables/           # LaTeX table fragments (.tex)
│   └── figures/          # Publication-quality plots (.png)
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| numpy | 1.23 | Numerical computing |
| pandas | 1.5 | Data loading and manipulation |
| scikit-learn | 1.2 | kNN classifier, GBM downstream eval, StandardScaler |
| scipy | 1.9 | Wilcoxon statistical test |
| matplotlib | 3.6 | Plot generation |
| seaborn | 0.12 | Supplementary visualisation |

## Quick Start

### Smoke test (fast pipeline verification)

```bash
python -m amofs.run --smoke
```

Runs a small synthetic dataset (500 samples, 20 features) with reduced population and generations to verify the full pipeline end-to-end.

### Run on a real dataset

```bash
python -m amofs.run --dataset data/processed/urlhaus_majestic.csv --name URLhausMajestic
```

Or run the single-file reference:

```bash
python amofs_all.py --dataset data/processed/urlhaus_majestic.csv --name URLhausMajestic
```

### Run on synthetic data

```bash
python -m amofs.run --synthetic
```

### Custom configuration

```bash
python -m amofs.run --dataset data/processed/phishing_majestic.csv \
    --name PhishingMajestic \
    --pop-size 100 --generations 200 --n-runs 30 \
    --knn-k 5 --n-windows 5 --mec-delta 0.05
```

## Four Objectives

AMOFS minimises four objectives simultaneously:

1. **Classification Error** (`f_err`) — `1 − balanced_accuracy` of a k-NN probe classifier on held-out validation data
2. **Feature Cardinality** (`f_card`) — fraction of features selected (penalises unnecessary features)
3. **Robustness / MEC** (`f_rob`) — normalised minimum evasion cost; penalises subsets vulnerable to cheap attacker perturbations
4. **Temporal Stability** (`f_stab`) — `1 − mean Jaccard overlap` of selected features across temporal windows; penalises unstable selections

## Threat Model

AMOFS evaluates feature subsets against a **black-box transfer evasion attack**:

- The attacker trains a surrogate detector on an independent data split
- Features are perturbed greedily from cheapest to most effective (ratio `d_i / g_i`)
- Total perturbation is bounded by an attacker budget `B`
- The evasion rate at budget `B` is the fraction of test malicious samples flipped to benign

Modification costs follow an ordinal scale:

| Tier | Cost | Examples |
|------|------|----------|
| Cheap | 1.0 | URL length, entropy, token counts, path/query properties |
| Moderate | 3.0 | Host length, subdomain count, HTTPS flag, port |
| Expensive | 8.0 | IP-host, host entropy, punycode, shortener tokens, TLD length |

## Key Algorithms

### ANCF (Adaptive Normalised Convergence Factor)

A generation-dependent schedule that adapts crossover probability `p_c` and mutation scale `η` based on hypervolume progress:

- **Decay factor** `g = (1 − t/T)^α` — drives convergence over generations
- **HV-progress factor** `h = (1 − HV/HV_ref)^β` — accelerates search when progress stalls

### AGDM (Archive-Guided Discrete Mutation)

Mutation probabilities are guided by archive statistics:

- **On-score** `s_i` — global selection frequency of feature `i` in the archive
- **Off-score** `c_i` — diversity-weighted selection frequency
- Per-bit mutation probabilities `p_on` and `p_off` are clipped to `[η·ε, η·(1−ε)]`

## Output

After a full run, results are written to the `results/` directory:

**Tables** (`results/tables/`):
- `main_<name>.tex` — HV, IGD, Balanced Accuracy, F1
- `extended_<name>.tex` — AUC, stability, feature count, latency
- `evasion_<name>.tex` — evasion rates across attacker budgets
- `ablation_<name>.tex` — AMOFS component ablations
- `dataset_<name>.tex` — dataset statistics
- `hyper_<name>.tex` — selected hyperparameters
- `stats_hv_<name>.tex` — Wilcoxon significance vs. baselines

**Figures** (`results/figures/`):
- `convergence_<name>.png` — HV convergence and AGDM guidance evidence

**Summary** (`results/`):
- `<name>_summary.json` — complete raw data, metrics, evasion curves, ablation results

## Ablation Study

The ablation evaluates five variants over 30 independent runs:

| Variant | Description |
|---------|-------------|
| Full AMOFS | ANCF + AGDM + robustness objective |
| w/o ANCF | Fixed generation schedule |
| w/o AGDM | Uniform bitflip mutation |
| w/o ANCF & AGDM | Both components removed |
| w/o f_rob objective | Robustness objective disabled |

## Reproducing Experiments

1. Prepare your feature CSV with columns: numeric features, `label` (0=benign, 1=malicious), and optionally `timestamp`
2. Run the experiment driver:

```bash
python -m amofs.run --dataset path/to/your_dataset.csv --name MyDataset
```

3. For manuscript-quality tables, the `export.py` module emits ready-to-`\input{}` LaTeX fragments

## Synthetic Data

The `--smoke` and `--synthetic` flags generate internally-consistent synthetic datasets for pipeline verification. These are **not substitutes** for real-world corpora and should not be reported as measurement results.

## License

This repository accompanies a research manuscript. For academic use, please cite the accompanying paper.

```
