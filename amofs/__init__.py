"""AMOFS -- Adversary-aware Multi-Objective Feature Selection for malicious URL detection.

Modules
-------
features      URL feature extractor with a declared cost tier per feature
prepare_data  builds timestamped corpora (URLhaus / PhishTank / ThreatFox + Hacker News benign)
data          loading, chronological split, temporal windows
objectives    the four objectives (error, cardinality, cheap-edit evasion cost, temporal instability)
amofs         ANCF schedule + Archive-Guided Discrete Mutation
baselines     NSGA-II, NSGA-II (2 obj.), MOEA/D, SPEA2, BPSO, BGWO, MI/RF rankings
attack        transfer and direct (query-based) evasion attacks
indicators    hypervolume / IGD
stats         Wilcoxon + Benjamini-Hochberg + effect sizes + Friedman
run           experiment driver (paired seeds, resumable, parallel)
sensitivity   delta / T / lambda / cost-model sensitivity
make_figures  figures, macros.tex and dataset table from real results
export        LaTeX emitters
"""
__version__ = "2.0.0"
