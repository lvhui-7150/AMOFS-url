"""AMOFS: Adversary-aware Multi-Objective Feature Selection.

Reference implementation accompanying the paper
"AMOFS: An Adversary-Aware Multi-Objective Evolutionary Feature Selection
Framework with Convergence Guarantees for Malicious URL Detection".

The package is organised as:
    config       -- experiment configuration dataclasses
    data         -- dataset loading, feature pipeline, cost model
    objectives   -- the four objectives (error, cardinality, MEC, stability)
    indicators   -- hypervolume and IGD
    nsga_common  -- shared MOEA primitives (non-dominated sort, crowding)
    amofs        -- the AMOFS algorithm (ANCF + AGDM)
    baselines    -- NSGA-II, MOEA/D, SPEA2, BPSO, BGWO
    attack       -- black-box transfer evasion attack
    stats        -- Wilcoxon + Benjamini-Hochberg + effect sizes
    export       -- LaTeX table emitters
    run          -- end-to-end experiment driver (python -m amofs.run)
"""

__version__ = "0.1.0"