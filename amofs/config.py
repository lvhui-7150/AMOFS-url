"""Experiment configuration.

All numbers that the paper exposes as hyperparameters live here so that a run
is fully described by a single config object. Defaults match the ranges quoted
in Table (hyperparameters) of the manuscript; the `smoke` profile is a small,
fast configuration used to verify the pipeline end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AMOFSParams:
    """AMOFS-specific parameters (ANCF + AGDM)."""
    alpha: float = 2.0          # ANCF generation-decay shape
    beta: float = 1.5           # ANCF hypervolume-progress shape
    pc_min: float = 0.6
    pc_max: float = 0.95
    eta_min: float = 0.02       # lower bound on per-bit mutation scale (>0)
    eta_max: float = 0.4
    lam: float = 0.5            # AGDM mixing weight between s_i and c_i
    eps: float = 0.05           # AGDM probability clip (0 < eps < 1/2)
    use_ancf: bool = True       # ablation switch: adaptive HV-aware schedule
    use_agdm: bool = True       # ablation switch: archive-guided mutation


@dataclass
class ExperimentConfig:
    """Top-level configuration for one experimental run."""
    # search budget
    pop_size: int = 100
    generations: int = 200
    n_runs: int = 30            # independent seeds (paper uses R=30)
    ablation_runs: int = 30     # paper ablation also reports mean over R runs
    base_seed: int = 20240101

    # objectives
    knn_k: int = 5              # probe classifier neighbours
    n_windows: int = 5          # temporal windows T for the stability objective
    mec_delta: float = 0.05     # guard term delta in the MEC surrogate

    # evasion attack budgets (cost units); five increasing levels
    attack_budgets: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

    # data split
    test_size: float = 0.20
    val_size: float = 0.20      # taken from the non-test remainder

    amofs: AMOFSParams = field(default_factory=AMOFSParams)

    @staticmethod
    def smoke() -> "ExperimentConfig":
        """A tiny, fast configuration for pipeline verification."""
        return ExperimentConfig(
            pop_size=20,
            generations=25,
            n_runs=3,
            ablation_runs=2,
            n_windows=3,
        )
