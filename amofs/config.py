"""Experiment configuration (all hyper-parameters quoted in the manuscript).

Three profiles:
    smoke  - minutes, synthetic data, verifies the pipeline end to end
    fast   - a few hours on a desktop; the configuration used for the paper
    paper  - alias of ``fast`` kept for clarity in scripts
    large  - pop 100 / 200 generations for a multi-core server
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class AMOFSParams:
    alpha: float = 2.0      # ANCF generation-decay exponent
    beta: float = 1.0       # ANCF progress exponent
    window: int = 5         # ANCF progress window w (generations)
    pc_min: float = 0.70
    pc_max: float = 0.90
    eta_min: float = 0.05   # lower bound of the per-bit mutation scale (> 0)
    eta_max: float = 0.30
    lam: float = 0.60       # AGDM mixing weight between s_i and c_i
    eps: float = 0.05       # AGDM clipping, 0 < eps < 1/2
    use_ancf: bool = True   # ablation switch
    use_agdm: bool = True   # ablation switch


@dataclass
class ExperimentConfig:
    # search budget (identical for every method)
    pop_size: int = 50
    generations: int = 100
    n_runs: int = 30
    ablation_runs: int = 30
    base_seed: int = 20260101
    n_jobs: int = -1                    # joblib workers (-1 = all cores - 1)

    # objectives
    knn_k: int = 5
    probe_train: int = 2000             # stratified subsample for the kNN probe
    probe_val: int = 1000
    n_windows: int = 8                  # temporal windows T
    mec_delta: float = 0.05

    # data split
    dev_fraction: float = 0.70          # chronological: first 70 % = development
    val_fraction: float = 0.25          # of the development period -> validation
    archive_cap: int = 0                # 0 = unbounded archive (Assumption A1)

    # attack
    attack_budgets: Tuple[float, ...] = (2.0, 4.0, 8.0, 12.0, 16.0)
    attack_max_samples: int = 1000      # malicious test URLs attacked per run

    # downstream detector
    hgb_max_iter: int = 200

    amofs: AMOFSParams = field(default_factory=AMOFSParams)

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- profiles -----------------------------------------------------
    @staticmethod
    def smoke() -> "ExperimentConfig":
        return ExperimentConfig(pop_size=12, generations=8, n_runs=2, ablation_runs=2,
                                probe_train=300, probe_val=150, n_windows=3,
                                attack_max_samples=100, hgb_max_iter=50, n_jobs=1)

    @staticmethod
    def fast() -> "ExperimentConfig":
        return ExperimentConfig()

    @staticmethod
    def large() -> "ExperimentConfig":
        return ExperimentConfig(pop_size=100, generations=200, probe_train=3000, probe_val=1500)


PROFILES = {"smoke": ExperimentConfig.smoke, "fast": ExperimentConfig.fast,
            "paper": ExperimentConfig.fast, "large": ExperimentConfig.large}
