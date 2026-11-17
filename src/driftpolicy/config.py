"""Configuration dataclasses for the retraining-policy experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SignalConfig:
    """Settings for the two drift signals (PSI and Silhouette).

    Attributes:
        reference_period: Number of initial batches used to calibrate the
            data-driven thresholds (theta1, theta2). No retraining decisions
            are made during this period.
        psi_bins: Number of bins used to compute PSI per feature.
        psi_agg: How to aggregate per-feature PSI values ("mean" or "max").
        n_clusters: Number of clusters for MiniBatch K-Means.
        minibatch_size: Mini-batch size for MiniBatch K-Means.
        silhouette_mode: How ΔS is defined -- "previous" compares the current
            consecutive-pair silhouette with the previous pair (as in the
            reference study), "reference" compares it with the mean silhouette
            of the reference period.
        threshold_method: "percentile" (high percentile of natural variation in
            the reference period) or "robust" (median + k * MAD).
        threshold_percentile: Percentile used when threshold_method="percentile".
        robust_k: Multiplier used when threshold_method="robust".
        default_psi_threshold: Fallback theta1 when calibration is impossible.
        default_delta_threshold: Fallback theta2 when calibration is impossible
            (0.05 is the empirical threshold of the reference study).
        seed: Random seed for clustering.
    """

    reference_period: int = 5
    psi_bins: int = 10
    psi_agg: Literal["mean", "max"] = "mean"
    n_clusters: int = 2
    minibatch_size: int = 25
    silhouette_mode: Literal["previous", "reference"] = "previous"
    threshold_method: Literal["percentile", "robust"] = "percentile"
    threshold_percentile: float = 0.90
    robust_k: float = 3.0
    default_psi_threshold: float = 0.1
    default_delta_threshold: float = 0.05
    # The base paper uses a fixed d_i > 0.05 threshold. Keep this switch for
    # paper-faithful replication; the thesis extension can use calibration.
    fixed_thresholds: bool = False
    # Mishra & Stamp's reference drift-aware policy uses d_i > 0.05.
    reference_drift_threshold: float = 0.05
    use_reference_threshold_for_drift_aware: bool = False
    seed: int = 42


@dataclass
class ModelConfig:
    """Classifier used in the reference protocol.

    name is one of: "random" (Random Forest), "mlp", "svm", "xgboost".
    """

    name: str = "random"
    params: dict = field(default_factory=dict)


@dataclass
class TrainingDataConfig:
    """How much historical positive-class data is retained after retraining."""

    mode: Literal["current", "rolling", "reservoir"] = "current"
    rolling_batches: int = 3
    reservoir_size: int = 250


@dataclass
class ExperimentConfig:
    """Top-level configuration of a full experiment run."""

    batch_size: int = 50
    model: ModelConfig = field(default_factory=ModelConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    training_data: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    non_inferiority_margin: float = 0.02  # δ = 2% as stated in the proposal
    bootstrap_iters: int = 1000
    seed: int = 42