"""Data-driven retraining policy for malware detection models.

This package implements the two-signal, two-stage retraining policy proposed
in the MSc thesis proposal "Improving Maintainability of Malware Detection
Models Using a Data-Driven Retraining":

1. A cheap distribution-shift gate: PSI of the current time window relative
   to a reference period (theta1).
2. An expensive structural confirmation: |ΔS| of the mean silhouette between
   consecutive MiniBatch K-Means clusterings (theta2).

Retraining is triggered only when *both* signals confirm a significant
change. The experimental protocol follows Mishra & Stamp (2025),
"Cluster Analysis and Concept Drift Detection in Malware", where a malware
family is split into chronological batches of 50 samples and compared against
a fixed "other family" set in a binary classification setup.
"""

from .config import ExperimentConfig, ModelConfig, SignalConfig, TrainingDataConfig
from .experiment import (
    ExperimentResult,
    PolicyResult,
    WindowRecord,
    run_experiment,
)
from .data.synthetic import make_synthetic_pair

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "SignalConfig",
    "TrainingDataConfig",
    "ExperimentResult",
    "PolicyResult",
    "WindowRecord",
    "run_experiment",
    "make_synthetic_pair",
]

__version__ = "0.1.0"