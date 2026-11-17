"""Metrics and statistical analysis (metrics + non-inferiority)."""

from .metrics import compute_metrics
from .noninferiority import non_inferiority_test, paired_bootstrap_ci

__all__ = ["compute_metrics", "non_inferiority_test", "paired_bootstrap_ci"]