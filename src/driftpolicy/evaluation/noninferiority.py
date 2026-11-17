"""Paired non-inferiority analysis (bootstrap).

Two resampling units are supported because they answer different questions:

* ``pooled_bootstrap_ci`` resamples the individual paired test samples.  It
  quantifies the finite-sample uncertainty of the *measured* pooled accuracy
  difference on the fixed evaluation stream.
* ``paired_bootstrap_ci`` resamples whole temporal windows (a block/cluster
  bootstrap).  It additionally absorbs the heterogeneity of the effect across
  time and is therefore the more conservative of the two.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    sided: str = "two",
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-window paired differences.

    ``sided="two"`` returns the 95% two-sided interval (2.5th/97.5th
    percentiles). ``sided="one"`` returns the 95% one-sided lower bound (5th
    percentile), which is what a non-inferiority claim calls for.
    """
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = diffs[idx].mean()
    if sided == "one":
        lo = float(np.quantile(means, alpha))
    elif sided == "two":
        lo = float(np.quantile(means, alpha / 2))
    else:
        raise ValueError(f"unknown sided: {sided!r} (use 'one' or 'two')")
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def pooled_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    sided: str = "one",
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of *pooled per-sample* differences.

    ``diffs`` holds one value per paired evaluation sample (e.g. +1 when the
    proposed model classified the sample correctly and the baseline did not,
    -1 for the reverse, 0 when both agreed).  Resampling is i.i.d. over
    samples, which is the natural unit for the pooled accuracy difference.
    """
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = diffs[idx].mean()
    if sided == "one":
        lo = float(np.quantile(means, alpha))
    elif sided == "two":
        lo = float(np.quantile(means, alpha / 2))
    else:
        raise ValueError(f"unknown sided: {sided!r} (use 'one' or 'two')")
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def pooled_non_inferiority_test(
    diffs: np.ndarray,
    margin: float = 0.02,
    n_boot: int = 1000,
    seed: int = 42,
    sided: str = "one",
) -> dict:
    """Non-inferiority of the pooled per-sample accuracy difference.

    Same decision rule as ``non_inferiority_test`` (certified when the lower
    bootstrap bound stays above ``-margin``) but computed on the pooled stream
    rather than on the distribution of per-window means.
    """
    diffs = np.asarray(diffs, dtype=float)
    lo, hi = pooled_bootstrap_ci(diffs, n_boot=n_boot, seed=seed, sided=sided)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "margin": float(margin),
        "non_inferior": bool(lo > -margin),
    }


def non_inferiority_test(
    proposed: np.ndarray,
    baseline: np.ndarray,
    margin: float = 0.02,
    n_boot: int = 1000,
    seed: int = 42,
    sided: str = "two",
) -> dict:
    """Paired non-inferiority of ``proposed`` vs ``baseline`` per-window scores.

    The proposal fixes the non-inferiority margin at δ = 2%. The verdict is
    "non-inferior" when the lower bound of the bootstrap CI of the mean
    difference (proposed - baseline) stays above -margin.
    """
    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if len(proposed) != len(baseline):
        raise ValueError("proposed and baseline must be aligned per window")
    diffs = proposed - baseline
    lo, hi = paired_bootstrap_ci(diffs, n_boot=n_boot, seed=seed, sided=sided)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "margin": float(margin),
        "non_inferior": bool(lo > -margin),
    }