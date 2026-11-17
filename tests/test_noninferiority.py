"""Tests for the paired bootstrap non-inferiority analysis."""

from __future__ import annotations

import numpy as np

from driftpolicy.evaluation.noninferiority import non_inferiority_test


def test_identical_scores_are_non_inferior():
    scores = np.array([0.8, 0.85, 0.9, 0.82, 0.88])
    ni = non_inferiority_test(scores, scores, margin=0.02, n_boot=500, seed=1)
    assert ni["non_inferior"] is True
    assert abs(ni["mean_diff"]) < 1e-12


def test_slightly_worse_is_still_non_inferior_within_margin():
    proposed = np.array([0.80, 0.82, 0.84, 0.86, 0.88])
    baseline = np.array([0.81, 0.83, 0.85, 0.87, 0.89])
    ni = non_inferiority_test(proposed, baseline, margin=0.02, n_boot=500, seed=2)
    assert ni["non_inferior"] is True


def test_large_degradation_is_not_non_inferior():
    proposed = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    baseline = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    ni = non_inferiority_test(proposed, baseline, margin=0.02, n_boot=500, seed=3)
    assert ni["non_inferior"] is False
    assert ni["mean_diff"] < -0.3


def test_misaligned_inputs_raise():
    import pytest

    with pytest.raises(ValueError):
        non_inferiority_test(np.array([0.5, 0.6]), np.array([0.5, 0.6, 0.7]))