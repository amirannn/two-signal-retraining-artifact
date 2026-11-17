"""Unit tests for the PSI signal."""

from __future__ import annotations

import numpy as np
import pytest

from driftpolicy.signals import psi_per_feature, psi_signal


def test_psi_identical_distributions_is_near_zero():
    # Many samples so the quantile-binning noise stays small.
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(5000, 5))
    cur = rng.normal(0, 1, size=(5000, 5))
    psi = psi_signal(ref, cur)
    assert psi < 0.02


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(200, 5))
    cur = rng.normal(3, 1, size=(200, 5))
    psi = psi_signal(ref, cur)
    assert psi > 0.5


def test_psi_per_feature_shape():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, size=(100, 4))
    cur = rng.normal(1, 1, size=(100, 4))
    per = psi_per_feature(ref, cur)
    assert per.shape == (4,)
    assert np.all(per >= 0)


def test_psi_requires_same_feature_count():
    ref = np.zeros((10, 3))
    cur = np.zeros((10, 4))
    with pytest.raises(ValueError):
        psi_signal(ref, cur)


def test_psi_max_agg():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, size=(200, 5))
    cur = rng.normal(2, 1, size=(200, 5))
    assert psi_signal(ref, cur, agg="max") >= psi_signal(ref, cur, agg="mean")