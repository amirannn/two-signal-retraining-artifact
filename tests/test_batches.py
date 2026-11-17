"""Tests for temporal batching."""

from __future__ import annotations

import numpy as np
import pytest

from driftpolicy.data.batches import TemporalBatches


def test_from_timeseries_sorts_and_batches():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    dates = np.array([3, 1, 2, 0] * 30)
    batches = TemporalBatches.from_timeseries(X, dates, batch_size=50)
    assert batches.n_batches == 2
    assert batches.batch(0).shape == (50, 5)
    assert batches.batch(1).shape == (50, 5)


def test_trailing_partial_batch_dropped():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(105, 5))
    dates = np.arange(105)
    batches = TemporalBatches.from_timeseries(X, dates, batch_size=50)
    assert batches.n_batches == 2


def test_train_and_test_halves():
    rng = np.random.default_rng(0)
    batches = TemporalBatches([rng.normal(size=(50, 4)) for _ in range(3)])
    tr = batches.train_half(0)
    te = batches.test_half(0)
    assert tr.shape == (25, 4)
    assert te.shape == (25, 4)


def test_invalid_batches_rejected():
    with pytest.raises(ValueError):
        TemporalBatches([])
    with pytest.raises(ValueError):
        TemporalBatches([np.zeros((10, 2)), np.zeros((10, 3))])