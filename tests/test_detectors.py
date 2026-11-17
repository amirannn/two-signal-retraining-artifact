"""Tests for the standard sequential detectors (ADWIN, KSWIN)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftpolicy.detectors import ADWIN, KSWIN, novelty_scores  # noqa: E402


def test_novelty_scores_are_zero_on_reference_samples():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(200, 5))
    scores = novelty_scores(reference, reference)
    assert scores.shape == (200,)
    # The mean squared z-score of the reference samples is 1 by construction.
    assert np.isclose(float(scores.mean()), 1.0, atol=0.02)


def test_novelty_scores_grow_with_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(300, 4))
    shifted = reference + 3.0
    assert novelty_scores(reference, shifted).mean() > novelty_scores(reference, reference).mean()


def test_adwin_stays_quiet_on_stationary_stream():
    rng = np.random.default_rng(1)
    detector = ADWIN(delta=0.002)
    alerts = [detector.update(v) for v in rng.normal(size=400)]
    assert sum(alerts) <= 1


def test_adwin_detects_mean_shift():
    rng = np.random.default_rng(1)
    detector = ADWIN(delta=0.002)
    alerts = [detector.update(v) for v in rng.normal(size=300)]
    shifted = [detector.update(v) for v in rng.normal(loc=2.0, size=200)]
    assert any(shifted)
    assert not any(alerts)


def test_adwin_drops_old_data_after_a_change():
    rng = np.random.default_rng(2)
    detector = ADWIN(delta=0.002)
    inserted = 0
    for value in rng.normal(size=400):
        detector.update(value)
        inserted += 1
    assert detector.n == inserted  # no drop while the stream is stationary
    for value in rng.normal(loc=3.0, size=400):
        detector.update(value)
        inserted += 1
    assert detector.n < inserted  # the pre-change window was discarded


def test_kswin_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(3)
    quiet = KSWIN(seed=3)
    fired = False
    for _ in range(20):
        fired = fired or quiet.update_many(rng.normal(size=50))
    assert not fired

    detector = KSWIN(seed=3)
    for _ in range(6):
        detector.update_many(rng.normal(size=50))
    fired = any(detector.update_many(rng.normal(loc=2.0, size=50)) for _ in range(4))
    assert fired


def test_detector_factories_expose_the_batch_interface():
    from driftpolicy.detectors import DETECTOR_FACTORIES

    for seed, factory in DETECTOR_FACTORIES.items():
        detector = factory(42)
        assert hasattr(detector, "update_many")
        assert detector.update_many([0.0] * 10) in (False, True)
