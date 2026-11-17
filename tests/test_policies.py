"""Unit tests for the four retraining policies."""

from __future__ import annotations

import numpy as np

from driftpolicy.policies import (
    DriftAwarePolicy,
    PeriodicPolicy,
    StaticPolicy,
    TwoSignalPolicy,
    CalibratedSequentialPolicy,
)
from driftpolicy.signals import SignalProvider


def _make_batches(n_batches=12, n_features=6, drift_at=6, seed=0):
    rng = np.random.default_rng(seed)
    batches = []
    for i in range(n_batches):
        mean = 2.0 if i >= drift_at else 0.0
        batches.append(rng.normal(mean, 0.3, size=(50, n_features)))
    return batches


def _policy(provider, theta1=0.1, theta2=0.05, ref=3):
    return provider, theta1, theta2, ref


def test_static_retrains_only_first_batch():
    provider = SignalProvider(_make_batches(), reference_period=3)
    pol = StaticPolicy(provider, 0.1, 0.05, reference_period=3)
    assert pol.decide(0)[0] is True
    assert pol.decide(1)[0] is False
    assert pol.decide(10)[0] is False


def test_periodic_retrains_every_batch():
    provider = SignalProvider(_make_batches(), reference_period=3)
    pol = PeriodicPolicy(provider, 0.1, 0.05, reference_period=3)
    for k in range(12):
        assert pol.decide(k)[0] is True


def test_drift_aware_retrains_at_drift():
    batches = _make_batches(drift_at=6)
    provider = SignalProvider(batches, reference_period=3)
    theta2 = 0.02  # small so the drift boundary crosses it
    pol = DriftAwarePolicy(provider, 0.1, theta2, reference_period=3)
    decisions = [pol.decide(k)[0] for k in range(len(batches))]
    assert decisions[0] is True  # initial model
    assert not any(decisions[1:3])  # reference period: no retraining
    assert decisions[6] is True  # drift boundary detected


def test_two_signal_skips_silhouette_when_psi_low():
    # An impossibly high theta1 means the PSI gate never fires, so the
    # expensive silhouette stage must never be computed.
    rng = np.random.default_rng(7)
    batches = [rng.normal(0, 1, size=(50, 6)) for _ in range(10)]
    provider = SignalProvider(batches, reference_period=3)
    pol = TwoSignalPolicy(provider, theta1=10.0, theta2=0.05, reference_period=3)
    for k in range(3, 10):
        retrain, info = pol.decide(k)
        assert retrain is False
        assert info.silhouette_computed is False
    assert provider.n_silhouette_computes == 0


def test_calibrated_sequential_uses_reference_p_values():
    batches = _make_batches(drift_at=6)
    provider = SignalProvider(batches, reference_period=3)
    pol = CalibratedSequentialPolicy(
        provider, 0.1, 0.05, reference_period=3,
        screen_alpha=1.0, evidence_threshold=0.1, min_new_batches=1,
    )
    retrain, info = pol.decide(6)
    assert info.detector_name == "calibrated_sequential"
    assert info.silhouette_computed is True
    assert info.fusion_score is not None


def test_two_signal_requires_both_signals():
    batches = _make_batches(drift_at=6)
    provider = SignalProvider(batches, reference_period=3)
    theta1 = 0.0  # PSI gate always fires
    theta2 = 0.02
    pol = TwoSignalPolicy(provider, theta1, theta2, reference_period=3)
    retrain, info = pol.decide(6)
    assert retrain is True
    assert info.psi_alert is True
    assert info.delta_alert is True