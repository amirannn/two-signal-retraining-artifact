"""Tests for the scored two-signal policy."""

from __future__ import annotations

import numpy as np

from driftpolicy.policies import ScoredTwoSignalPolicy
from driftpolicy.signals import SignalProvider


def test_scored_policy_uses_emergency_psi_trigger():
    rng = np.random.default_rng(5)
    batches = [rng.normal(0, 1, size=(40, 4)) for _ in range(4)]
    batches.append(rng.normal(8, 1, size=(40, 4)))
    provider = SignalProvider(batches, reference_period=3)
    policy = ScoredTwoSignalPolicy(
        provider,
        theta1=0.01,
        theta2=0.05,
        reference_period=3,
        emergency_multiplier=2.0,
    )
    retrain, info = policy.decide(4)
    assert retrain is True
    assert info.emergency_trigger is True
    assert info.fusion_score is not None


def test_scored_policy_requires_score_when_not_emergency():
    rng = np.random.default_rng(6)
    batches = [rng.normal(0, 1, size=(40, 4)) for _ in range(5)]
    provider = SignalProvider(batches, reference_period=3)
    policy = ScoredTwoSignalPolicy(
        provider,
        theta1=0.1,
        theta2=10.0,
        reference_period=3,
        emergency_multiplier=100.0,
        score_threshold=10.0,
    )
    retrain, info = policy.decide(4)
    assert retrain is False
    assert info.emergency_trigger is False
    assert info.fusion_score is not None
    assert info.fusion_score < 10.0
