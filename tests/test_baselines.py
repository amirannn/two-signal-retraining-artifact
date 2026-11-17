"""Tests for label-free detector baselines."""

from __future__ import annotations

import numpy as np

from driftpolicy.policies import DetectorPolicy
from driftpolicy.signals import (
    SignalProvider,
    incremental_ks_signal,
    mmd_signal,
    ocsvm_signal,
)


def test_incremental_ks_detects_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, size=(100, 4))
    current = rng.normal(3, 1, size=(100, 4))
    assert incremental_ks_signal(reference, current) > 0.5


def test_mmd_detects_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, size=(50, 4))
    current = rng.normal(2, 1, size=(50, 4))
    assert mmd_signal(reference, current) > 0.01


def test_ocsvm_detects_outlier_shift():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, size=(100, 4))
    current = rng.normal(5, 1, size=(100, 4))
    assert ocsvm_signal(reference, current) > 0.2


def test_detector_policy_uses_reference_calibrated_threshold():
    rng = np.random.default_rng(4)
    batches = [rng.normal(0, 1, size=(30, 3)) for _ in range(4)]
    batches.append(rng.normal(3, 1, size=(30, 3)))
    provider = SignalProvider(batches, reference_period=3)
    threshold = provider.calibrate_detector_threshold("ks", percentile=0.9)
    policy = DetectorPolicy("ks", provider, threshold, 0.0, reference_period=3)
    retrain, info = policy.decide(4)
    assert info.detector_name == "ks"
    assert info.detector_value is not None
    assert info.detector_alert is True
    assert retrain is True
