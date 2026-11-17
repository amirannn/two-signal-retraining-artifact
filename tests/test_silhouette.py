"""Unit tests for the silhouette signal and the signal provider."""

from __future__ import annotations

import numpy as np

from driftpolicy.signals import (
    SignalProvider,
    clustering_silhouette,
)


def test_separated_clusters_have_high_silhouette():
    rng = np.random.default_rng(0)
    a = rng.normal(-3, 0.5, size=(50, 4))
    b = rng.normal(3, 0.5, size=(50, 4))
    X = np.vstack([a, b])
    s = clustering_silhouette(X, n_clusters=2)
    assert s > 0.5


def test_single_blob_has_lower_silhouette():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(100, 4))
    s = clustering_silhouette(X, n_clusters=2)
    assert s < 0.5


def test_provider_delta_spikes_at_drift():
    rng = np.random.default_rng(0)
    # 12 batches; batch 6 shifts abruptly (full-feature drift)
    batches = []
    for i in range(12):
        mean = 2.0 if i >= 6 else 0.0
        batches.append(rng.normal(mean, 0.3, size=(50, 6)))
    provider = SignalProvider(batches, reference_period=3)
    # ΔS at the drift boundary should exceed ΔS inside a stable region
    delta_at_drift = provider.delta_s(6)
    stable_deltas = [provider.delta_s(k) for k in range(3, 6)]
    assert delta_at_drift is not None
    assert delta_at_drift > max(stable_deltas)


def test_provider_caches_and_counts_silhouette():
    rng = np.random.default_rng(2)
    batches = [rng.normal(0, 1, size=(50, 4)) for _ in range(6)]
    provider = SignalProvider(batches, reference_period=3)
    s1 = provider.silhouette_pair(0)
    s2 = provider.silhouette_pair(0)  # cached
    assert s1 == s2
    assert provider.n_silhouette_computes == 1


def test_calibrate_thresholds_returns_floats():
    rng = np.random.default_rng(3)
    batches = [rng.normal(0, 1, size=(50, 4)) for _ in range(8)]
    provider = SignalProvider(batches, reference_period=4)
    theta1, theta2 = provider.calibrate_thresholds()
    assert theta1 > 0
    assert theta2 >= 0