"""Synthetic drifted malware family generator.

Used to run the whole pipeline end-to-end without downloading the (large)
KronoDroid dataset. The data is generated from a low-rank latent structure:

* ``n_latent`` latent dimensions are projected to ``n_features`` observed
  features through a shared random orthonormal projection (so the two classes
  live in the same feature space and the geometry is well defined).

* Family X drifts along the *class direction* (toward family Y) at two
  configurable points in time, but never crosses family Y's position. A
  model trained on early batches therefore degrades after the drift, while
  a retrained model recovers.

* A single transient spike scales a few feature columns for one batch only;
  this shifts the marginal distribution (so PSI fires) without changing the
  underlying cluster structure (so the silhouette confirmation stays quiet),
  which exercises the two-signal confirmation logic.
"""

from __future__ import annotations

import numpy as np

from .batches import TemporalBatches


def _projection(rng: np.random.Generator, n_features: int, n_latent: int) -> np.ndarray:
    """Random orthonormal (n_features x n_latent) projection."""
    A, _ = np.linalg.qr(rng.normal(size=(n_features, n_latent)))
    return A


def make_synthetic_pair(
    n_batches: int = 40,
    batch_size: int = 50,
    n_features: int = 16,
    n_latent: int = 3,
    seed: int = 42,
    drift_at: tuple[int, int] | None = (15, 25),
    drift_steps: tuple[float, float] = (2.0, 1.0),
    spike_at: int | None = 8,
    spike_features: int = 4,
    spike_scale: float = 4.0,
    other_shift: float = 4.0,
    latent_scale: float = 0.35,
    noise: float = 0.2,
) -> tuple[TemporalBatches, np.ndarray, np.ndarray]:
    """Generate a synthetic (family X, family Y) experiment pair.

    Returns (batches, other_train, other_test): ``batches`` is the drifting
    family split into chronological batches; ``other_train`` / ``other_test``
    are the fixed 25+25 samples of the "other" class.

    X starts at the origin and drifts toward Y (located at ``other_shift``
    along the first latent axis) by ``drift_steps[0]`` at ``drift_at[0]`` and
    by ``drift_steps[1]`` more at ``drift_at[1]``. The total drift stays
    below ``other_shift`` so the classes remain separable for a retrained
    model, while a static model (trained on the origin) degrades once X
    crosses the original decision boundary.
    """
    rng = np.random.default_rng(seed)
    A = _projection(rng, n_features, n_latent)
    e1 = np.zeros(n_latent)
    e1[0] = 1.0  # class direction: X drifts along it toward Y

    spike_cols: np.ndarray | None = None
    if spike_at is not None:
        spike_cols = rng.choice(n_features, size=spike_features, replace=False)

    batches: list[np.ndarray] = []
    for i in range(n_batches):
        z_mean = np.zeros(n_latent)
        if drift_at is not None:
            if i >= drift_at[0]:
                z_mean = z_mean + e1 * drift_steps[0]
            if i >= drift_at[1]:
                z_mean = z_mean + e1 * drift_steps[1]
        z = rng.normal(loc=z_mean, scale=latent_scale, size=(batch_size, n_latent))
        X = z @ A.T
        if spike_at is not None and i == spike_at and spike_cols is not None:
            X[:, spike_cols] *= spike_scale
        X = X + rng.normal(0.0, noise, size=X.shape)
        batches.append(X)

    z_other = rng.normal(
        loc=e1 * other_shift, scale=latent_scale, size=(50, n_latent)
    )
    Y = z_other @ A.T + rng.normal(0.0, noise, size=(50, n_features))
    return TemporalBatches(batches), Y[:25], Y[25:]