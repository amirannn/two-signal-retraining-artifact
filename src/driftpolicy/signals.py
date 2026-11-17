"""Drift signals used by the retraining policies.

Two complementary signals are implemented:

* **PSI** (Population Stability Index) -- a cheap, label-free, distribution-
  shift signal. Computed per feature between the current time window and the
  reference period, then aggregated (mean or max) across features.

* **Silhouette** -- an expensive structural signal. MiniBatch K-Means is
  applied to two consecutive batches and the mean silhouette coefficient
  measures how well the pair clusters. The structural signal ΔS is the
  absolute change between consecutive pair silhouettes (as in the reference
  study), or the change relative to the reference period.

The :class:`SignalProvider` computes both lazily and caches results, and it
counts how many expensive silhouette clusterings were actually computed. This
lets the proposed two-signal policy demonstrate its computational savings:
the silhouette stage is only executed when the cheap PSI gate fires.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.svm import OneClassSVM


# --------------------------------------------------------------------------
# PSI
# --------------------------------------------------------------------------

def psi_per_feature(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-6
) -> np.ndarray:
    """PSI per feature column between two 2-D arrays (same feature count).

    Bin edges are derived from the reference distribution (quantiles), and the
    PSI formula PSI = sum((A - E) * ln(A / E)) is applied per feature.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.ndim != 2 or current.ndim != 2:
        raise ValueError("PSI expects 2-D arrays (samples x features)")
    if reference.shape[1] != current.shape[1]:
        raise ValueError("reference and current must have the same feature count")
    n_features = reference.shape[1]
    values = np.empty(n_features)
    for j in range(n_features):
        ref = reference[:, j]
        cur = current[:, j]
        edges = np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
        ref_pct = np.clip(ref_counts / ref_counts.sum(), eps, 1.0)
        cur_pct = np.clip(cur_counts / cur_counts.sum(), eps, 1.0)
        values[j] = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return values


def psi_signal(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, agg: str = "mean"
) -> float:
    """Aggregate PSI of a window relative to a reference distribution."""
    per_feature = psi_per_feature(reference, current, n_bins=n_bins)
    if agg == "mean":
        return float(per_feature.mean())
    if agg == "max":
        return float(per_feature.max())
    raise ValueError(f"unknown PSI aggregation: {agg!r} (use 'mean' or 'max')")


def incremental_ks_signal(reference: np.ndarray, current: np.ndarray, agg: str = "max") -> float:
    """Feature-wise two-sample KS statistic, aggregated across features."""
    values = np.asarray([
        ks_2samp(reference[:, j], current[:, j], method="auto").statistic
        for j in range(reference.shape[1])
    ])
    if agg == "max":
        return float(values.max())
    if agg == "mean":
        return float(values.mean())
    raise ValueError(f"unknown KS aggregation: {agg!r}")


def mmd_signal(reference: np.ndarray, current: np.ndarray) -> float:
    """Gaussian-kernel MMD statistic for two batches."""
    X = np.vstack([reference, current])
    distances = pairwise_distances(X, metric="euclidean", squared=True)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / max(bandwidth, 1e-12)
    kernel = np.exp(-gamma * distances)
    n, m = len(reference), len(current)
    k_xx = kernel[:n, :n]
    k_yy = kernel[n:, n:]
    k_xy = kernel[:n, n:]
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def ocsvm_signal(reference: np.ndarray, current: np.ndarray, nu: float = 0.05) -> float:
    """Change in outlier rate from a reference-trained One-Class SVM."""
    detector = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    detector.fit(reference)
    ref_outlier_rate = float(np.mean(detector.predict(reference) < 0))
    cur_outlier_rate = float(np.mean(detector.predict(current) < 0))
    return cur_outlier_rate - ref_outlier_rate


# --------------------------------------------------------------------------
# Silhouette
# --------------------------------------------------------------------------

def clustering_silhouette(
    X: np.ndarray, n_clusters: int = 2, minibatch_size: int = 25, seed: int = 42
) -> float:
    """Mean silhouette coefficient of MiniBatch K-Means on ``X``."""
    X = np.asarray(X, dtype=float)
    if len(X) < n_clusters + 1:
        return 0.0
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=minibatch_size,
        random_state=seed,
        n_init=3,
        max_iter=100,
    )
    labels = kmeans.fit_predict(X)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score(X, labels))


# --------------------------------------------------------------------------
# Signal provider
# --------------------------------------------------------------------------

def _threshold_from_values(
    values: list[float],
    method: str = "percentile",
    percentile: float = 0.90,
    robust_k: float = 3.0,
    default: float = 0.1,
) -> float:
    """Derive a data-driven threshold from observed signal values."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return default
    if method == "percentile":
        return float(np.quantile(clean, percentile))
    if method == "robust":
        med = float(np.median(clean))
        mad = float(np.median(np.abs(np.asarray(clean) - med)))
        scale = 1.4826 * mad if mad > 0 else 1.0
        return med + robust_k * scale
    raise ValueError(f"unknown threshold method: {method!r}")


class SignalProvider:
    """Lazily computes and caches the two drift signals per window.

    Silhouette pairs are only computed when requested, so policies that gate
    on PSI first (the proposed two-signal policy) can measure the savings in
    expensive clustering computations via :attr:`n_silhouette_computes`.
    """

    def __init__(
        self,
        batches: list[np.ndarray],
        reference_period: int = 5,
        psi_bins: int = 10,
        psi_agg: str = "mean",
        n_clusters: int = 2,
        minibatch_size: int = 25,
        seed: int = 42,
    ) -> None:
        if reference_period < 2:
            raise ValueError("reference_period must be >= 2")
        self.batches = batches
        self.n = len(batches)
        self.reference_period = min(reference_period, self.n)
        self.psi_bins = psi_bins
        self.psi_agg = psi_agg
        self.n_clusters = n_clusters
        self.minibatch_size = minibatch_size
        self.seed = seed
        self._psi_cache: dict[int, float] = {}
        self._sil_cache: dict[int, float] = {}
        self.n_silhouette_computes = 0
        self.silhouette_sample_count = 0
        self._detector_cache: dict[tuple[str, int], float] = {}

    # ---- PSI -------------------------------------------------------------

    def _reference_aggregate(self, exclude: int | None = None) -> np.ndarray:
        idx = [i for i in range(self.reference_period) if i != exclude]
        return np.vstack([self.batches[i] for i in idx])

    def psi(self, k: int) -> float:
        """PSI of batch ``k`` relative to the reference aggregate."""
        if k not in self._psi_cache:
            self._psi_cache[k] = psi_signal(
                self._reference_aggregate(),
                self.batches[k],
                n_bins=self.psi_bins,
                agg=self.psi_agg,
            )
        return self._psi_cache[k]

    def reference_psi_values(self) -> list[float]:
        """Leave-one-out PSI inside the reference period (natural variation)."""
        return [
            psi_signal(
                self._reference_aggregate(exclude=k),
                self.batches[k],
                n_bins=self.psi_bins,
                agg=self.psi_agg,
            )
            for k in range(self.reference_period)
        ]

    def detector_value(self, kind: str, k: int) -> float | None:
        """Compute and cache a label-free detector value for batch ``k``."""
        if k < 1:
            return None
        key = (kind, k)
        if key not in self._detector_cache:
            reference = self.batches[k - 1]
            current = self.batches[k]
            if kind == "psi":
                value = psi_signal(self._reference_aggregate(), current, self.psi_bins, self.psi_agg)
            elif kind == "ks":
                value = incremental_ks_signal(reference, current)
            elif kind == "mmd":
                value = mmd_signal(reference, current)
            elif kind == "ocsvm":
                value = ocsvm_signal(self._reference_aggregate(), current)
            else:
                raise ValueError(f"unknown detector kind: {kind!r}")
            self._detector_cache[key] = float(value)
        return self._detector_cache[key]

    def reference_detector_values(self, kind: str) -> list[float]:
        """Detector values on adjacent reference batches for calibration."""
        return [
            self.detector_value(kind, k)
            for k in range(1, self.reference_period)
        ]

    # ---- Silhouette ------------------------------------------------------

    def silhouette_pair(self, i: int) -> float:
        """Mean silhouette for the pair {batch i, batch i+1}."""
        if i not in self._sil_cache:
            X = np.vstack([self.batches[i], self.batches[i + 1]])
            self._sil_cache[i] = clustering_silhouette(
                X,
                n_clusters=self.n_clusters,
                minibatch_size=self.minibatch_size,
                seed=self.seed,
            )
            self.n_silhouette_computes += 1
            self.silhouette_sample_count += len(X)
        return self._sil_cache[i]

    def delta_s(self, k: int) -> float | None:
        """ΔS at batch k: |s(k-1) - s(k-2)| (change vs the previous pair)."""
        if k < 2:
            return None
        return abs(self.silhouette_pair(k - 1) - self.silhouette_pair(k - 2))

    def delta_s_reference(self, k: int) -> float | None:
        """ΔS at batch k: |s(k-1) - mean(reference pairs)|."""
        if k < 1:
            return None
        s_ref = float(
            np.mean([self.silhouette_pair(i) for i in range(self.reference_period - 1)])
        )
        return abs(self.silhouette_pair(k - 1) - s_ref)

    def reference_delta_values(self) -> list[float]:
        """ΔS values inside the reference period (natural variation)."""
        return [self.delta_s(k) for k in range(2, self.reference_period)]

    # ---- Empirical calibration -------------------------------------------

    @staticmethod
    def _upper_tail_p(value: float, reference_values: list[float]) -> float:
        """Finite-sample empirical upper-tail p-value with a +1 correction."""
        clean = [float(v) for v in reference_values if v is not None and np.isfinite(v)]
        if not clean:
            return 1.0
        return (1.0 + sum(v >= value for v in clean)) / (len(clean) + 1.0)

    def psi_p_value(self, k: int) -> float:
        """Empirical upper-tail p-value for PSI using reference-only variation."""
        return self._upper_tail_p(self.psi(k), self.reference_psi_values())

    def delta_s_p_value(self, k: int) -> float | None:
        """Empirical upper-tail p-value for structural change at batch ``k``."""
        value = self.delta_s(k)
        if value is None:
            return None
        return self._upper_tail_p(value, self.reference_delta_values())

    # ---- Threshold calibration -------------------------------------------

    def calibrate_detector_threshold(
        self,
        kind: str,
        method: str = "percentile",
        percentile: float = 0.90,
        robust_k: float = 3.0,
        default: float = 0.1,
    ) -> float:
        """Calibrate one detector threshold from reference-only values."""
        return _threshold_from_values(
            self.reference_detector_values(kind), method, percentile, robust_k, default
        )

    def calibrate_thresholds(
        self,
        method: str = "percentile",
        percentile: float = 0.90,
        robust_k: float = 3.0,
        default_psi: float = 0.1,
        default_delta: float = 0.05,
    ) -> tuple[float, float]:
        """Calibrate (theta1, theta2) from the reference-period variation."""
        psi_vals = self.reference_psi_values()
        delta_vals = self.reference_delta_values()
        theta1 = _threshold_from_values(
            psi_vals, method, percentile, robust_k, default_psi
        )
        theta2 = _threshold_from_values(
            delta_vals, method, percentile, robust_k, default_delta
        )
        return theta1, theta2