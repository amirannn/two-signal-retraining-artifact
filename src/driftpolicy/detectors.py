"""Standard sequential change detectors used as baseline retraining triggers.

The two detectors implemented here are the classical streaming baselines that
a drift-aware maintenance policy is expected to be compared against:

* :class:`ADWIN` -- Bifet & Gavalda (2007).  An adaptive window is maintained
  as an exponential histogram of buckets; whenever the older and the newer
  part of the window differ in mean by more than the Hoeffding cut, the older
  part is dropped and a change is reported.
* :class:`KSWIN` -- Raab et al. (2020).  A sliding window is split into a
  recent and a reference part; a two-sample Kolmogorov-Smirnov statistic with
  a permutation test decides whether the two parts come from the same
  distribution.

Both detectors are univariate and label-free: they consume one scalar per
sample.  In this codebase the scalar stream is the per-sample novelty score of
:func:`novelty_scores` (the mean squared z-score of a sample against the
reference period), so the detectors see exactly the same information the
PSI/Silhouette pipeline sees and can drive the same retraining protocol
without ground-truth labels.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp


def novelty_scores(reference: np.ndarray, batch: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Label-free per-sample novelty score against the reference period.

    ``score(x) = mean_d ((x_d - mu_d) / (sigma_d + eps))**2``, i.e. the mean
    squared standardized deviation from the reference location/scale.  The
    score is invariant to the feature units and is one number per sample.
    """
    reference = np.asarray(reference, dtype=float)
    batch = np.asarray(batch, dtype=float)
    if reference.ndim != 2 or batch.ndim != 2:
        raise ValueError("novelty_scores expects 2-D arrays (samples x features)")
    if reference.shape[1] != batch.shape[1]:
        raise ValueError("reference and batch must have the same feature count")
    mu = reference.mean(axis=0)
    sigma = reference.std(axis=0) + eps
    z = (batch - mu) / sigma
    return np.mean(z * z, axis=1)


@dataclass
class _Bucket:
    """Aggregate of a run of samples that share the same size class."""

    count: int
    total: float
    total_sq: float

    def merge(self, other: "_Bucket") -> "_Bucket":
        return _Bucket(
            count=self.count + other.count,
            total=self.total + other.total,
            total_sq=self.total_sq + other.total_sq,
        )


class ADWIN:
    """ADWIN adaptive-window drift detector (Bifet & Gavalda, 2007).

    The window is stored as a list of buckets ordered from oldest to newest.
    At most ``max_buckets`` buckets of the same size are kept; once a level
    overflows, its two oldest buckets are merged into a bucket of double the
    size (the exponential-histogram compression of the original algorithm).
    A change is reported when some cut of the window into an older and a newer
    part has sub-window means that differ by more than the Hoeffding bound at
    confidence ``delta``; the older part is then discarded.
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_buckets: int = 5,
        min_window: int = 5,
        clock: int = 32,
    ) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window = min_window
        self.clock = clock
        self._buckets: list[_Bucket] = []
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self._tick = 0
        self.n_cuts = 0

    # -- window statistics -------------------------------------------------
    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        mean = self.total / self.n
        return max(self.total_sq / self.n - mean * mean, 0.0)

    # -- streaming interface ----------------------------------------------
    def update(self, value: float) -> bool:
        """Insert one observation; return True when a change is detected."""
        value = float(value)
        self._buckets.append(_Bucket(1, value, value * value))
        self.n += 1
        self.total += value
        self.total_sq += value * value
        self._tick += 1
        if self._tick % self.clock == 0:
            self._compress()
        return self._detect()

    def update_many(self, values) -> bool:
        """Insert a batch of observations; True if any of them triggered."""
        fired = False
        for value in values:
            if self.update(value):
                fired = True
        return fired

    # -- internal ----------------------------------------------------------
    def _compress(self) -> None:
        """Keep at most ``max_buckets`` buckets per size class."""
        level = 1
        while level <= self.n:
            idx = [i for i, b in enumerate(self._buckets) if b.count == level]
            if len(idx) > self.max_buckets:
                i1, i2 = idx[0], idx[1]
                merged = self._buckets[i1].merge(self._buckets[i2])
                del self._buckets[i2]
                del self._buckets[i1]
                self._buckets.insert(i1, merged)
                continue
            level *= 2

    def _drop_older(self, keep_from: int) -> None:
        """Discard buckets ``[0, keep_from)`` and fix the running sums."""
        for bucket in self._buckets[:keep_from]:
            self.n -= bucket.count
            self.total -= bucket.total
            self.total_sq -= bucket.total_sq
        del self._buckets[:keep_from]
        self.n_cuts += 1

    def _detect(self) -> bool:
        n = self.n
        if n < self.min_window or len(self._buckets) < 2:
            return False
        variance = max(self.variance(), 1e-12)
        # Hoeffding/Bernstein cut of the original algorithm: the confidence is
        # split over the possible cuts (delta / ln n) and the bound is
        # normalized by the harmonic size m = 1 / (1/n_old + 1/n_new),
        #   eps_cut = sqrt(2 * var / m * ln(2/delta')) + (2/3m) * ln(2/delta').
        delta_prime = self.delta / math.log(n)
        log_term = math.log(2.0 / delta_prime)
        total = self.total
        n_old = 0
        sum_old = 0.0
        # Test every cut that puts the boundary between two buckets; the newest
        # bucket is never used as the old part (it is still being filled).
        for i, bucket in enumerate(self._buckets[:-1]):
            n_old += bucket.count
            sum_old += bucket.total
            n_new = n - n_old
            if n_old < 1 or n_new < 1:
                continue
            m = 1.0 / (1.0 / n_old + 1.0 / n_new)
            epsilon = math.sqrt(2.0 * variance / m * log_term) + (
                2.0 / (3.0 * m)
            ) * log_term
            mean_old = sum_old / n_old
            mean_new = (total - sum_old) / n_new
            if abs(mean_old - mean_new) > epsilon:
                self._drop_older(i + 1)
                return True
        return False


class KSWIN:
    """KSWIN sliding-window Kolmogorov-Smirnov detector (Raab et al., 2020).

    The detector keeps the most recent ``window_size`` scores.  After each
    batch it compares the most recent ``stat_size`` scores with the
    ``stat_size`` immediately preceding ones using the two-sample KS statistic
    and a permutation test; a change is reported when the permutation p-value
    drops below ``alpha``.  The test is evaluated once per batch (not once per
    sample) because that is the resolution at which the retraining protocol
    can act.
    """

    def __init__(
        self,
        window_size: int = 100,
        stat_size: int = 50,
        alpha: float = 0.01,
        n_permutations: int = 100,
        seed: int = 42,
    ) -> None:
        if stat_size * 2 > window_size:
            raise ValueError("window_size must be at least twice stat_size")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.seed = seed
        self._window: deque[float] = deque(maxlen=window_size)
        self._rng = np.random.default_rng(seed)
        self.last_p_value: float | None = None
        self.last_statistic: float | None = None
        self.n_alerts = 0

    def update_many(self, values) -> bool:
        """Feed a batch of scores; return True when the window test fires."""
        for value in np.asarray(values, dtype=float).ravel():
            self._window.append(float(value))
        if len(self._window) < 2 * self.stat_size:
            return False
        window = np.fromiter(self._window, dtype=float, count=len(self._window))
        recent = window[-self.stat_size :]
        reference = window[-2 * self.stat_size : -self.stat_size]
        statistic = float(ks_2samp(recent, reference).statistic)
        p_value = self._permutation_p_value(recent, reference, statistic)
        self.last_statistic = statistic
        self.last_p_value = p_value
        fired = p_value < self.alpha
        if fired:
            self.n_alerts += 1
        return fired

    def _permutation_p_value(
        self, recent: np.ndarray, reference: np.ndarray, statistic: float
    ) -> float:
        pooled = np.concatenate([recent, reference])
        size = len(recent)
        count = 0
        for _ in range(self.n_permutations):
            self._rng.shuffle(pooled)
            perm = float(ks_2samp(pooled[:size], pooled[size:]).statistic)
            if perm >= statistic:
                count += 1
        return (1.0 + count) / (self.n_permutations + 1.0)


DETECTOR_FACTORIES = {
    "adwin": lambda seed: ADWIN(),
    "kswin": lambda seed: KSWIN(seed=seed),
}
