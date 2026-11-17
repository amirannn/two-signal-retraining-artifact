"""Temporal batching utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalBatches:
    """Fixed-size chronological batches."""

    batches: list[np.ndarray]

    def __post_init__(self) -> None:
        if not self.batches:
            raise ValueError("at least one batch is required")
        n_features = self.batches[0].shape[1]
        if any(batch.ndim != 2 or batch.shape[1] != n_features for batch in self.batches):
            raise ValueError("all batches must be 2-D with the same feature count")

    @property
    def n_batches(self) -> int:
        return len(self.batches)

    def batch(self, k: int) -> np.ndarray:
        return self.batches[k]

    def train_half(self, k: int) -> np.ndarray:
        batch = self.batches[k]
        return batch[: len(batch) // 2]

    def test_half(self, k: int) -> np.ndarray:
        batch = self.batches[k]
        return batch[len(batch) // 2 :]

    @classmethod
    def from_timeseries(
        cls, X: np.ndarray, times: np.ndarray, batch_size: int = 50
    ) -> "TemporalBatches":
        X = np.asarray(X)
        times = np.asarray(times)
        if len(X) != len(times):
            raise ValueError("X and times must have equal length")
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        order = np.argsort(times, kind="stable")
        X = X[order]
        n_complete = len(X) // batch_size
        if n_complete < 1:
            raise ValueError(f"not enough samples ({len(X)}) for at least one batch of size {batch_size}")
        return cls([X[i * batch_size : (i + 1) * batch_size] for i in range(n_complete)])
