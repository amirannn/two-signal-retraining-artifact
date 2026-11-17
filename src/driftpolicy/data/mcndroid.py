"""McNdroid dataset loader and temporal experiment preparation.

McNdroid is a multi-year longitudinal Android malware benchmark (2013-2016)
stored as one sparse ``.npz`` per year: CSR components in ``{year}_test_X.npz``
and ``(y, hash)`` in ``{year}_test_meta.npz``. It carries 2,390 multimodal
binary features and a binary benign/malware label (no malware-family column).

The experiment follows the same controlled design as KronoDroid and LAMDA:
malware samples form the chronological target stream, a fixed set of benign
samples supplies the negative class, and the StandardScaler is fitted only on
the reference prefix plus the fixed training half of the negative set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

from .batches import TemporalBatches

YEARS = ("2013", "2014", "2016")
NEGATIVE_POOL = 50


@dataclass
class McNdroidData:
    """Dense malware stream plus a benign pool ready for batching."""

    X: np.ndarray          # malware samples, chronological (n_mal, d)
    y: np.ndarray          # malware timestamps as year integers
    benign: np.ndarray     # benign pool (n_ben, d)
    n_features: int


def load_mcndroid(path: str | Path) -> McNdroidData:
    """Load every available year and stack them into one chronological stream."""
    base = Path(path)
    malware, benign, years = [], [], []
    for year in YEARS:
        x_file = base / f"{year}_test_X.npz"
        m_file = base / f"{year}_test_meta.npz"
        if not x_file.exists() or not m_file.exists():
            continue
        x_npz = np.load(x_file, allow_pickle=True)
        meta = np.load(m_file, allow_pickle=True)
        shape = tuple(int(v) for v in x_npz["shape"])
        matrix = csr_matrix(
            (x_npz["data"], x_npz["indices"], x_npz["indptr"]), shape=shape
        ).toarray().astype(np.float32)
        labels = np.asarray(meta["y"]).astype(int)
        is_malware = labels == 1
        malware.append(matrix[is_malware])
        benign.append(matrix[~is_malware])
        years.append(np.full(int(is_malware.sum()), int(year), dtype=int))

    if not malware:
        raise FileNotFoundError(f"no McNdroid .npz files found in {path}")

    X = np.vstack(malware)
    y = np.concatenate(years)
    benign_all = np.vstack(benign)
    n_features = X.shape[1]
    if benign_all.shape[1] != n_features:
        raise ValueError("benign and malware feature counts differ")
    return McNdroidData(X=X, y=y, benign=benign_all, n_features=n_features)


def prepare_mcndroid_experiment(
    data: McNdroidData,
    batch_size: int = 50,
    seed: int = 42,
    reference_period: int = 5,
    max_batches: int | None = None,
) -> tuple[TemporalBatches, np.ndarray, np.ndarray]:
    """Build the temporal experiment (malware stream vs fixed benign set)."""
    order = np.argsort(data.y, kind="stable")
    X = data.X[order]
    ts = data.y[order]

    rng = np.random.default_rng(seed)
    pool = data.benign
    if len(pool) < NEGATIVE_POOL:
        raise ValueError("not enough benign samples for the fixed negative set")
    idx = rng.choice(len(pool), size=NEGATIVE_POOL, replace=False)
    Y = pool[idx]

    if max_batches is not None:
        keep = min(len(X), max_batches * batch_size)
        X = X[:keep]
        ts = ts[:keep]

    reference_count = min(len(X), max(reference_period, 1) * batch_size)
    scaler = StandardScaler()
    scaler.fit(np.vstack([X[:reference_count], Y[: NEGATIVE_POOL // 2]]))
    X = scaler.transform(X)
    Y = scaler.transform(Y)

    batches = TemporalBatches.from_timeseries(X, ts, batch_size=batch_size)
    return batches, Y[: NEGATIVE_POOL // 2], Y[NEGATIVE_POOL // 2 :]
