"""LAMDA dataset loader and temporal experiment preparation.

LAMDA (Haque et al., 2025) is a longitudinal Android malware benchmark
spanning 2013-2025 with 925 features (var_thresh_0.01) and monthly timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .batches import TemporalBatches

_META_COLS = {"hash", "label", "family", "vt_count", "year_month"}


@dataclass
class LamdaData:
    """Parsed LAMDA table ready for temporal batching."""

    df: pd.DataFrame
    date_col: str
    family_col: str
    label_col: str
    feature_cols: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


def load_lamda_dataset(path: str | Path) -> LamdaData:
    """Load LAMDA dataset from a parquet file or folder of parquet files."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in directory {path}")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_parquet(p)

    date_col = "year_month" if "year_month" in df.columns else None
    family_col = "family" if "family" in df.columns else None
    label_col = "label" if "label" in df.columns else None

    if date_col is None:
        raise ValueError("Could not find 'year_month' column in LAMDA dataset")
    if family_col is None:
        raise ValueError("Could not find 'family' column in LAMDA dataset")
    if label_col is None:
        raise ValueError("Could not find 'label' column in LAMDA dataset")

    feature_cols = [c for c in df.columns if c not in _META_COLS]
    if not feature_cols:
        raise ValueError("Could not find feature columns in LAMDA dataset")

    return LamdaData(
        df=df,
        date_col=date_col,
        family_col=family_col,
        label_col=label_col,
        feature_cols=feature_cols,
    )


def prepare_lamda_experiment(
    data: LamdaData,
    family: str,
    other_family: str = "benign",
    batch_size: int = 50,
    seed: int = 42,
    reference_period: int = 5,
) -> tuple[TemporalBatches, np.ndarray, np.ndarray]:
    """Build the temporal experiment for family X versus fixed other family Y.

    Following the proposal protocol:
    StandardScaler is fitted only on the initial reference prefix of family X
    and the fixed training half of other_family Y. Reused for all future batches.
    """
    df = data.df.copy()
    family_vals = df[data.family_col].astype(str).str.strip().str.lower()
    fam_x = df[family_vals == family.lower()].copy()
    fam_y = df[family_vals == other_family.lower()].copy()

    if fam_x.empty:
        raise ValueError(f"Family '{family}' not found in LAMDA dataset")
    if fam_y.empty:
        raise ValueError(f"Other family '{other_family}' not found in LAMDA dataset")

    fam_x = fam_x.sort_values(data.date_col)
    X = np.nan_to_num(fam_x[data.feature_cols].to_numpy(dtype=float), nan=0.0)

    rng = np.random.default_rng(seed)
    Y = np.nan_to_num(fam_y[data.feature_cols].to_numpy(dtype=float), nan=0.0)
    if len(Y) < 50:
        raise ValueError(f"Other family '{other_family}' has fewer than 50 samples")
    idx = rng.choice(len(Y), size=50, replace=False)
    Y = Y[idx]

    reference_count = min(len(X), max(reference_period, 1) * batch_size)
    scaler = StandardScaler()
    scaler.fit(np.vstack([X[:reference_count], Y[:25]]))
    X = scaler.transform(X)
    Y = scaler.transform(Y)

    # Use month as integer timestamp for sorting
    dates = pd.to_datetime(fam_x[data.date_col], format="%Y-%m", errors="coerce")
    ts = dates.astype("int64").to_numpy()

    batches = TemporalBatches.from_timeseries(
        X,
        ts,
        batch_size=batch_size,
    )
    return batches, Y[:25], Y[25:]
