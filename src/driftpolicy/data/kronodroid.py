"""KronoDroid loading and paper-style temporal experiment preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .batches import TemporalBatches

_DATE_ALIASES = {
    "date", "timestamp", "first_seen", "first_seen_date", "last_modified",
    "lastmoddate", "lastmodifieddate", "highestmoddate", "highest_mod_date",
    "firstmoddate", "earliestmoddate", "earliest_mod_date",
}
_FAMILY_ALIASES = {"family", "family_id", "family_name", "malfamily", "mal_family"}
_LABEL_ALIASES = {"label", "class", "malware", "is_malware"}
_META_COLUMNS = {
    "sha1", "sha256", "md5", "package", "app_name", "version", "category",
    "dataset", "date", "timestamp", "first_seen", "first_seen_date",
    "last_modified", "lastmoddate", "lastmodifieddate", "highestmoddate",
    "highest_mod_date", "firstmoddate", "earliestmoddate", "earliest_mod_date",
    "label", "class", "malware", "is_malware", "family", "family_id",
    "family_name", "malfamily", "mal_family",
}


@dataclass
class KronoDroidData:
    """Parsed KronoDroid table ready for batching."""

    df: pd.DataFrame
    date_col: str
    family_col: str
    label_col: str
    feature_cols: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


def load_kronodroid_csv(path: str) -> KronoDroidData:
    """Load KronoDroid, using its last-modified date as the temporal key."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, engine="python")
    lowered = {c: c.lower().strip().replace(" ", "_") for c in df.columns}

    def find_column(aliases: set[str]) -> str | None:
        for col, low in lowered.items():
            if low in aliases:
                return col
        return None

    date_col = find_column(_DATE_ALIASES)
    family_col = find_column(_FAMILY_ALIASES)
    label_col = find_column(_LABEL_ALIASES)
    if date_col is None:
        raise ValueError("could not detect a date/timestamp column in the KronoDroid CSV")

    feature_cols = [
        c for c in df.columns
        if lowered[c] not in _META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("no numeric feature columns found in the KronoDroid CSV")

    return KronoDroidData(
        df=df,
        date_col=date_col,
        family_col=family_col or "",
        label_col=label_col or "",
        feature_cols=feature_cols,
    )


def prepare_kronodroid_experiment(
    data: KronoDroidData,
    family: str,
    other_family: str,
    batch_size: int = 50,
    seed: int = 42,
    reference_period: int = 5,
) -> tuple[TemporalBatches, np.ndarray, np.ndarray]:
    """Build the paper-style family-X versus fixed-family-Y experiment.

    The scaler is fitted only on the initial reference prefix of family X and
    the fixed training half of family Y. The same transform is then applied to
    all future batches, preventing future-information leakage while ensuring
    that both the classifier and the clustering signal use normalized features.
    """
    if not data.family_col:
        raise ValueError("the dataset has no family column; cannot select families")

    df = data.df.copy()
    parsed_dates = pd.to_datetime(df[data.date_col], errors="coerce")
    df = df.loc[parsed_dates.notna()].copy()
    df["__parsed_date"] = parsed_dates.loc[df.index]
    family_values = df[data.family_col].astype(str).str.strip().str.lower()
    fam_x = df[family_values == family.lower()]
    fam_y = df[family_values == other_family.lower()]
    if fam_x.empty:
        raise ValueError(f"family '{family}' not found in the dataset")
    if fam_y.empty:
        raise ValueError(f"family '{other_family}' not found in the dataset")

    fam_x = fam_x.sort_values("__parsed_date")
    X = np.nan_to_num(fam_x[data.feature_cols].to_numpy(dtype=float), nan=0.0)

    rng = np.random.default_rng(seed)
    Y = np.nan_to_num(fam_y[data.feature_cols].to_numpy(dtype=float), nan=0.0)
    if len(Y) < 50:
        raise ValueError(f"family '{other_family}' has fewer than 50 samples")
    idx = rng.choice(len(Y), size=50, replace=False)
    Y = Y[idx]

    reference_count = min(len(X), max(reference_period, 1) * batch_size)
    scaler = StandardScaler()
    scaler.fit(np.vstack([X[:reference_count], Y[:25]]))
    X = scaler.transform(X)
    Y = scaler.transform(Y)

    batches = TemporalBatches.from_timeseries(
        X,
        fam_x["__parsed_date"].astype("int64").to_numpy(),
        batch_size=batch_size,
    )
    return batches, Y[:25], Y[25:]
