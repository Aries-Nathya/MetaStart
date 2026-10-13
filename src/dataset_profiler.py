"""Dataset meta-feature extraction shared by offline and online workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


META_FEATURES = [
    "n_rows", "n_features", "n_numeric", "n_categorical", "missing_ratio",
    "n_classes", "imbalance_ratio", "mean_feature_mean", "mean_feature_std",
]


def profile_dataset(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Return a stable numeric profile for a classification dataset."""
    X = X.copy()
    y = pd.Series(y)
    numeric = X.select_dtypes(include=np.number)
    categorical = X.select_dtypes(exclude=np.number)
    counts = y.value_counts(normalize=True, dropna=True)
    imbalance = float(counts.max() / counts.min()) if len(counts) > 1 and counts.min() > 0 else 1.0
    means = numeric.mean()
    stds = numeric.std().replace([np.inf, -np.inf], np.nan)
    return {
        "n_rows": float(len(X)),
        "n_features": float(X.shape[1]),
        "n_numeric": float(numeric.shape[1]),
        "n_categorical": float(categorical.shape[1]),
        "missing_ratio": float(X.isna().to_numpy().mean()) if X.size else 0.0,
        "n_classes": float(y.nunique(dropna=True)),
        "imbalance_ratio": imbalance,
        "mean_feature_mean": float(means.mean()) if len(means) else 0.0,
        "mean_feature_std": float(stds.mean()) if len(stds.dropna()) else 0.0,
    }


def drop_serialized_index_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop only recognizable CSV row-index columns containing a 0..N or 1..N sequence."""
    result = frame.copy()
    removed = []
    expected = np.arange(len(result), dtype=float)
    expected_one_based = expected + 1
    for column in list(result.columns):
        name = str(column).strip().lower()
        looks_like_index = name.startswith("unnamed:") or name in {"index", "level_0"}
        if not looks_like_index:
            continue
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if np.array_equal(values, expected) or np.array_equal(values, expected_one_based):
            result = result.drop(columns=[column])
            removed.append(str(column))
    return result, removed
