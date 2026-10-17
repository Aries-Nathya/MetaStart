"""Dataset meta-features shared by benchmarks and upload-time recommendations.

``META_FEATURES`` remains the legacy feature set consumed by the frozen model.
``ADDITIONAL_META_FEATURES`` are experimental profile outputs; they are not
automatically included in model training or prediction.

Missing numeric values are skipped by pandas moments and pairwise correlations;
non-finite numeric values are treated as missing for those calculations.
Categorical entropy/cardinality ignore missing values. Constant numeric columns
are ignored when aggregating moments and undefined correlations; constant
categorical columns contribute zero normalized entropy and cardinality one.
Aggregates with no applicable columns/pairs return zero. Correlations are
absolute Pearson correlations, calculated on at most 256 numeric columns chosen
deterministically with seed 1729; the high-correlation fraction uses only finite,
non-constant pairs as its denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


META_FEATURES = [
    "n_rows", "n_features", "n_numeric", "n_categorical", "missing_ratio",
    "n_classes", "imbalance_ratio", "mean_feature_mean", "mean_feature_std",
]
ADDITIONAL_META_FEATURES = [
    "normalized_class_entropy",
    "log10_samples_per_feature",
    "median_abs_numeric_skewness",
    "median_abs_numeric_excess_kurtosis",
    "mean_abs_numeric_correlation",
    "fraction_numeric_pairs_abs_corr_ge_0_8",
    "mean_normalized_categorical_entropy",
    "mean_categorical_cardinality",
]
PROFILE_FEATURES = [*META_FEATURES, *ADDITIONAL_META_FEATURES]
MAX_CORRELATION_FEATURES = 256
CORRELATION_SAMPLE_SEED = 1729


def _normalized_entropy(values: pd.Series) -> float:
    """Shannon entropy in bits, normalized to [0, 1]; zero for <=1 class."""
    counts = values.value_counts(dropna=True).to_numpy(dtype=float)
    if len(counts) <= 1 or counts.sum() <= 0:
        return 0.0
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return float(entropy / np.log2(len(counts)))


def _finite_median_abs(values: pd.Series) -> float:
    values = values.replace([np.inf, -np.inf], np.nan).dropna().abs()
    result = float(values.median()) if len(values) else 0.0
    return result if np.isfinite(result) else 0.0


def _numeric_correlation_features(numeric: pd.DataFrame) -> tuple[float, float]:
    """Summarize valid absolute Pearson pairs after deterministic width sampling."""
    if numeric.shape[1] > MAX_CORRELATION_FEATURES:
        rng = np.random.default_rng(CORRELATION_SAMPLE_SEED)
        selected = np.sort(rng.choice(numeric.shape[1], MAX_CORRELATION_FEATURES,
                                      replace=False))
        numeric = numeric.iloc[:, selected]
    matrix = numeric.corr(method="pearson").abs().to_numpy(dtype=float)
    if matrix.shape[0] < 2:
        return 0.0, 0.0
    pairs = matrix[np.triu_indices(matrix.shape[0], k=1)]
    pairs = pairs[np.isfinite(pairs)]
    if not len(pairs):
        return 0.0, 0.0
    return float(pairs.mean()), float(np.mean(pairs >= 0.8))


def profile_dataset(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Return the legacy profile followed by eight experimental meta-features.

    Numeric columns follow pandas' numeric dtype selection; every other column,
    including booleans and strings, is treated as categorical. Class entropy
    ignores missing target labels. ``log10_samples_per_feature`` uses a minimum
    numerator and denominator of one, so empty shapes still produce finite
    output. Constant/undefined moments and correlations use the zero fallbacks
    documented in the module docstring.
    """
    y = pd.Series(y)
    numeric = X.select_dtypes(include=np.number).copy()
    numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    categorical = X.select_dtypes(exclude=np.number)
    counts = y.value_counts(dropna=True)
    imbalance = float(counts.max() / counts.min()) if len(counts) > 1 and counts.min() > 0 else 1.0
    means = numeric.mean()
    stds = numeric.std().replace([np.inf, -np.inf], np.nan)
    finite_means = means.replace([np.inf, -np.inf], np.nan).dropna()
    profile = {
        "n_rows": float(len(X)),
        "n_features": float(X.shape[1]),
        "n_numeric": float(numeric.shape[1]),
        "n_categorical": float(categorical.shape[1]),
        "missing_ratio": float(X.isna().to_numpy().mean()) if X.size else 0.0,
        "n_classes": float(y.nunique(dropna=True)),
        "imbalance_ratio": imbalance,
        "mean_feature_mean": float(finite_means.mean()) if len(finite_means) else 0.0,
        "mean_feature_std": float(stds.mean()) if len(stds.dropna()) else 0.0,
    }
    class_entropy = _normalized_entropy(y)
    varying_numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    skewness = varying_numeric.skew(axis=0)
    excess_kurtosis = varying_numeric.kurt(axis=0)
    mean_abs_correlation, high_correlation_fraction = _numeric_correlation_features(numeric)

    categorical_entropies = [_normalized_entropy(categorical[column])
                             for column in categorical.columns]
    categorical_cardinalities = [float(categorical[column].nunique(dropna=True))
                                 for column in categorical.columns]
    profile.update({
        "normalized_class_entropy": class_entropy,
        "log10_samples_per_feature": float(np.log10(max(len(X), 1) / max(X.shape[1], 1))),
        "median_abs_numeric_skewness": _finite_median_abs(skewness),
        "median_abs_numeric_excess_kurtosis": _finite_median_abs(excess_kurtosis),
        "mean_abs_numeric_correlation": mean_abs_correlation,
        "fraction_numeric_pairs_abs_corr_ge_0_8": high_correlation_fraction,
        "mean_normalized_categorical_entropy": (
            float(np.mean(categorical_entropies)) if categorical_entropies else 0.0
        ),
        "mean_categorical_cardinality": (
            float(np.mean(categorical_cardinalities)) if categorical_cardinalities else 0.0
        ),
    })
    # Keep field order explicit and stable even if implementation details change.
    return {feature: float(profile[feature]) for feature in PROFILE_FEATURES}


def drop_serialized_index_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop only recognizable CSV row-index columns containing a 0..N or 1..N sequence."""
    result = frame
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
