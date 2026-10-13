import pandas as pd

from src.benchmark import build_pipeline, get_models
from src.dataset_profiler import META_FEATURES, drop_serialized_index_columns, profile_dataset


def test_profile_dataset_returns_stable_numeric_features():
    X = pd.DataFrame({"number": [1.0, None, 3.0], "category": ["a", "b", "a"]})
    profile = profile_dataset(X, pd.Series([0, 0, 1]))
    assert list(profile) == META_FEATURES
    assert profile["n_rows"] == 3
    assert profile["n_numeric"] == 1
    assert profile["n_categorical"] == 1
    assert profile["missing_ratio"] > 0


def test_candidate_pipeline_fits_mixed_features():
    X = pd.DataFrame({"number": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                      "category": ["a", "b", "a", "b", "a", "b"]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    pipeline = build_pipeline(get_models()["Decision Tree"], X)
    pipeline.fit(X, y)
    assert len(pipeline.predict(X)) == len(y)


def test_drop_serialized_index_only_removes_sequential_index():
    frame = pd.DataFrame({"Unnamed: 0": [0, 1, 2], "feature": [7, 8, 9],
                          "index": [10, 20, 30]})
    cleaned, removed = drop_serialized_index_columns(frame)
    assert removed == ["Unnamed: 0"]
    assert list(cleaned.columns) == ["feature", "index"]
