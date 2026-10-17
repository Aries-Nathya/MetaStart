import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from pathlib import Path

from src.benchmark import (ALGORITHM_NAMES, META_DATASET_COLUMNS, OPENML_CACHE_DIR, ROOT,
                           build_pipeline, get_datasets, get_models,
                           load_openml_datasets, run_benchmarks,
                           validate_classification_dataset)
from src.dataset_profiler import (ADDITIONAL_META_FEATURES, META_FEATURES, PROFILE_FEATURES,
                                 drop_serialized_index_columns, profile_dataset)
from src.dataset_catalog import SYNTHETIC_DATASET_SPECS
from src.provenance import filter_trainable_records, write_dataset_provenance
from src.app_helpers import production_profile_table


def test_profile_dataset_returns_stable_numeric_features():
    X = pd.DataFrame({"number": [1.0, None, 3.0], "category": ["a", "b", "a"]})
    profile = profile_dataset(X, pd.Series([0, 0, 1]))
    assert list(profile) == PROFILE_FEATURES
    assert list(profile)[:len(META_FEATURES)] == META_FEATURES
    assert not set(ADDITIONAL_META_FEATURES) & set(META_FEATURES)
    assert all(np.isfinite(value) for value in profile.values())
    assert profile["n_rows"] == 3
    assert profile["n_numeric"] == 1
    assert profile["n_categorical"] == 1
    assert profile["missing_ratio"] > 0


def test_extended_profile_definitions_are_deterministic_for_mixed_data():
    X = pd.DataFrame({
        "number": [1.0, 2.0, None, 4.0, 5.0],
        "double": [2.0, 4.0, 6.0, 8.0, 10.0],
        "constant": [7.0] * 5,
        "category": ["a", "a", "b", None, "c"],
    })
    y = pd.Series(["yes", "yes", "yes", "no", "no"])
    first = profile_dataset(X, y)
    second = profile_dataset(X, y)
    assert first == second
    assert list(first) == PROFILE_FEATURES
    assert all(np.isfinite(value) for value in first.values())
    assert first["normalized_class_entropy"] == pytest.approx(
        -(0.6 * np.log2(0.6) + 0.4 * np.log2(0.4))
    )
    assert first["log10_samples_per_feature"] == pytest.approx(np.log10(5 / 4))
    assert first["mean_abs_numeric_correlation"] == pytest.approx(1.0)
    assert first["fraction_numeric_pairs_abs_corr_ge_0_8"] == pytest.approx(1.0)
    assert first["mean_normalized_categorical_entropy"] == pytest.approx(
        1.5 / np.log2(3)
    )
    assert first["mean_categorical_cardinality"] == pytest.approx(3.0)


@pytest.mark.parametrize("X,y", [
    (pd.DataFrame({"category": ["a", "b", "a", None]}), pd.Series([0, 0, 1, 1])),
    (pd.DataFrame({"constant": [1.0] * 4}), pd.Series([0, 0, 1, 1])),
    (pd.DataFrame({"one_numeric": [1.0, 2.0, 3.0, 4.0]}), pd.Series([0, 0, 1, 1])),
    (pd.DataFrame(index=range(4)), pd.Series([0, 0, 1, 1])),
    (pd.DataFrame(), pd.Series(dtype="int64")),
])
def test_extended_profile_edge_cases_have_finite_values(X, y):
    profile = profile_dataset(X, y)
    assert list(profile) == PROFILE_FEATURES
    assert all(np.isfinite(value) for value in profile.values())
    if X.shape[1] < 2 or X.select_dtypes(include=np.number).shape[1] < 2:
        assert profile["mean_abs_numeric_correlation"] == 0.0
        assert profile["fraction_numeric_pairs_abs_corr_ge_0_8"] == 0.0


def test_wide_profile_correlation_sampling_is_deterministic():
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(40, 300)))
    y = pd.Series([0, 1] * 20)
    first = profile_dataset(X, y)
    second = profile_dataset(X, y)
    assert first == second
    assert np.isfinite(first["mean_abs_numeric_correlation"])
    assert 0.0 <= first["fraction_numeric_pairs_abs_corr_ge_0_8"] <= 1.0


def test_constant_numeric_columns_do_not_change_skewness_or_kurtosis():
    varying = pd.DataFrame({
        "skewed": [0.0, 1.0, 1.0, 1.0, 2.0, 10.0],
        "second_varying": [3.0, 4.0, 5.0, 7.0, 11.0, 18.0],
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])
    with_constant = varying.assign(constant=4.0, all_missing=np.nan,
                                   non_finite=[np.inf, -np.inf, np.inf,
                                               -np.inf, np.inf, -np.inf])
    original_profile = profile_dataset(varying, y)
    extended_profile = profile_dataset(with_constant, y)
    assert extended_profile["median_abs_numeric_skewness"] == pytest.approx(
        original_profile["median_abs_numeric_skewness"]
    )
    assert extended_profile["median_abs_numeric_excess_kurtosis"] == pytest.approx(
        original_profile["median_abs_numeric_excess_kurtosis"]
    )


def test_all_constant_and_all_missing_numeric_moments_use_zero_fallback():
    X = pd.DataFrame({
        "constant": [5.0] * 6,
        "missing": [np.nan] * 6,
        "infinite": [np.inf, -np.inf] * 3,
    })
    profile = profile_dataset(X, pd.Series([0, 0, 0, 1, 1, 1]))
    assert profile["median_abs_numeric_skewness"] == 0.0
    assert profile["median_abs_numeric_excess_kurtosis"] == 0.0
    assert all(np.isfinite(value) for value in profile.values())


def test_digits_constant_columns_do_not_enter_moment_aggregates():
    X, y = next((X, y) for name, X, y in get_datasets() if name == "digits")
    numeric = X.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
    varying = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    assert numeric.shape[1] - varying.shape[1] == 3

    def median_abs_moment(values):
        values = values.replace([np.inf, -np.inf], np.nan).dropna().abs()
        return float(values.median()) if len(values) else 0.0

    profile = profile_dataset(X, y)
    assert profile["median_abs_numeric_skewness"] == pytest.approx(
        median_abs_moment(varying.skew(axis=0))
    )
    assert profile["median_abs_numeric_excess_kurtosis"] == pytest.approx(
        median_abs_moment(varying.kurt(axis=0))
    )


def test_recommender_keeps_frozen_model_feature_inputs(monkeypatch):
    from src import recommender

    class InspectingModel:
        def predict(self, candidates):
            assert list(candidates.columns) == META_FEATURES + ["algorithm"]
            return np.arange(len(candidates), dtype=float)

    monkeypatch.setattr(recommender.joblib, "load", lambda _path: {
        "algorithms": ["SVM", "Random Forest"], "model": InspectingModel(),
    })
    X = pd.DataFrame({"number": [1.0, 2.0, 3.0, 4.0],
                      "category": ["a", "b", "a", "b"]})
    result = recommender.recommend_algorithms(X, pd.Series([0, 0, 1, 1]),
                                              artifact_path="unused.joblib")
    assert list(result.columns) == META_FEATURES + ["algorithm", "predicted_macro_f1"]
    assert len(result) == 2


def test_production_profile_display_uses_only_nine_labeled_model_features():
    X = pd.DataFrame({
        "number": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "category": ["a", "b", "a", "b", "a", "b"],
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])
    profile = profile_dataset(X, y)

    table = production_profile_table(profile)

    assert len(profile) == 17
    assert table["Characteristic"].tolist() == [
        "Training rows", "Features", "Numeric features", "Categorical features",
        "Missing-value ratio", "Target classes", "Class imbalance ratio",
        "Mean numeric feature mean", "Mean numeric feature standard deviation",
    ]
    assert table["Value"].tolist() == [profile[feature] for feature in META_FEATURES]
    assert len(table) == len(META_FEATURES) == 9


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


def test_validate_classification_dataset_rejects_invalid_targets_and_shapes():
    X = pd.DataFrame({"x": range(6)})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    clean_X, clean_y = validate_classification_dataset(X, y)
    assert clean_X.shape == (6, 1)
    assert clean_y.nunique() == 2
    for bad_X, bad_y in [
        (pd.DataFrame(index=range(6)), y),
        (X, pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
        (X, pd.Series([0, 0, 0, 1, 1, 2])),
    ]:
        try:
            validate_classification_dataset(bad_X, bad_y)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dataset should be rejected")


def test_openml_duplicate_ids_are_reported_without_refetching():
    calls = []

    def fetcher(**kwargs):
        calls.append(kwargs["data_id"])
        return SimpleNamespace(data=pd.DataFrame({"x": range(6)}),
                               target=pd.Series([0, 0, 0, 1, 1, 1]))

    datasets, manifest = load_openml_datasets([(123, "demo"), (123, "demo")], fetcher=fetcher)
    assert len(datasets) == 1
    assert calls == [123]
    assert [row["status"] for row in manifest] == ["loaded", "failed"]


def test_openml_fetch_uses_project_cache_directory():
    observed = {}

    def fetcher(**kwargs):
        cache = Path(kwargs["data_home"]).resolve()
        observed.update(kwargs)
        assert cache.is_relative_to(ROOT.resolve())
        assert cache == OPENML_CACHE_DIR.resolve()
        assert cache.is_dir()
        return SimpleNamespace(data=pd.DataFrame({"x": range(6)}),
                              target=pd.Series([0, 0, 0, 1, 1, 1]))

    load_openml_datasets([(456, "cache-check")], fetcher=fetcher)
    assert observed["data_id"] == 456
    assert observed["as_frame"] is True
    assert observed["data_home"] == str(OPENML_CACHE_DIR)


def test_openml_manifest_captures_reported_license_and_target_labels():
    def fetcher(**kwargs):
        return SimpleNamespace(
            data=pd.DataFrame({"number": [1, None, 3, 4, 5, 6],
                               "category": ["a", "b", None, "a", "b", "a"]}),
            target=pd.Series(["001", "001", "001", "002", "002", "002"]),
            details={"licence": "Public"},
        )

    datasets, manifest = load_openml_datasets([(123, "labels")], fetcher=fetcher)
    assert manifest[0]["status"] == "loaded"
    assert manifest[0]["license_status"] == "Public"
    assert datasets[0][2].tolist() == ["001", "001", "001", "002", "002", "002"]
    pipeline = build_pipeline(get_models()["Decision Tree"], datasets[0][1])
    pipeline.fit(datasets[0][1], datasets[0][2])
    assert len(pipeline.predict(datasets[0][1])) == 6


def test_numeric_class_codes_are_allowed_but_high_cardinality_targets_are_rejected():
    X = pd.DataFrame({"x": range(153)})
    class_codes = pd.Series([1, 2] * 76 + [1])
    assert validate_classification_dataset(X, class_codes)[1].nunique() == 2

    many_classes = pd.Series([label for label in range(51) for _ in range(3)])
    many_features = pd.DataFrame({"x": range(len(many_classes))})
    try:
        validate_classification_dataset(many_features, many_classes)
    except ValueError as exc:
        assert "verify an explicit classification task" in str(exc)
    else:
        raise AssertionError("ambiguous high-cardinality numeric target should be rejected")


def test_new_local_synthetic_datasets_are_deterministic_and_match_profiles():
    first = {name: (X, y) for name, X, y in get_datasets()
             if name in SYNTHETIC_DATASET_SPECS}
    second = {name: (X, y) for name, X, y in get_datasets()
              if name in SYNTHETIC_DATASET_SPECS}
    assert set(first) == set(second) == set(SYNTHETIC_DATASET_SPECS)
    expected = {
        "synthetic_local_300x6_imbalanced": (300, 6, 2, [0.90, 0.10]),
        "synthetic_local_1200x20_four_class": (1200, 20, 4, [0.55, 0.25, 0.15, 0.05]),
        "synthetic_local_2500x60_three_class": (2500, 60, 3, [1 / 3, 1 / 3, 1 / 3]),
    }
    expected_seeds = {
        "synthetic_42": 42, "synthetic_43": 43, "synthetic_44": 44,
        "synthetic_local_300x6_imbalanced": 20261001,
        "synthetic_local_1200x20_four_class": 20261002,
        "synthetic_local_2500x60_three_class": 20261003,
    }
    for name, (X, y) in first.items():
        other_X, other_y = second[name]
        assert X.equals(other_X)
        assert y.equals(other_y)
        assert SYNTHETIC_DATASET_SPECS[name]["seed"] == expected_seeds[name]
        if name not in expected:
            continue
        n_rows, n_features, n_classes, expected_weights = expected[name]
        assert X.shape == (n_rows, n_features)
        assert y.nunique() == n_classes
        actual_weights = y.value_counts(normalize=True).sort_index().to_numpy()
        assert len(actual_weights) == len(expected_weights)
        assert all(abs(actual - desired) < 0.03
                   for actual, desired in zip(actual_weights, expected_weights))


def test_provenance_filter_excludes_unverified_openml_rows(tmp_path):
    rows = []
    for dataset in ["iris", "synthetic_local_300x6_imbalanced", "openml_11_balance_scale"]:
        for algorithm in ALGORITHM_NAMES:
            rows.append({"dataset": dataset, "algorithm": algorithm, "macro_f1": 0.8})
    data = pd.DataFrame(rows)
    local = pd.DataFrame({"x": range(6)})
    labels = pd.Series([0, 0, 0, 1, 1, 1])
    registry = write_dataset_provenance(
        data, [("iris", local, labels),
               ("synthetic_local_300x6_imbalanced", local, labels)], tmp_path,
        run_results=data,
    )
    assert registry.set_index("dataset").loc["openml_11_balance_scale", "provenance_status"] == "unverified"
    filtered = filter_trainable_records(data, tmp_path / "results/dataset_provenance.csv")
    assert set(filtered.dataset) == {"iris", "synthetic_local_300x6_imbalanced"}
    assert "openml_11_balance_scale" not in set(filtered.dataset)


def test_benchmark_records_all_five_algorithms_as_complete(tmp_path, monkeypatch):
    import src.benchmark as benchmark

    X = pd.DataFrame({"x": range(15), "category": ["a", "b", "c"] * 5})
    y = pd.Series([0, 1, 2] * 5)
    monkeypatch.setattr(benchmark, "get_datasets",
                        lambda: iter([("synthetic_local_300x6_imbalanced", X, y)]))
    results = run_benchmarks(tmp_path)
    assert set(results.algorithm) == set(ALGORITHM_NAMES)
    assert len(results) == len(ALGORITHM_NAMES)
    registry = pd.read_csv(tmp_path / "results/dataset_provenance.csv").set_index("dataset")
    assert registry.loc["synthetic_local_300x6_imbalanced", "benchmark_status"] == "complete"


def test_benchmark_default_is_offline_and_preserves_existing_rows(tmp_path, monkeypatch):
    import src.benchmark as benchmark

    X = pd.DataFrame({"x": range(12)})
    y = pd.Series([0] * 6 + [1] * 6)
    monkeypatch.setattr(benchmark, "get_datasets", lambda: iter([("tiny", X, y)]))
    monkeypatch.setattr(benchmark, "load_openml_datasets",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network loader called")))
    old = pd.DataFrame([{
        "dataset": "historical", "n_rows": 10, "n_features": 1, "n_numeric": 1,
        "n_categorical": 0, "missing_ratio": 0, "n_classes": 2, "imbalance_ratio": 1,
        "mean_feature_mean": 0, "mean_feature_std": 1, "algorithm": name,
        "macro_f1": .5, "training_seconds": .1, "wall_seconds": .2,
    } for name in benchmark.ALGORITHM_NAMES], columns=META_DATASET_COLUMNS)
    (tmp_path / "data").mkdir()
    old.to_csv(tmp_path / "data/meta_dataset.csv", index=False)

    output = run_benchmarks(tmp_path)
    assert list(output.columns) == META_DATASET_COLUMNS
    assert set(output.dataset) == {"historical", "tiny"}
    assert not (tmp_path / "results/openml_manifest.csv").exists()


def _complete_records(*identities):
    return pd.DataFrame([
        {"dataset": identity, "algorithm": algorithm, "macro_f1": 0.75}
        for identity in identities for algorithm in ALGORITHM_NAMES
    ])


def _write_registry(tmp_path, data):
    path = tmp_path / "results/dataset_provenance.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_provenance(data, [], tmp_path, run_results=data)
    return path


def test_provenance_filter_rejects_arbitrary_synthetic_identity(tmp_path):
    data = _complete_records("synthetic_fake")
    registry = _write_registry(tmp_path, data)
    with pytest.raises(ValueError, match="not in the approved catalog"):
        filter_trainable_records(data, registry)


def test_provenance_filter_rejects_forged_category_status_pair(tmp_path):
    data = _complete_records("iris")
    registry = _write_registry(tmp_path, data)
    metadata = pd.read_csv(registry)
    metadata.loc[0, "provenance_category"] = "synthetic"
    metadata.to_csv(registry, index=False)
    with pytest.raises(ValueError, match="Contradictory or unapproved provenance"):
        filter_trainable_records(data, registry)


def test_provenance_filter_recomputes_completeness_instead_of_trusting_registry(tmp_path):
    complete = _complete_records("iris")
    registry = _write_registry(tmp_path, complete)
    missing = complete.iloc[:-1].copy()
    with pytest.raises(ValueError, match="Stale benchmark_status"):
        filter_trainable_records(missing, registry)


def test_provenance_filter_rejects_duplicate_algorithm_rows(tmp_path):
    data = _complete_records("iris")
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    registry = _write_registry(tmp_path, data)
    with pytest.raises(ValueError, match="Duplicate algorithm records"):
        filter_trainable_records(data, registry)


def test_provenance_filter_rejects_unexpected_algorithm_name(tmp_path):
    data = _complete_records("iris")
    data.loc[data.index[0], "algorithm"] = "Unregistered Model"
    registry = _write_registry(tmp_path, data)
    with pytest.raises(ValueError, match="Unexpected algorithms"):
        filter_trainable_records(data, registry)


def test_unverified_identity_is_never_in_training_records(tmp_path):
    data = _complete_records("iris", "openml_11_balance_scale")
    registry = _write_registry(tmp_path, data)
    filtered = filter_trainable_records(data, registry)
    assert set(filtered.dataset) == {"iris"}


def test_training_and_evaluation_reject_invalid_provenance(tmp_path, monkeypatch):
    from src import evaluation, meta_learner

    data = _complete_records("iris")
    registry = _write_registry(tmp_path, data)
    metadata = pd.read_csv(registry)
    metadata.loc[0, "provenance_status"] = "synthetic_reproducible"
    metadata.to_csv(registry, index=False)
    data_path = tmp_path / "meta.csv"
    data.to_csv(data_path, index=False)

    def checked_filter(records):
        return filter_trainable_records(records, registry)

    monkeypatch.setattr(meta_learner, "filter_trainable_records", checked_filter)
    monkeypatch.setattr(evaluation, "filter_trainable_records", checked_filter)
    with pytest.raises(ValueError, match="Contradictory or unapproved provenance"):
        meta_learner.train_meta_model(data_path=data_path,
                                      artifact_path=tmp_path / "model.joblib")
    with pytest.raises(ValueError, match="Contradictory or unapproved provenance"):
        evaluation.evaluate_meta_learning(data_path=data_path,
                                          results_dir=tmp_path / "eval")
