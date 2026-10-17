from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.app_helpers import (build_algorithm_comparison, build_selection_rationale,
                             dataset_diagnostics_table, feature_structure_tables)
from src.app_workflow import (
    CV_MAX_ROWS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_CELLS,
    MAX_UPLOAD_FEATURES,
    MAX_UPLOAD_ROWS,
    UploadLimits,
    UploadValidationError,
    evaluate_ranked_candidates,
    load_uploaded_csv,
    prepare_classification_data,
    recommendation_download_table,
    _full_fit_resource_reason,
)
from src.benchmark import ALGORITHM_NAMES


def _mixed_csv_bytes():
    rows = []
    for index in range(12):
        target = "positive" if index < 6 else "negative"
        rows.append({
            "number": "" if index == 2 else str(index),
            "category": "" if index == 3 else ("red" if index % 2 else "blue"),
            "all_missing": "",
            "target": "" if index == 11 else target,
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def test_mixed_csv_upload_normalizes_missing_values_and_features():
    frame = load_uploaded_csv(BytesIO(_mixed_csv_bytes()))
    prepared = prepare_classification_data(frame, "target")

    assert prepared.X.shape == (11, 2)
    assert prepared.X["number"].isna().sum() == 1
    assert prepared.X["category"].isna().sum() == 1
    assert prepared.dropped_target_rows == 1
    assert prepared.dropped_all_missing_features == ("all_missing",)
    assert prepared.y.value_counts().to_dict() == {"positive": 6, "negative": 5}
    assert prepared.X["category"].dropna().map(type).eq(str).all()


@pytest.mark.parametrize("payload,message", [
    (b"", "empty or has no CSV header"),
    (b"a,b\n1,2,3\n", "Malformed CSV"),
])
def test_csv_reader_reports_empty_and_malformed_uploads(payload, message):
    with pytest.raises(UploadValidationError, match=message):
        load_uploaded_csv(payload)


def test_csv_reader_reports_non_utf8_upload():
    with pytest.raises(UploadValidationError, match="encoding.*UTF-8"):
        load_uploaded_csv(b"feature,target\n\xff,0\n")


def test_validation_requires_a_selected_existing_target():
    frame = pd.DataFrame({"x": range(8), "target": [0] * 4 + [1] * 4})
    for target in (None, "missing"):
        with pytest.raises(UploadValidationError, match="Choose an existing target"):
            prepare_classification_data(frame, target)


def test_validation_rejects_single_class_and_insufficient_class_examples():
    with pytest.raises(UploadValidationError, match="at least two distinct classes"):
        prepare_classification_data(
            pd.DataFrame({"x": range(8), "target": [1] * 8}), "target")
    with pytest.raises(UploadValidationError, match="at least 4 rows"):
        prepare_classification_data(
            pd.DataFrame({"x": range(7), "target": [0] * 4 + [1] * 3}), "target")


def test_validation_drops_all_missing_columns_and_rejects_no_usable_features():
    frame = pd.DataFrame({"x": range(8), "empty": [np.nan] * 8,
                          "target": [0] * 4 + [1] * 4})
    prepared = prepare_classification_data(frame, "target")
    assert prepared.X.columns.tolist() == ["x"]
    assert prepared.dropped_all_missing_features == ("empty",)

    no_features = pd.DataFrame({"x": [np.nan] * 8, "target": [0] * 4 + [1] * 4})
    with pytest.raises(UploadValidationError, match="No usable feature columns"):
        prepare_classification_data(no_features, "target")


def test_validation_rejects_unsupported_feature_types_and_ambiguous_target():
    unsupported = pd.DataFrame({
        "x": [[1], [2], [3], [4], [5], [6], [7], [8]],
        "target": [0] * 4 + [1] * 4,
    })
    with pytest.raises(UploadValidationError, match="unsupported values"):
        prepare_classification_data(unsupported, "target")

    continuous_target = pd.DataFrame({"x": range(120), "target": range(120)})
    with pytest.raises(UploadValidationError, match="app currently supports classification only"):
        prepare_classification_data(continuous_target, "target")


def test_validation_enforces_row_feature_and_class_limits():
    frame = pd.DataFrame({"x": range(8), "target": [0] * 4 + [1] * 4})
    with pytest.raises(UploadValidationError, match="row limit"):
        prepare_classification_data(frame, "target", max_rows=7)
    wide = pd.DataFrame({f"x{i}": range(8) for i in range(3)})
    wide["target"] = [0] * 4 + [1] * 4
    with pytest.raises(UploadValidationError, match="feature columns"):
        prepare_classification_data(wide, "target", max_features=2)
    assert MAX_UPLOAD_ROWS > 0
    assert MAX_UPLOAD_FEATURES > 0


def test_configured_upload_limits_cover_cover_type_and_allow_lower_operator_limits():
    assert MAX_UPLOAD_ROWS >= 581_012
    assert MAX_UPLOAD_FEATURES >= 54
    assert MAX_UPLOAD_CELLS >= 581_012 * 55
    assert MAX_UPLOAD_BYTES >= 100 * 1024 * 1024
    assert CV_MAX_ROWS == 5_000
    local = UploadLimits(max_rows=120, max_file_bytes=1_000_000,
                         max_features=8, max_cells=250)
    frame = pd.DataFrame({"x": range(120), "target": [0] * 60 + [1] * 60})
    assert len(prepare_classification_data(frame, "target", max_rows=local.max_rows,
                                           max_features=local.max_features,
                                           max_cells=local.max_cells).y) == 120
    with pytest.raises(UploadValidationError, match="processing budget"):
        prepare_classification_data(frame.assign(z=range(120)), "target",
                                   max_rows=local.max_rows,
                                   max_features=local.max_features,
                                   max_cells=local.max_cells)


def test_upload_parser_enforces_file_and_row_limits_before_materializing_over_limit_data():
    small_limits = UploadLimits(max_rows=100, max_file_bytes=100_000,
                                max_features=5, max_cells=10_000)
    frame = pd.DataFrame({"x": range(100), "target": [0] * 50 + [1] * 50})
    accepted = load_uploaded_csv(frame.to_csv(index=False).encode(), limits=small_limits)
    assert len(accepted) == 100
    over_rows = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(UploadValidationError, match="100-row limit"):
        load_uploaded_csv(over_rows.to_csv(index=False).encode(), limits=small_limits)
    with pytest.raises(UploadValidationError, match="configured limit"):
        load_uploaded_csv(b"x" * 100_001, limits=small_limits)


@pytest.mark.parametrize("row_count", [49_999, 50_001, MAX_UPLOAD_ROWS - 1])
def test_prepare_accepts_below_old_cap_above_old_cap_and_near_configured_max(row_count):
    frame = pd.DataFrame({"x": np.arange(row_count, dtype=np.int32),
                          "target": np.arange(row_count, dtype=np.int8) % 2})
    prepared = prepare_classification_data(frame, "target")
    assert len(prepared.X) == row_count


def test_large_csv_upload_prepares_and_runs_cv_on_disclosed_bounded_sample():
    row_count = 70_001
    target = np.arange(row_count, dtype=np.int8) % 2
    source = pd.DataFrame({"signal": target.astype(np.float32),
                           "noise": np.arange(row_count, dtype=np.int32) % 101,
                           "target": target})
    parsed = load_uploaded_csv(source.to_csv(index=False).encode("utf-8"))
    prepared = prepare_classification_data(parsed, "target")
    X_train, X_test = prepared.X.iloc[:-17_501], prepared.X.iloc[-17_501:]
    y_train, y_test = prepared.y.iloc[:-17_501], prepared.y.iloc[-17_501:]
    evaluation = evaluate_ranked_candidates(
        _ranking(), X_train, y_train, X_test, y_test,
        model_factory=_candidate_factory(), pipeline_builder=_pipeline_builder,
    )
    assert len(prepared.X) == row_count
    assert evaluation.cv_source_rows > 50_000
    assert evaluation.cv_rows == CV_MAX_ROWS
    assert evaluation.test_result.iloc[0]["Selected Algorithm"] == evaluation.selected_algorithm
    assert "Held-out Test Macro-F1" not in evaluation.selection_results


def test_large_candidate_refit_resource_guards_are_specific_and_actionable():
    svm = _full_fit_resource_reason("SVM", 20_001, 5_000)
    forest = _full_fit_resource_reason("Random Forest", 250_001, 5_000)
    knn = _full_fit_resource_reason("k-Nearest Neighbors", 40_000, 10_000)
    logistic = _full_fit_resource_reason("Logistic Regression", 581_012, 145_253)
    assert "RBF SVM full-fit limit" in svm
    assert "Random Forest safety limit" in forest
    assert "train/test row pairs" in knn
    assert logistic is None


class _LabelClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, fail_cv=False, fail_full_size=None):
        self.fail_cv = fail_cv
        self.fail_full_size = fail_full_size

    def fit(self, X, y):
        if self.fail_cv or (self.fail_full_size is not None and len(X) == self.fail_full_size):
            raise ValueError("intentional candidate failure")
        self.label_ = pd.Series(y).value_counts().idxmax()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.repeat(self.label_, len(X))


def _candidate_data():
    return (pd.DataFrame({"x": np.arange(40), "z": np.tile([0, 1], 20)}),
            pd.Series([0] * 20 + [1] * 20),
            pd.DataFrame({"x": [41, 42, 43, 44], "z": [1, 0, 1, 0]}))


def _ranking():
    return pd.DataFrame({
        "algorithm": ALGORITHM_NAMES,
        "predicted_macro_f1": [0.95, 0.90, 0.85, 0.80, 0.75],
    })


def _profile():
    return {
        "n_rows": 90.0,
        "n_features": 3.0,
        "n_numeric": 2.0,
        "n_categorical": 1.0,
        "missing_ratio": 0.05,
        "n_classes": 3.0,
        "imbalance_ratio": 3.5,
    }


def _cv_results(scores):
    return pd.DataFrame([
        {"Algorithm": name, "CV Macro-F1 Mean": score,
         "CV Macro-F1 Std": 0.02, "CV Fit Seconds Mean": 0.04}
        for name, score in scores.items()
    ])


def test_recommendation_download_has_all_five_models_scores_and_prediction_order():
    ranking = _ranking().iloc[[2, 0, 4, 1, 3]].reset_index(drop=True)
    table = recommendation_download_table(ranking)
    csv_table = pd.read_csv(BytesIO(table.to_csv(index=False).encode("utf-8")))

    expected = ranking.sort_values("predicted_macro_f1", ascending=False, kind="stable")
    assert table.columns.tolist() == ["Rank", "Algorithm", "Predicted Macro-F1"]
    assert len(table) == 5
    assert table["Algorithm"].is_unique
    assert table["Algorithm"].tolist() == expected["algorithm"].tolist()
    assert table["Rank"].tolist() == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(table["Predicted Macro-F1"], expected["predicted_macro_f1"])
    assert csv_table.equals(table)


def test_comparison_and_selection_rationale_match_actual_prediction_and_cv_scores():
    ranking = _ranking()
    scores = {
        "Logistic Regression": 0.900,
        "Decision Tree": 0.895,
        "Random Forest": 0.890,
        "SVM": 0.810,
    }
    cv = _cv_results(scores)
    failures = (("Logistic Regression", "final fit", "injected full fit failure"),)
    selected = "Decision Tree"

    comparison = build_algorithm_comparison(
        ranking, cv, _profile(), failures, selected_algorithm=selected,
        full_training_seconds=0.12,
    )
    rationale = build_selection_rationale(cv, failures, selected, ranking)

    assert len(comparison) == 5
    assert comparison["Algorithm"].tolist() == ranking["algorithm"].tolist()
    assert comparison.loc[0, "Predicted rank"] == 1
    assert comparison.loc[0, "Predicted Macro-F1"] == pytest.approx(0.95)
    selected_row = comparison.loc[comparison["Algorithm"].eq(selected)].iloc[0]
    assert selected_row["Mean training-only CV Macro-F1"] == pytest.approx(0.895)
    assert selected_row["Selected full-training fit seconds"] == pytest.approx(0.12)
    assert "CV completed" in selected_row["CV status"]
    assert "90 rows" in selected_row["Dataset-specific suitability evidence"]
    assert "one-hot encoded" in selected_row["Dataset-specific suitability evidence"]
    assert "class balancing" in comparison.loc[
        comparison["Algorithm"].eq("Random Forest"), "Dataset-specific suitability evidence"
    ].iloc[0]
    assert "Decision Tree" in rationale["selected"]
    assert "0.895" in rationale["selected"]
    assert "rank 2" in rationale["selected"]
    assert "Logistic Regression" in rationale["selected"]
    assert "could not be used for full-data evaluation" in rationale["selected"]
    assert any("within 0.010" in text for text in rationale["alternatives"])
    assert any("lower than the selected candidate by 0.085" in text
               for text in rationale["alternatives"])


def test_close_cv_scores_are_described_as_close_not_meaningfully_inferior():
    ranking = _ranking()
    cv = _cv_results({
        "Logistic Regression": 0.881,
        "Decision Tree": 0.875,
        "Random Forest": 0.70,
    })
    rationale = build_selection_rationale(cv, (), "Logistic Regression", ranking)
    decision_tree_reason = next(
        text for text in rationale["alternatives"] if text.startswith("Decision Tree:")
    )
    assert "within 0.010" in decision_tree_reason
    assert "does not establish that it is meaningfully worse" in decision_tree_reason


def test_missing_and_failed_cv_results_are_not_interpreted_as_low_scores():
    ranking = _ranking()
    cv = _cv_results({"Logistic Regression": 0.84, "Random Forest": 0.80})
    failures = (("Decision Tree", "cross-validation", "ValueError: insufficient fold data"),)

    comparison = build_algorithm_comparison(ranking, cv, _profile(), failures,
                                            selected_algorithm="Logistic Regression")
    rationale = build_selection_rationale(cv, failures, "Logistic Regression", ranking)
    failed_row = comparison.loc[comparison["Algorithm"].eq("Decision Tree")].iloc[0]
    not_evaluated_row = comparison.loc[comparison["Algorithm"].eq("SVM")].iloc[0]
    failed_reason = next(text for text in rationale["alternatives"] if text.startswith("Decision Tree:"))
    not_evaluated_reason = next(text for text in rationale["alternatives"] if text.startswith("SVM:"))

    assert "CV failed" in failed_row["CV status"]
    assert pd.isna(failed_row["Mean training-only CV Macro-F1"])
    assert "measured performance is unknown" in not_evaluated_row["CV status"]
    assert "no score-based comparison is justified" in failed_reason
    assert "cannot be compared" in not_evaluated_reason


def test_no_selection_explanation_distinguishes_cv_failure_from_final_fit_failure():
    ranking = _ranking()
    completed_cv = _cv_results({"Logistic Regression": 0.84})
    fit_failure = (("Logistic Regression", "final fit", "RuntimeError: estimator failed"),)
    fit_rationale = build_selection_rationale(completed_cv, fit_failure, None, ranking)
    cv_rationale = build_selection_rationale(
        pd.DataFrame(), (("Logistic Regression", "cross-validation", "ValueError: cv failed"),),
        None, ranking,
    )
    assert "CV completed" in fit_rationale["selected"]
    assert "full-training fit" in fit_rationale["selected"]
    assert "No candidate completed training-only CV" in cv_rationale["selected"]


def _candidate_factory(fail_cv=(), fail_full=()):
    return lambda: {
        name: _LabelClassifier(fail_cv=name in fail_cv,
                               fail_full_size=40 if name in fail_full else None)
        for name in ALGORITHM_NAMES
    }


def _pipeline_builder(model, _X):
    from sklearn.pipeline import make_pipeline
    return make_pipeline(model)


def test_selection_uses_training_cv_not_test_labels_and_preserves_full_ranking():
    ranking = _ranking()
    original_ranking = ranking.copy(deep=True)
    X_train, y_train, X_test = _candidate_data()
    first = evaluate_ranked_candidates(ranking, X_train, y_train, X_test,
        pd.Series([0, 0, 1, 1]), model_factory=_candidate_factory(),
        pipeline_builder=_pipeline_builder)
    second = evaluate_ranked_candidates(ranking, X_train, y_train, X_test,
        pd.Series([1, 1, 0, 0]), model_factory=_candidate_factory(),
        pipeline_builder=_pipeline_builder)

    assert ranking.equals(original_ranking)
    assert len(ranking) == 5
    deterministic_columns = [
        column for column in first.selection_results
        if column != "CV Fit Seconds Mean"
    ]
    pd.testing.assert_frame_equal(
        first.selection_results[deterministic_columns],
        second.selection_results[deterministic_columns],
    )
    assert first.selected_algorithm == second.selected_algorithm
    assert set(first.selection_results["Algorithm"]) == set(ALGORITHM_NAMES)
    assert first.cv_rows == first.cv_source_rows == len(X_train)
    assert first.cv_n_splits == 3
    assert set(first.class_metrics["Algorithm"]) == set(ALGORITHM_NAMES)
    assert "Held-out Test Macro-F1" not in first.class_metrics.columns
    assert first.boundary_diagnostic.iloc[0]["Diagnostic"].startswith("Linear-kernel SVM")
    assert len(first.test_result) == 1
    assert first.test_result.iloc[0]["Selected Algorithm"] == first.selected_algorithm
    assert "Held-out Test Macro-F1" not in first.selection_results.columns
    assert set(first.fitted_models) == {first.selected_algorithm}


def test_cv_failure_is_reported_and_other_candidates_continue():
    X_train, y_train, X_test = _candidate_data()
    failed_name = ALGORITHM_NAMES[0]
    evaluation = evaluate_ranked_candidates(
        _ranking(), X_train, y_train, X_test, pd.Series([0, 1, 0, 1]),
        model_factory=_candidate_factory(fail_cv={failed_name}),
        pipeline_builder=_pipeline_builder,
    )
    assert evaluation.failures[0][:2] == (failed_name, "cross-validation")
    assert failed_name not in evaluation.selection_results["Algorithm"].tolist()
    assert evaluation.selected_algorithm is not None


def test_final_fit_failure_falls_back_by_cv_rank_and_cv_all_failure_is_clean():
    X_train, y_train, X_test = _candidate_data()
    evaluation = evaluate_ranked_candidates(
        _ranking(), X_train, y_train, X_test, pd.Series([0, 1, 0, 1]),
        model_factory=_candidate_factory(fail_full={ALGORITHM_NAMES[0]}),
        pipeline_builder=_pipeline_builder,
    )
    assert evaluation.failures[0][:2] == (ALGORITHM_NAMES[0], "final fit")
    assert evaluation.selected_algorithm != ALGORITHM_NAMES[0]
    assert len(evaluation.test_result) == 1

    partial_fail = evaluate_ranked_candidates(
        _ranking(), X_train, y_train, X_test, pd.Series([0, 1, 0, 1]),
        model_factory=_candidate_factory(fail_cv=set(ALGORITHM_NAMES[:3])),
        pipeline_builder=_pipeline_builder,
    )
    assert set(partial_fail.selection_results["Algorithm"]) == set(ALGORITHM_NAMES[3:])
    assert partial_fail.selected_algorithm is not None
    assert len([failure for failure in partial_fail.failures
                if failure[1] == "cross-validation"]) == 3
    # Measured-model failures do not affect the separate recommendation export.
    exported = recommendation_download_table(_ranking())
    assert len(exported) == 5
    assert exported["Algorithm"].is_unique


def test_dataset_diagnostics_reflect_class_balance_types_missingness_and_numeric_shape():
    profile = {
        "n_rows": 20, "n_features": 3, "n_classes": 2, "n_numeric": 2,
        "n_categorical": 1, "missing_ratio": .1, "imbalance_ratio": 3,
        "normalized_class_entropy": .8, "median_abs_numeric_skewness": .3,
        "median_abs_numeric_excess_kurtosis": .4, "mean_abs_numeric_correlation": .2,
        "fraction_numeric_pairs_abs_corr_ge_0_8": .1,
        "mean_normalized_categorical_entropy": .9, "mean_categorical_cardinality": 4,
    }
    table = dataset_diagnostics_table(profile, pd.Series(["a"] * 15 + ["b"] * 5))
    values = dict(zip(table["Characteristic"], table["Value"]))
    assert values["Categorical features"] == "1"
    assert values["Missing feature cells"] == "0.100"
    assert values["Class a (count; share)"] == "15; 75.0%"
    assert "numeric correlation" in build_algorithm_comparison(
        _ranking(), _cv_results({"Logistic Regression": .8}), profile
    ).iloc[0]["Dataset-specific suitability evidence"]


def test_feature_structure_diagnostics_show_mixed_distributions_and_correlations():
    frame = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, np.nan],
        "x_copy": [2.0, 4.0, 6.0, 8.0],
        "kind": ["a", "a", "b", None],
    })
    features, correlations = feature_structure_tables(frame)
    assert features["Feature"].tolist() == ["x", "x_copy", "kind"]
    assert features.loc[features["Feature"].eq("kind"), "Distinct values"].iloc[0] == 2
    assert features.loc[features["Feature"].eq("x"), "Missing share"].iloc[0] == .25
    assert correlations.iloc[0]["Absolute Pearson correlation"] == pytest.approx(1.0)


def test_dataset_specific_evidence_changes_with_measured_dataset_profile():
    ranking = _ranking()
    cv = _cv_results({"Logistic Regression": .82, "SVM": .81})
    numeric_profile = {**_profile(), "n_rows": 120, "n_categorical": 0,
                      "missing_ratio": 0.0, "imbalance_ratio": 1.0,
                      "mean_abs_numeric_correlation": .05}
    mixed_profile = {**_profile(), "n_rows": 2400, "n_categorical": 4,
                     "missing_ratio": .2, "imbalance_ratio": 6.0,
                     "mean_abs_numeric_correlation": .72}
    first = build_algorithm_comparison(ranking, cv, numeric_profile).iloc[0][
        "Dataset-specific suitability evidence"]
    second = build_algorithm_comparison(ranking, cv, mixed_profile).iloc[0][
        "Dataset-specific suitability evidence"]
    assert "120 rows" in first and "0.0% missing feature cells" in first
    assert "2,400 rows" in second and "20.0% missing feature cells" in second
    assert "4 categorical" in second and "0.72" in second
    assert first != second


def test_selection_rationale_reports_exact_scores_and_does_not_accept_test_metrics():
    import inspect

    ranking = _ranking()
    cv = _cv_results({
        "Logistic Regression": .9647, "SVM": .9624,
        "k-Nearest Neighbors": .9538,
    })
    result = build_selection_rationale(cv, (), "Logistic Regression", ranking)
    assert "test_result" not in inspect.signature(build_selection_rationale).parameters
    assert "heldout_test" not in inspect.signature(build_selection_rationale).parameters
    assert "0.965" in result["selected"]
    assert "0.002" in result["selected"]
    assert "not a significance test" in result["selected"]
    svm_reason = next(text for text in result["alternatives"] if text.startswith("SVM:"))
    assert "0.002" in svm_reason or "0.002" in result["selected"]


def test_large_dataset_multiclass_imbalanced_and_missing_values():
    row_count = 52_000
    rng = np.random.default_rng(42)
    # 4 imbalanced classes
    classes = np.array(["alpha", "beta", "gamma", "delta"])
    probs = [0.60, 0.25, 0.10, 0.05]
    y_raw = rng.choice(classes, size=row_count, p=probs)
    # Features with numeric, categorical, and missing values
    num_feat = rng.normal(0, 1, size=row_count)
    num_feat[rng.choice(row_count, size=500, replace=False)] = np.nan
    cat_feat = rng.choice(["low", "med", "high"], size=row_count)
    frame = pd.DataFrame({
        "num": num_feat,
        "cat": cat_feat,
        "target": y_raw,
    })

    prepared = prepare_classification_data(frame, "target")
    assert len(prepared.X) == row_count
    assert prepared.X["num"].isna().sum() == 500
    assert len(prepared.y.unique()) == 4

    X_train, X_test = prepared.X.iloc[:39_000], prepared.X.iloc[39_000:]
    y_train, y_test = prepared.y.iloc[:39_000], prepared.y.iloc[39_000:]
    evaluation = evaluate_ranked_candidates(
        _ranking(), X_train, y_train, X_test, y_test,
        model_factory=_candidate_factory(), pipeline_builder=_pipeline_builder,
    )
    assert evaluation.cv_source_rows == 39_000
    assert evaluation.cv_rows == CV_MAX_ROWS
    assert len(evaluation.selection_results) == 5
    assert len(evaluation.test_result) == 1
    assert "Held-out Test Macro-F1" not in evaluation.selection_results.columns


def test_actionable_error_messages_on_exceeded_limits():
    small_limits = UploadLimits(max_rows=50_000, max_features=5, max_cells=200_000)
    over_rows = pd.DataFrame({"x": range(50_001), "target": [0] * 25000 + [1] * 25001})
    with pytest.raises(UploadValidationError, match="METASTART_MAX_UPLOAD_ROWS"):
        load_uploaded_csv(over_rows.to_csv(index=False).encode(), limits=small_limits)

    wide_data = pd.DataFrame({f"feat_{i}": range(10) for i in range(6)})
    wide_data["target"] = [0] * 5 + [1] * 5
    with pytest.raises(UploadValidationError, match="METASTART_MAX_UPLOAD_FEATURES"):
        load_uploaded_csv(wide_data.to_csv(index=False).encode(), limits=small_limits)

    high_cell_limits = UploadLimits(max_rows=100_000, max_features=10, max_cells=50_000)
    high_cell_data = pd.DataFrame({f"f_{i}": range(15_000) for i in range(4)})
    high_cell_data["target"] = [0] * 7500 + [1] * 7500
    with pytest.raises(UploadValidationError, match="METASTART_MAX_UPLOAD_CELLS"):
        load_uploaded_csv(high_cell_data.to_csv(index=False).encode(), limits=high_cell_limits)
