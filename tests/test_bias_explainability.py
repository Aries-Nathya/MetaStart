import builtins
import io
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

import src.app_workflow as app_workflow
import src.recommender as recommender
import src.bias_explainability as bias_module
from src.benchmark import build_pipeline, get_models
from src.bias_explainability import (
    IMBALANCE_RATIO_THRESHOLD,
    class_balance_diagnostics,
    run_bias_explainability,
)
from src.experiment_results import build_experiment_downloads


def _small_fitted_data():
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.iloc[:180].reset_index(drop=True)
    y = bunch.target.iloc[:180].reset_index(drop=True)
    # Ensure each class has adequate stratified CV support in this deterministic slice.
    X = bunch.data.reset_index(drop=True)
    y = bunch.target.reset_index(drop=True)
    X_train = X.iloc[:160].reset_index(drop=True)
    y_train = y.iloc[:160].reset_index(drop=True)
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    fitted = build_pipeline(model, X_train).fit(X_train, y_train)
    factory = get_models
    # The production app's tree candidate configuration is equivalent here.
    return X_train, y_train, fitted, factory


def test_class_balance_counts_proportions_threshold_and_empty_edge_case():
    table, ratio, flagged = class_balance_diagnostics(pd.Series(["a"] * 6 + ["b"] * 2))
    assert table.set_index("Class")["Count"].to_dict() == {"a": 6, "b": 2}
    assert table.set_index("Class")["Proportion"].to_dict() == {"a": 0.75, "b": 0.25}
    assert ratio == IMBALANCE_RATIO_THRESHOLD
    assert flagged is True

    balanced, balanced_ratio, balanced_flag = class_balance_diagnostics(pd.Series([0, 0, 1, 1]))
    assert balanced_ratio == 1.0
    assert balanced_flag is False
    empty, empty_ratio, empty_flag = class_balance_diagnostics(pd.Series([], dtype=object))
    assert empty.empty and np.isnan(empty_ratio) and empty_flag is False


def test_oof_class_metrics_and_confusion_counts_are_consistent():
    X, y, fitted, factory = _small_fitted_data()
    result = run_bias_explainability(
        X, y, selected_algorithm="Decision Tree", fitted_pipeline=fitted,
        model_factory=factory, pipeline_builder=build_pipeline,
        cv_sample_limit=120, explanation_sample_limit=24, local_feature_limit=7,
    )
    assert set(result["class_metrics"]["Class"]) == {"0", "1"}
    assert result["class_metrics"]["Support"].sum() == result["confusion_matrix"]["Count"].sum()
    assert result["confusion_matrix"]["Count"].sum() == result["class_metrics"]["Support"].sum()
    assert result["error_counts"]["Count"].sum() == sum(
        row["False Negative"] for row in result["class_metrics"].to_dict(orient="records")
    )
    assert result["class_metrics"]["Precision"].between(0, 1).all()
    assert result["class_metrics"]["Recall"].between(0, 1).all()
    assert "No sensitive attributes" in result["fairness_limitation"]


def test_explanation_results_are_bounded_and_do_not_call_selection_path(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("diagnostics must not invoke production selection")

    monkeypatch.setattr(recommender, "recommend_algorithms", forbidden)
    monkeypatch.setattr(app_workflow, "evaluate_ranked_candidates", forbidden)
    X, y, fitted, factory = _small_fitted_data()
    result = run_bias_explainability(
        X, y, selected_algorithm="Decision Tree", fitted_pipeline=fitted,
        model_factory=factory, pipeline_builder=build_pipeline,
        cv_sample_limit=90, explanation_sample_limit=18, local_feature_limit=5,
    )
    explanation_records = [record for record in result["records"] if record.stage == "explainability"]
    assert {record.method for record in explanation_records} == {
        "permutation_importance", "single_feature_ablation_sensitivity",
    }
    global_record = next(record for record in explanation_records if record.method == "permutation_importance")
    local_record = next(record for record in explanation_records if record.method.startswith("single_feature"))
    assert global_record.leaderboard_metrics["sample_count"] <= 18
    assert global_record.leaderboard_metrics["repeats"] == 3
    assert local_record.leaderboard_metrics["sample_count"] == 1
    assert local_record.leaderboard_metrics["features_explained"] <= 5
    assert len(result["importance"]) == X.shape[1]
    assert len(result["local_explanation"]) <= 5
    assert "not causation" in global_record.leaderboard_metrics["interpretation"]


def test_shap_unavailable_does_not_break_documented_permutation_alternative(monkeypatch):
    original_import = builtins.__import__

    def reject_shap(name, *args, **kwargs):
        if name == "shap" or name.startswith("shap."):
            raise ImportError("SHAP intentionally unavailable in this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_shap)
    X, y, fitted, factory = _small_fitted_data()
    result = run_bias_explainability(
        X, y, selected_algorithm="Decision Tree", fitted_pipeline=fitted,
        model_factory=factory, pipeline_builder=build_pipeline,
        cv_sample_limit=75, explanation_sample_limit=12, local_feature_limit=3,
    )
    assert result["explanation_status"] == "completed"
    assert not result["importance"].empty


def test_permutation_failure_is_recorded_and_local_explanation_continues(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError("optional importance dependency unavailable")

    monkeypatch.setattr(bias_module, "permutation_importance", unavailable)
    X, y, fitted, factory = _small_fitted_data()
    result = run_bias_explainability(
        X, y, selected_algorithm="Decision Tree", fitted_pipeline=fitted,
        model_factory=factory, pipeline_builder=build_pipeline,
        cv_sample_limit=60, explanation_sample_limit=10, local_feature_limit=2,
    )
    failed = next(record for record in result["records"] if record.stage == "explainability")
    assert failed.status == "skipped_or_failed"
    assert "optional importance dependency unavailable" in failed.warnings[0]
    assert not result["local_explanation"].empty


def test_bias_and_explainability_diagnostics_are_exported_without_touching_leakage_records():
    X, y, fitted, factory = _small_fitted_data()
    result = run_bias_explainability(
        X, y, selected_algorithm="Decision Tree", fitted_pipeline=fitted,
        model_factory=factory, pipeline_builder=build_pipeline,
        cv_sample_limit=60, explanation_sample_limit=12, local_feature_limit=4,
    )
    leakage = __import__("src.leakage_experiment", fromlist=["run_leakage_demonstration"])
    leakage_records, _ = leakage.run_leakage_demonstration()
    downloads = build_experiment_downloads([*leakage_records, *result["records"]], {
        "diagnostic_sample_count": result["importance_sample_count"],
    })

    flat = pd.read_csv(io.BytesIO(downloads["experiment_results.csv"]))
    bias_row = flat.loc[flat["Stage"].eq("bias")].iloc[0]
    assert bias_row["Framework Metrics"]
    assert "confusion_matrix" in bias_row["Framework Metrics"]
    assert "Warnings" in flat.columns and "Failures" in flat.columns
    assert flat.loc[flat["Stage"].eq("leakage"), "Status"].tolist() == [
        "completed", "invalid_for_real_world",
    ]
    detailed = pd.read_csv(io.BytesIO(downloads["experiment_detailed_metrics.csv"]))
    assert "True Positive" in detailed.columns
    assert "False Negative" in detailed.columns
    assert detailed["Importance Mean"].notna().any()
    assert detailed["Score Change"].notna().any()
    assert detailed["Class Count"].notna().any()
    assert detailed["Explanation Method"].notna().any()
    report = json.loads(downloads["experiment_report.json"])
    explain = [record for record in report["experiments"] if record["stage"] == "explainability"]
    assert {record["method"] for record in explain} == {
        "permutation_importance", "single_feature_ablation_sensitivity",
    }
    permutation_record = next(record for record in explain if record["method"] == "permutation_importance")
    assert permutation_record["leaderboard_metrics"]["sample_count"] == 12
    assert permutation_record["leaderboard_metrics"]["feature_importance"]
    with __import__("zipfile").ZipFile(io.BytesIO(downloads["experiment_results_bundle.zip"])) as archive:
        assert archive.testzip() is None
        zipped_report = json.loads(archive.read("experiment_report.json"))
        assert len(zipped_report["experiments"]) == len(report["experiments"])
