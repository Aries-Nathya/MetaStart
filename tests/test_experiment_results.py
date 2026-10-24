import io
import json
import zipfile

import pandas as pd

from src.experiment_results import (
    DETAILED_COLUMNS,
    FLAT_COLUMNS,
    ExperimentRecord,
    build_experiment_downloads,
    build_recommendation_records,
)


def _record():
    return ExperimentRecord(
        experiment_id="experiment-1",
        method="grid_search",
        algorithm="Random Forest",
        stage="held_out_test",
        configuration="best",
        primary_metric_value=0.75,
        metrics={
            "accuracy": 0.8,
            "macro_precision": 0.7,
            "macro_recall": 0.75,
            "macro_f1": 0.75,
            "weighted_precision": 0.81,
            "weighted_recall": 0.8,
            "weighted_f1": 0.79,
            "micro_f1": 0.8,
            "roc_auc": 0.9,
            "pr_auc": 0.85,
        },
        class_breakdown=(
            {
                "Class": "positive",
                "Precision": 0.75,
                "Recall": 0.75,
                "F1": 0.75,
                "Support": 4,
                "True Positive": 3,
                "False Positive": 1,
                "False Negative": 1,
                "True Negative": 5,
            },
        ),
        cv_folds=3,
        cv_rows=12,
        cv_macro_f1=0.7,
        cv_macro_f1_std=0.02,
        cv_fit_seconds_mean=0.1,
        fit_seconds=0.3,
        test_rows=8,
        seed=7,
        trials_evaluated=4,
        best_parameters={"n_estimators": 100},
    )


def test_experiment_downloads_preserve_columns_metrics_and_open_as_files():
    downloads = build_experiment_downloads([_record()], {"dataset": "sample"})

    flat = pd.read_csv(io.BytesIO(downloads["experiment_results.csv"]))
    assert list(flat.columns) == list(FLAT_COLUMNS)
    assert flat.loc[0, "Macro F1"] == 0.75
    assert flat.loc[0, "ROC AUC"] == 0.9
    assert flat.loc[0, "CV Macro F1"] == 0.7
    assert flat.loc[0, "Trials Evaluated"] == 4

    detailed = pd.read_csv(io.BytesIO(downloads["experiment_detailed_metrics.csv"]))
    assert list(detailed.columns) == list(DETAILED_COLUMNS)
    assert detailed.loc[0, "Class"] == "positive"
    assert detailed.loc[0, "True Positive"] == 3
    assert detailed.loc[0, "True Negative"] == 5

    report = json.loads(downloads["experiment_report.json"])
    assert report["metadata"] == {"dataset": "sample"}
    assert report["experiments"][0]["metrics"]["macro_f1"] == 0.75

    with zipfile.ZipFile(io.BytesIO(downloads["experiment_results_bundle.zip"])) as archive:
        assert set(archive.namelist()) == {
            "experiment_results.csv",
            "experiment_detailed_metrics.csv",
            "experiment_report.json",
            "experiment_bundle_manifest.json",
        }
        assert pd.read_csv(io.BytesIO(archive.read("experiment_results.csv"))).loc[0, "Macro F1"] == 0.75
        assert json.loads(archive.read("experiment_report.json"))["experiments"][0]["experiment_id"] == "experiment-1"


def test_recommendation_records_keep_cv_and_failed_statuses_separate():
    ranking = pd.DataFrame([
        {"algorithm": "A", "predicted_macro_f1": 0.8},
        {"algorithm": "B", "predicted_macro_f1": 0.7},
    ])
    selection = pd.DataFrame([{
        "Algorithm": "A",
        "CV Macro-F1 Mean": 0.75,
        "CV Macro-F1 Std": 0.02,
        "CV Fit Seconds Mean": 0.1,
    }])
    classes = pd.DataFrame([{
        "Algorithm": "A", "Class": "positive", "Precision": 0.8,
        "Recall": 0.75, "F1": 0.77, "Support": 4,
    }])
    test_result = pd.DataFrame([{
        "Selected Algorithm": "A",
        "Held-out Test Accuracy": 0.8,
        "Held-out Test Macro-F1": 0.78,
        "Training seconds": 0.2,
    }])

    records = build_recommendation_records(
        ["A", "B"], ranking, selection,
        [("B", "cross-validation", "ValueError: failed")],
        selected_algorithm="A", test_result=test_result,
        class_metrics=classes, cv_source_rows=12, cv_rows=12, cv_n_splits=3,
        test_rows=4,
    )
    recommendation = {record.algorithm: record for record in records
                      if record.stage == "recommendation"}
    cv_record = next(record for record in records if record.stage == "cv")
    heldout = next(record for record in records if record.stage == "held_out_test")

    assert recommendation["A"].status == "completed"
    assert recommendation["B"].status.startswith("cv_failed:")
    assert recommendation["B"].primary_value == 0.7
    assert cv_record.class_breakdown[0]["Class"] == "positive"
    assert cv_record.cv_macro_f1 == 0.75
    assert heldout.metrics == {"accuracy": 0.8, "macro_f1": 0.78}
