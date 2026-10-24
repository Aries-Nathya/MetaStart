import io
import json
import zipfile

from src.experiment_results import ExperimentRecord, build_experiment_downloads
from src.model_card import MODEL_CARD_MIME_TYPE, build_model_card


def _records():
    return [
        ExperimentRecord(
            experiment_id="recommendation::SVM", method="meta_start_production",
            algorithm="SVM", stage="recommendation", status="completed",
            metrics={"macro_f1": 0.61}, primary_metric_value=0.61,
            notes=("Predicted score is an estimate.",),
        ),
        ExperimentRecord(
            experiment_id="cv::SVM", method="meta_start_candidate_evaluation",
            algorithm="SVM", stage="cv", status="completed",
            metrics={"macro_f1": 0.82}, primary_metric_value=0.82,
            cv_macro_f1=0.82, cv_macro_f1_std=0.02, cv_fit_seconds_mean=0.12,
            cv_folds=3, cv_rows=90, cv_source_rows=100, seed=42,
            class_breakdown=(
                {"Class": "negative", "Precision": 0.84, "Recall": 0.80,
                 "F1": 0.82, "Support": 40},
                {"Class": "positive", "Precision": 0.83, "Recall": 0.85,
                 "F1": 0.84, "Support": 50},
            ),
        ),
        ExperimentRecord(
            experiment_id="held_out_test::SVM", method="meta_start_production",
            algorithm="SVM", stage="held_out_test", status="completed",
            metrics={"accuracy": 0.81, "macro_f1": 0.80}, primary_metric_value=0.80,
            fit_seconds=0.4, test_rows=25, seed=42,
        ),
        ExperimentRecord(
            experiment_id="grid::SVM", method="grid_search", algorithm="SVM",
            stage="grid_search", status="completed", metrics={"macro_f1": 0.83},
            fit_seconds=1.7, cv_folds=3, cv_rows=90, trials_evaluated=4,
        ),
        ExperimentRecord(
            experiment_id="external::flaml", method="flaml_automl", algorithm="FLAML",
            stage="external_automl", status="completed", metrics={"macro_f1": 0.79},
            fit_seconds=60.0, cv_folds=3, cv_rows=100,
            leaderboard_metrics={"time_budget_seconds": 60},
        ),
        ExperimentRecord(
            experiment_id="external::flaml::test", method="flaml_automl", algorithm="FLAML: RF",
            stage="external_automl_held_out_test", status="completed",
            metrics={"accuracy": 0.76, "macro_f1": 0.75}, test_rows=25,
        ),
        ExperimentRecord(
            experiment_id="explain::global", method="permutation_importance", algorithm="SVM",
            stage="explainability", status="completed", test_rows=20,
            leaderboard_metrics={
                "method": "permutation importance",
                "sample_count": 20,
                "runtime_seconds": 0.3,
                "feature_importance": [
                    {"Feature": "feature_a", "Importance Mean": 0.14, "Importance Std": 0.02},
                ],
            },
        ),
        ExperimentRecord(
            experiment_id="explain::local", method="single_feature_ablation_sensitivity",
            algorithm="SVM", stage="explainability", status="completed", test_rows=1,
            leaderboard_metrics={
                "method": "single-feature replacement", "sample_count": 1,
                "runtime_seconds": 0.01, "score_type": "signed decision score",
                "local_feature_effects": [
                    {"Feature": "feature_a", "Score change after replacement": 0.2},
                ],
            },
        ),
        ExperimentRecord(
            experiment_id="leak::bad", method="deliberate_target_leakage_demo",
            algorithm="Decision Tree", stage="leakage", status="invalid_for_real_world",
            metrics={"macro_f1": 1.0}, primary_metric_value=1.0,
            leak=True, invalid_for_real_world=True,
        ),
    ]


def _metadata(**overrides):
    metadata = {
        "selected_algorithm": "SVM",
        "model_name": "MetaStart SVM classifier",
        "model_version": "IA-3 prototype v0",
        "card_created_at": "2026-10-23T12:00:00+00:00",
        "run_id": "current-upload-hash:target",
        "dataset_name": "toy.csv",
        "dataset_source": "User-uploaded CSV",
        "dataset_rows": 125,
        "dataset_features": 4,
        "target": "label",
        "class_distribution": [
            {"class": "negative", "count": 50, "proportion": 0.4},
            {"class": "positive", "count": 75, "proportion": 0.6},
        ],
        "imbalance_ratio": 1.5,
        "imbalance_flagged": False,
        "selected_model_parameters": {"C": 1.0, "kernel": "rbf"},
        "training_rows": 100,
        "held_out_rows": 25,
        "resource_limits": {"cv_max_rows": 5000},
        "environment": "Python test; sklearn test",
    }
    metadata.update(overrides)
    return metadata


def test_model_card_contains_required_sections_and_current_run_metadata():
    card = build_model_card(_records(), _metadata())
    for heading in (
        "## Purpose and intended users",
        "## Dataset and preprocessing",
        "## Split and reproducibility",
        "## Selected model and fitted configuration",
        "## MetaStart recommendation — predicted, not measured",
        "## Selected model cross-validation — training data only",
        "## Selected model held-out evaluation",
        "## Explainability",
        "## External AutoML comparisons",
        "## Limitations, bias, and resource constraints",
        "## Leakage integrity",
    ):
        assert heading in card
    model_section = card.split("## Selected model and fitted configuration", 1)[1].split(
        "## MetaStart recommendation", 1
    )[0]
    assert "Model type:** Supervised classification pipeline" in model_section
    assert "Algorithm:** SVM" in model_section
    assert "toy.csv" in card
    assert "current-upload-hash:target" in card
    assert "2026-10-23T12:00:00+00:00" in card
    assert "feature_a" in card
    assert "not causation" in card.lower()
    assert "demographic fairness is therefore not established" in card.lower()
    external_section = card.split("## External AutoML comparisons", 1)[1].split("## Limitations", 1)[0]
    assert "0.7900" in external_section
    assert "0.7500" in external_section
    assert "60" in external_section


def test_predicted_cv_and_heldout_metrics_are_distinct_and_match_records():
    card = build_model_card(_records(), _metadata())
    predicted_section = card.split("## MetaStart recommendation", 1)[1].split("## Selected model cross-validation", 1)[0]
    cv_section = card.split("## Selected model cross-validation", 1)[1].split("## Selected model held-out evaluation", 1)[0]
    heldout_section = card.split("## Selected model held-out evaluation", 1)[1].split("## Explainability", 1)[0]

    assert "0.6100" in predicted_section
    assert "not the selected model's measured" in predicted_section
    assert "0.8200" in cv_section
    assert "0.0200" in cv_section
    assert "0.8100" in heldout_section
    assert "0.8000" in heldout_section
    assert "0.6100" not in cv_section + heldout_section


def test_missing_metadata_and_unavailable_scores_are_disclosed():
    records = [record for record in _records() if record.stage not in {
        "recommendation", "cv", "held_out_test", "leakage", "explainability",
    }]
    card = build_model_card(records, {"selected_algorithm": "SVM"})
    assert "Not available" in card
    assert "Not measured" in card
    assert "Not separately versioned" in card
    assert "Not run in this session" in card


def test_leakage_score_is_excluded_and_warning_is_explicit():
    card = build_model_card(_records(), _metadata())
    leakage_section = card.split("## Leakage integrity", 1)[1]
    assert "INVALID FOR REAL-WORLD PERFORMANCE — DELIBERATE LEAKAGE DEMONSTRATION" in leakage_section
    assert "1.0000" not in card


def test_cards_change_with_dataset_and_current_run_metadata():
    records = _records()
    first = build_model_card(records, _metadata(dataset_name="first.csv", run_id="run-one"))
    second = build_model_card(records, _metadata(dataset_name="second.csv", run_id="run-two"))
    assert first != second
    assert "first.csv" in first and "second.csv" not in first
    assert "run-two" in second and "run-one" not in second


def test_model_change_does_not_reuse_another_algorithms_metrics_or_explanations():
    card = build_model_card(_records(), _metadata(selected_algorithm="Logistic Regression"))
    predicted_section = card.split("## MetaStart recommendation", 1)[1].split(
        "## Selected model cross-validation", 1
    )[0]
    cv_section = card.split("## Selected model cross-validation", 1)[1].split(
        "## Selected model held-out evaluation", 1
    )[0]
    explainability_section = card.split("## Explainability", 1)[1].split(
        "## Optional tuning experiments", 1
    )[0]

    assert "Predicted Macro-F1:** Not measured" in predicted_section
    assert "Measured CV Macro-F1 mean:** Not measured" in cv_section
    assert "Not run in this session" in explainability_section


def test_model_card_download_and_zip_bundle_preserve_existing_exports():
    card = build_model_card(_records(), _metadata())
    downloads = build_experiment_downloads(
        _records(), {"selected_algorithm": "SVM"}, model_card_text=card,
    )
    assert MODEL_CARD_MIME_TYPE == "text/markdown"
    assert downloads["model_card.md"].decode("utf-8") == card
    assert downloads["experiment_results.csv"]
    assert downloads["experiment_detailed_metrics.csv"]
    assert json.loads(downloads["experiment_report.json"])["experiments"]
    with zipfile.ZipFile(io.BytesIO(downloads["experiment_results_bundle.zip"])) as archive:
        assert archive.testzip() is None
        assert "model_card.md" in archive.namelist()
        assert archive.read("model_card.md").decode("utf-8") == card
        assert "experiment_results.csv" in archive.namelist()
        assert "experiment_report.json" in archive.namelist()

