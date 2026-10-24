"""Unified experiment-results tracking and comprehensive export layer.

This module is additive: it turns the existing candidate evaluation outputs and
the new advanced-experiment outcomes into one machine-readable experiment
ledger, flat leaderboard frames, detailed per-class metrics, JSON reports, and
downloadable CSV/JSON/ZIP bundles.

Stage identifiers used across the ledger:

- ``recommendation``: production meta-model predicted ranking.
- ``cv``: measured training-partition cross-validation.
- ``manual_baseline``: untuned scikit-learn baseline.
- ``grid_search`` / ``random_search`` / ``optuna``: hyperparameter searches.
- ``external_automl``: external AutoML framework leaderboard results.
- ``held_out_test``: final selected-model held-out evaluation.
- ``leakage``: deliberately leaky vs valid experiment.
- ``bias`` / ``explainability``: bias diagnostics and explanations.

Predicted MetaStart Macro-F1, measured CV Macro-F1, and held-out test metrics are
kept as separate fields so the report never silently conflates them.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

FLAT_COLUMNS = (
    "Experiment", "Stage", "Method", "Algorithm", "Configuration", "Status",
    "Primary Metric", "Primary Metric Value", "Accuracy", "Macro Precision",
    "Macro Recall", "Macro F1", "Weighted Precision", "Weighted Recall",
    "Weighted F1", "Micro F1", "ROC AUC", "PR AUC", "CV Folds", "CV Rows",
    "CV Macro F1", "CV Macro F1 Std", "CV Fit Seconds Mean", "Fit Seconds",
    "Test Scoring Seconds",
    "Test Rows", "Seed", "Trials Evaluated", "Best Parameters", "Notes",
    "Framework Metrics", "Leak", "Invalid for Real-World", "Warnings", "Failures",
)

DETAILED_COLUMNS = (
    "Experiment", "Stage", "Method", "Algorithm", "Configuration", "Class",
    "Class Count", "Class Proportion",
    "Precision", "Recall", "F1", "Support", "True Positive", "False Positive",
    "False Negative", "True Negative", "Feature", "Importance Mean", "Importance Std",
    "Score Change", "Sample Count", "Runtime Seconds", "Explanation Method", "Warnings", "Status",
    "Leak", "Invalid for Real-World",
)

PRIMARY_METRIC = "macro_f1"


@dataclass(frozen=True)
class ExperimentRecord:
    """One measured or attempted experiment outcome."""

    experiment_id: str
    method: str
    algorithm: str
    stage: str
    status: str = "completed"
    configuration: str = "untuned"
    primary_metric: str = PRIMARY_METRIC
    primary_metric_value: float | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    class_breakdown: tuple[dict[str, Any], ...] = ()
    cv_folds: int | None = None
    cv_rows: int | None = None
    cv_source_rows: int | None = None
    cv_macro_f1: float | None = None
    cv_macro_f1_std: float | None = None
    cv_fit_seconds_mean: float | None = None
    fit_seconds: float | None = None
    test_rows: int | None = None
    test_seconds: float | None = None
    seed: int | None = None
    trials_evaluated: int | None = None
    best_parameters: dict[str, Any] | None = None
    leaderboard_metrics: dict[str, Any] = field(default_factory=dict)
    leak: bool = False
    invalid_for_real_world: bool = False
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    @property
    def primary_value(self) -> float | None:
        value = self.metrics.get(self.primary_metric)
        if value is None and self.primary_metric_value is not None:
            value = self.primary_metric_value
        return None if value is None else float(value)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "method": self.method,
            "algorithm": self.algorithm,
            "stage": self.stage,
            "status": self.status,
            "configuration": self.configuration,
            "primary_metric": self.primary_metric,
            "primary_metric_value": self.primary_value,
            "metrics": dict(self.metrics),
            "class_breakdown": list(self.class_breakdown),
            "cv_folds": self.cv_folds,
            "cv_rows": self.cv_rows,
            "cv_source_rows": self.cv_source_rows,
            "cv_macro_f1": self.cv_macro_f1,
            "cv_macro_f1_std": self.cv_macro_f1_std,
            "cv_fit_seconds_mean": self.cv_fit_seconds_mean,
            "fit_seconds": self.fit_seconds,
            "test_rows": self.test_rows,
            "test_seconds": self.test_seconds,
            "seed": self.seed,
            "trials_evaluated": self.trials_evaluated,
            "best_parameters": (dict(self.best_parameters)
                                 if self.best_parameters is not None else None),
            "leaderboard_metrics": dict(self.leaderboard_metrics),
            "leak": self.leak,
            "invalid_for_real_world": self.invalid_for_real_world,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "failures": list(self.failures),
        }


def build_flat_leaderboard(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    """Build the consolidated flat experiment leaderboard."""
    rows = []
    for record in records:
        mapping = record.to_mapping()
        row: dict[str, Any] = {column: None for column in FLAT_COLUMNS}
        row.update({
            "Experiment": record.experiment_id,
            "Stage": record.stage,
            "Method": record.method,
            "Algorithm": record.algorithm,
            "Configuration": record.configuration,
            "Status": record.status,
            "Primary Metric": record.primary_metric,
            "Primary Metric Value": record.primary_value,
            "Accuracy": (record.metrics or {}).get("accuracy"),
            "Macro Precision": (record.metrics or {}).get("macro_precision"),
            "Macro Recall": (record.metrics or {}).get("macro_recall"),
            "Macro F1": (record.metrics or {}).get("macro_f1"),
            "Weighted Precision": (record.metrics or {}).get("weighted_precision"),
            "Weighted Recall": (record.metrics or {}).get("weighted_recall"),
            "Weighted F1": (record.metrics or {}).get("weighted_f1"),
            "Micro F1": (record.metrics or {}).get("micro_f1"),
            "ROC AUC": (record.metrics or {}).get("roc_auc"),
            "PR AUC": (record.metrics or {}).get("pr_auc"),
            "CV Folds": record.cv_folds,
            "CV Rows": record.cv_rows,
            "CV Macro F1": record.cv_macro_f1,
            "CV Macro F1 Std": record.cv_macro_f1_std,
            "CV Fit Seconds Mean": record.cv_fit_seconds_mean,
            "Fit Seconds": record.fit_seconds,
            "Test Scoring Seconds": record.test_seconds,
            "Test Rows": record.test_rows,
            "Seed": record.seed,
            "Trials Evaluated": record.trials_evaluated,
            "Best Parameters": json.dumps(record.best_parameters, sort_keys=True, default=str)
                                 if record.best_parameters is not None else None,
            "Notes": "; ".join(record.notes) if record.notes else None,
            "Framework Metrics": (json.dumps(record.leaderboard_metrics, sort_keys=True, default=str)
                                  if record.leaderboard_metrics else None),
            "Leak": record.leak,
            "Invalid for Real-World": record.invalid_for_real_world,
            "Warnings": "; ".join(record.warnings) if record.warnings else None,
            "Failures": "; ".join(record.failures) if record.failures else None,
        })
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(FLAT_COLUMNS))
    numeric_columns = (
        "Primary Metric Value", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
        "Weighted Precision", "Weighted Recall", "Weighted F1", "Micro F1", "ROC AUC", "PR AUC",
        "CV Macro F1", "CV Macro F1 Std", "CV Fit Seconds Mean", "Fit Seconds",
        "Test Scoring Seconds",
        "CV Folds", "CV Rows", "Test Rows", "Seed", "Trials Evaluated",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[list(FLAT_COLUMNS)]


def build_detailed_metrics(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    """Build the detailed per-class metrics frame with confusion-derived counts."""
    rows = []
    for record in records:
        record_row_count = len(rows)
        for entry in record.class_breakdown:
            rows.append({
                "Experiment": record.experiment_id,
                "Stage": record.stage,
                "Method": record.method,
                "Algorithm": record.algorithm,
                "Configuration": record.configuration,
                "Class": str(entry.get("Class")),
                "Class Count": entry.get("Class Count"),
                "Class Proportion": entry.get("Class Proportion"),
                "Precision": entry.get("Precision"),
                "Recall": entry.get("Recall"),
                "F1": entry.get("F1"),
                "Support": entry.get("Support"),
                "True Positive": entry.get("True Positive"),
                "False Positive": entry.get("False Positive"),
                "False Negative": entry.get("False Negative"),
                "True Negative": entry.get("True Negative"),
                "Feature": None, "Importance Mean": None, "Importance Std": None,
                "Score Change": None, "Sample Count": record.cv_rows,
                "Runtime Seconds": record.fit_seconds,
                "Explanation Method": record.method,
                "Warnings": "; ".join(record.warnings) if record.warnings else None,
                "Status": record.status,
                "Leak": record.leak,
                "Invalid for Real-World": record.invalid_for_real_world,
            })
        details = record.leaderboard_metrics or {}
        for feature in details.get("feature_importance", []):
            rows.append({
                "Experiment": record.experiment_id, "Stage": record.stage,
                "Method": record.method, "Algorithm": record.algorithm,
                "Configuration": record.configuration, "Class": None,
                "Class Count": None, "Class Proportion": None,
                "Precision": None, "Recall": None, "F1": None, "Support": None,
                "True Positive": None, "False Positive": None,
                "False Negative": None, "True Negative": None,
                "Feature": feature.get("Feature"),
                "Importance Mean": feature.get("Importance Mean"),
                "Importance Std": feature.get("Importance Std"), "Score Change": None,
                "Sample Count": details.get("sample_count"),
                "Runtime Seconds": details.get("runtime_seconds"),
                "Explanation Method": details.get("method", record.method),
                "Warnings": "; ".join(record.warnings) if record.warnings else None,
                "Status": record.status, "Leak": record.leak,
                "Invalid for Real-World": record.invalid_for_real_world,
            })
        if len(rows) == record_row_count and record.stage in {"bias", "explainability"}:
            rows.append({
                "Experiment": record.experiment_id, "Stage": record.stage,
                "Method": record.method, "Algorithm": record.algorithm,
                "Configuration": record.configuration, "Status": record.status,
                "Warnings": "; ".join(record.warnings) if record.warnings else None,
                "Leak": record.leak,
                "Invalid for Real-World": record.invalid_for_real_world,
            })
        for feature in details.get("local_feature_effects", []):
            rows.append({
                "Experiment": record.experiment_id, "Stage": record.stage,
                "Method": record.method, "Algorithm": record.algorithm,
                "Configuration": record.configuration, "Class": feature.get("Class explained"),
                "Class Count": None, "Class Proportion": None,
                "Precision": None, "Recall": None, "F1": None, "Support": None,
                "True Positive": None, "False Positive": None,
                "False Negative": None, "True Negative": None,
                "Feature": feature.get("Feature"), "Importance Mean": None,
                "Importance Std": None,
                "Score Change": feature.get("Score change after replacement"),
                "Sample Count": details.get("sample_count"),
                "Runtime Seconds": details.get("runtime_seconds"),
                "Explanation Method": details.get("method", record.method),
                "Warnings": "; ".join(record.warnings) if record.warnings else None,
                "Status": record.status, "Leak": record.leak,
                "Invalid for Real-World": record.invalid_for_real_world,
            })
    frame = pd.DataFrame(rows, columns=list(DETAILED_COLUMNS))
    for column in ("Class Count", "Class Proportion", "Precision", "Recall", "F1", "Support",
                   "Importance Mean", "Importance Std", "Score Change", "Sample Count",
                   "Runtime Seconds", "True Positive", "False Positive",
                   "False Negative", "True Negative"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_experiment_report(records: Iterable[ExperimentRecord],
                            metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the JSON experiment report document."""
    return {
        "experiment_report_version": 1,
        "generated_by": "MetaStart AutoML",
        "metadata": dict(metadata or {}),
        "experiments": [record.to_mapping() for record in records],
    }


def flat_leaderboard_csv(records: Iterable[ExperimentRecord]) -> str:
    return build_flat_leaderboard(records).to_csv(index=False, na_rep="N/A")


def detailed_metrics_csv(records: Iterable[ExperimentRecord]) -> str:
    return build_detailed_metrics(records).to_csv(index=False, na_rep="N/A")


def experiment_report_json(records: Iterable[ExperimentRecord],
                           metadata: Mapping[str, Any] | None = None) -> str:
    return json.dumps(build_experiment_report(records, metadata), indent=2, default=str)


def build_experiment_downloads(records: Iterable[ExperimentRecord],
                               metadata: Mapping[str, Any] | None = None,
                               *, model_card_text: str | None = None) -> dict[str, bytes]:
    """Return the comprehensive report files and an optional ZIP bundle."""
    records = list(records)
    flat_csv = flat_leaderboard_csv(records).encode("utf-8")
    detailed_csv = detailed_metrics_csv(records).encode("utf-8")
    report_json = experiment_report_json(records, metadata).encode("utf-8")
    files = {
        "experiment_results.csv": flat_csv,
        "experiment_detailed_metrics.csv": detailed_csv,
        "experiment_report.json": report_json,
    }
    if model_card_text:
        files["model_card.md"] = model_card_text.encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        archive.writestr("experiment_bundle_manifest.json", json.dumps({
            "bundle_version": 1,
            "files": sorted(files),
            "experiment_count": len(records),
        }, indent=2).encode("utf-8"))
    downloads = dict(files)
    downloads["experiment_results_bundle.zip"] = buffer.getvalue()
    return downloads


def build_recommendation_records(
        algorithms: Sequence[str],
        ranking: pd.DataFrame,
        selection_results: pd.DataFrame,
        failures: Sequence[tuple[str, str, str]],
        *,
        selected_algorithm: str | None = None,
        full_training_seconds: float | None = None,
        test_result: pd.DataFrame | None = None,
        class_metrics: pd.DataFrame | None = None,
        cv_source_rows: int = 0,
        cv_rows: int = 0,
        cv_n_splits: int = 0,
        test_rows: int = 0,
        dataset_label: str = "uploaded",
        seed: int = 42,
) -> list[ExperimentRecord]:
    """Convert the existing five-algorithm evaluation outputs into ledger records.

    This is additive: it reuses the already measured scores without changing any
    selection behavior.
    """
    records: list[ExperimentRecord] = []
    predicted_by_algorithm = (
        ranking.set_index("algorithm")["predicted_macro_f1"].to_dict() if not ranking.empty else {}
    )
    cv_by_algorithm = (
        selection_results.set_index("Algorithm").to_dict("index")
        if not selection_results.empty and "Algorithm" in selection_results else {}
    )
    failures_by_algorithm: dict[str, list[tuple[str, str]]] = {}
    classes_by_algorithm: dict[str, tuple[dict[str, Any], ...]] = {}
    if class_metrics is not None and not class_metrics.empty:
        for algorithm, rows in class_metrics.groupby("Algorithm", sort=False):
            classes_by_algorithm[str(algorithm)] = tuple({
                "Class": row["Class"], "Precision": row["Precision"],
                "Recall": row["Recall"], "F1": row["F1"], "Support": row["Support"],
            } for _, row in rows.iterrows())
    for name, phase, message in failures:
        failures_by_algorithm.setdefault(name, []).append((phase, message))
    for algorithm in algorithms:
        predicted = predicted_by_algorithm.get(algorithm)
        cv = cv_by_algorithm.get(algorithm)
        algorithm_failures = failures_by_algorithm.get(algorithm, [])
        cross_failure = next((message for phase, message in algorithm_failures if phase == "cross-validation"), None)
        fit_failure = next((message for phase, message in algorithm_failures if phase == "final fit"), None)
        resource_failure = next((message for phase, message in algorithm_failures if phase == "resource limit"), None)
        heldout_failure = next((message for phase, message in algorithm_failures if phase == "held-out evaluation"), None)
        if cross_failure:
            status = f"cv_failed:{cross_failure}"
        elif cv is None and fit_failure:
            status = f"fit_failed:{fit_failure}"
        elif resource_failure:
            status = f"resource_skipped:{resource_failure}"
        elif fit_failure:
            status = f"final_fit_failed:{fit_failure}"
        elif heldout_failure and algorithm == selected_algorithm:
            status = f"held_out_failed:{heldout_failure}"
        elif cv is None:
            status = "not_run"
        else:
            status = "completed"
        metrics: dict[str, float | None] = {"macro_f1": float(predicted) if predicted is not None else None}
        record = ExperimentRecord(
            experiment_id=f"recommendation::{algorithm}",
            method="meta_start_production",
            algorithm=algorithm,
            stage="recommendation",
            status=status,
            configuration="production_model_A",
            primary_metric_value=float(predicted) if predicted is not None else None,
            metrics=metrics,
            cv_folds=cv_n_splits or None,
            cv_rows=cv_rows or None,
            cv_source_rows=cv_source_rows or None,
            cv_macro_f1=float(cv["CV Macro-F1 Mean"]) if cv is not None else None,
            cv_macro_f1_std=float(cv["CV Macro-F1 Std"]) if cv is not None else None,
            cv_fit_seconds_mean=float(cv["CV Fit Seconds Mean"]) if cv is not None else None,
            fit_seconds=full_training_seconds if algorithm == selected_algorithm else None,
            test_rows=int(test_rows) if algorithm == selected_algorithm else None,
            seed=seed,
            notes=(f"Dataset: {dataset_label}. Predicted MetaStart Macro-F1 is an estimate; "
                   f"measured CV Macro-F1 is {float(cv['CV Macro-F1 Mean']):.3f}." if cv is not None else
                   f"Dataset: {dataset_label}. Predicted MetaStart Macro-F1 is an estimate."),
            warnings=("MetaStart predicted Macro-F1 is not a measured CV score." if predicted is not None else ()),
            failures=tuple(f"{phase}:{message}" for phase, message in algorithm_failures),
        )
        records.append(record)
        if cv is not None:
            records.append(ExperimentRecord(
                experiment_id=f"cv::{algorithm}", method="meta_start_candidate_evaluation",
                algorithm=algorithm, stage="cv", status="completed",
                configuration="training_partition_stratified_oof",
                primary_metric_value=float(cv["CV Macro-F1 Mean"]),
                metrics={"macro_f1": float(cv["CV Macro-F1 Mean"])},
                class_breakdown=classes_by_algorithm.get(algorithm, ()),
                cv_folds=cv_n_splits or None, cv_rows=cv_rows or None,
                cv_source_rows=cv_source_rows or None,
                cv_macro_f1=float(cv["CV Macro-F1 Mean"]),
                cv_macro_f1_std=float(cv["CV Macro-F1 Std"]),
                cv_fit_seconds_mean=float(cv["CV Fit Seconds Mean"]), seed=seed,
                notes=("Out-of-fold class metrics use training-partition CV predictions.",),
            ))
    if selected_algorithm is not None and not test_result.empty:
        test_row = test_result.iloc[0]
        heldout_metrics = {
            "accuracy": float(test_row["Held-out Test Accuracy"]),
            "macro_f1": float(test_row["Held-out Test Macro-F1"]),
        }
        records.append(ExperimentRecord(
            experiment_id=f"held_out_test::{selected_algorithm}",
            method="meta_start_production",
            algorithm=selected_algorithm,
            stage="held_out_test",
            status="completed",
            configuration="production_model_A",
            primary_metric_value=float(test_row["Held-out Test Macro-F1"]),
            metrics=heldout_metrics,
            fit_seconds=float(test_row["Training seconds"]) if pd.notna(test_row.get("Training seconds")) else None,
            test_rows=int(test_rows),
            seed=seed,
            notes=("Final selected-model held-out test metrics. Test labels were not used to select."),
        ))
    return records
