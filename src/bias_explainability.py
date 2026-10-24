"""Opt-in, training-partition-only bias diagnostics and model explanations."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.evaluation_metrics import class_breakdown_rows, confusion_matrix_rows
from src.experiment_results import ExperimentRecord

IMBALANCE_RATIO_THRESHOLD = 3.0
DEFAULT_CV_SAMPLE_LIMIT = 3_000
DEFAULT_EXPLANATION_SAMPLE_LIMIT = 200
DEFAULT_LOCAL_FEATURE_LIMIT = 20


def class_balance_diagnostics(y: pd.Series) -> tuple[pd.DataFrame, float, bool]:
    """Return counts/proportions and flag a max/min class ratio of at least 3:1."""
    labels = pd.Series(y).reset_index(drop=True)
    counts = labels.value_counts(dropna=False, sort=False)
    if counts.empty:
        return pd.DataFrame(columns=["Class", "Count", "Proportion"]), float("nan"), False
    ratio = float(counts.max() / counts.min()) if counts.min() else float("inf")
    table = pd.DataFrame({
        "Class": counts.index.astype(str),
        "Count": counts.to_numpy(dtype=int),
        "Proportion": counts.to_numpy(dtype=float) / len(labels),
    })
    return table, ratio, ratio >= IMBALANCE_RATIO_THRESHOLD


def _bounded_stratified_sample(X: pd.DataFrame, y: pd.Series, limit: int, seed: int):
    if len(X) <= limit:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    if limit < y.nunique():
        raise ValueError("sample limit is smaller than the number of classes")
    indices, _ = train_test_split(
        np.arange(len(y)), train_size=limit, random_state=seed, stratify=y,
    )
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


def _prediction_strength(estimator, row: pd.DataFrame, class_label: Any) -> tuple[float, str]:
    if hasattr(estimator, "predict_proba"):
        classes = list(estimator.classes_)
        if class_label in classes:
            return float(estimator.predict_proba(row)[0, classes.index(class_label)]), "class probability"
    if hasattr(estimator, "decision_function"):
        classes = list(estimator.classes_)
        decision = np.asarray(estimator.decision_function(row), dtype=float)
        if decision.ndim == 1 and len(classes) == 2:
            direction = 1.0 if class_label == classes[1] else -1.0
            return float(direction * decision[0]), "signed decision-function score"
        if decision.ndim == 2 and decision.shape[1] == len(classes) and class_label in classes:
            return float(decision[0, classes.index(class_label)]), "class decision-function score"
    return float(estimator.predict(row)[0] == class_label), "predicted-class indicator"


def run_bias_explainability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    selected_algorithm: str,
    fitted_pipeline: Any,
    model_factory,
    pipeline_builder,
    experiment_id: str = "bias_explainability",
    cv_sample_limit: int = DEFAULT_CV_SAMPLE_LIMIT,
    explanation_sample_limit: int = DEFAULT_EXPLANATION_SAMPLE_LIMIT,
    local_feature_limit: int = DEFAULT_LOCAL_FEATURE_LIMIT,
) -> dict[str, Any]:
    """Compute OOF class diagnostics and bounded permutation/local explanations.

    The API intentionally receives only the training partition and its already
    selected fitted pipeline; held-out rows or labels cannot enter these analyses.
    """
    X = pd.DataFrame(X_train).reset_index(drop=True)
    y = pd.Series(y_train).reset_index(drop=True)
    if len(X) != len(y) or X.empty:
        raise ValueError("training features and labels must be non-empty and aligned")
    if selected_algorithm not in model_factory():
        raise ValueError(f"unknown selected algorithm: {selected_algorithm}")

    balance, imbalance_ratio, imbalance_flag = class_balance_diagnostics(y)
    X_cv, y_cv = _bounded_stratified_sample(X, y, cv_sample_limit, seed=42)
    folds = min(3, int(y_cv.value_counts().min()))
    if folds < 2:
        raise ValueError("at least two examples per class are required for out-of-fold diagnostics")
    cv_pipeline = pipeline_builder(model_factory()[selected_algorithm], X_cv)
    cv_started = time.perf_counter()
    oof = cross_val_predict(
        cv_pipeline, X_cv, y_cv,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
        method="predict", n_jobs=1,
    )
    cv_seconds = time.perf_counter() - cv_started
    labels = sorted(pd.unique(y_cv).tolist(), key=str)
    class_rows = class_breakdown_rows(y_cv, oof, labels)
    balance_by_label = balance.set_index("Class").to_dict(orient="index")
    for row in class_rows:
        class_balance = balance_by_label.get(str(row["Class"]), {})
        row["Class Count"] = class_balance.get("Count")
        row["Class Proportion"] = class_balance.get("Proportion")
    class_metrics = pd.DataFrame(class_rows)
    class_metrics.insert(0, "Class", class_metrics.pop("Class"))
    class_metrics["Algorithm"] = selected_algorithm
    class_metrics["Lower F1 than class mean"] = class_metrics["F1"] < class_metrics["F1"].mean()
    confusion = pd.DataFrame(confusion_matrix_rows(y_cv, oof, labels))
    errors = confusion.loc[confusion["True Class"] != confusion["Predicted Class"]].copy()
    errors = errors.rename(columns={"True Class": "Actual Class", "Predicted Class": "Predicted As"})
    diagnostics_record = ExperimentRecord(
        experiment_id=f"{experiment_id}::class_diagnostics",
        method="stratified_out_of_fold_class_diagnostics",
        algorithm=selected_algorithm,
        stage="bias",
        status="completed",
        configuration="training_partition_stratified_oof",
        class_breakdown=tuple(class_rows),
        cv_folds=folds,
        cv_rows=len(y_cv),
        cv_source_rows=len(y),
        fit_seconds=cv_seconds,
        test_rows=len(y_cv),
        seed=42,
        leaderboard_metrics={
            "imbalance_ratio": imbalance_ratio,
            "imbalance_threshold": IMBALANCE_RATIO_THRESHOLD,
            "imbalance_flagged": imbalance_flag,
            "class_counts": balance.to_dict(orient="records"),
            "confusion_matrix": confusion.to_dict(orient="records"),
            "error_counts": errors.to_dict(orient="records"),
            "lower_f1_classes": class_metrics.loc[
                class_metrics["Lower F1 than class mean"], "Class"
            ].astype(str).tolist(),
            "fairness_limitations": (
                "Class labels are outcome categories, not assumed protected groups. "
                "No sensitive attributes or appropriate subgroup labels were supplied; "
                "demographic or group fairness cannot be established."
            ),
        },
        notes=(
            f"Class imbalance is flagged at largest/smallest count ratio >= {IMBALANCE_RATIO_THRESHOLD:g}:1.",
            "Class performance and confusion counts are stratified out-of-fold estimates on training rows only.",
            "Class-wise metrics do not establish fairness; target classes are not treated as protected groups.",
        ),
    )

    explain_X, explain_y = _bounded_stratified_sample(
        X, y, explanation_sample_limit, seed=43,
    )
    explanation_records: list[ExperimentRecord] = []
    warnings: list[str] = []
    importance = pd.DataFrame(columns=["Feature", "Importance Mean", "Importance Std"])
    explanation_status = "completed"
    importance_seconds = 0.0
    try:
        started = time.perf_counter()
        result = permutation_importance(
            fitted_pipeline, explain_X, explain_y, scoring="f1_macro",
            n_repeats=3, random_state=42, n_jobs=1,
        )
        importance_seconds = time.perf_counter() - started
        importance = pd.DataFrame({
            "Feature": X.columns.astype(str),
            "Importance Mean": result.importances_mean,
            "Importance Std": result.importances_std,
        }).sort_values("Importance Mean", ascending=False, kind="stable").reset_index(drop=True)
        importance_record = ExperimentRecord(
            experiment_id=f"{experiment_id}::global_importance",
            method="permutation_importance",
            algorithm=selected_algorithm,
            stage="explainability",
            status="completed",
            configuration="training_partition_sample_macro_f1_3_repeats",
            fit_seconds=importance_seconds,
            test_rows=len(explain_y),
            seed=42,
            leaderboard_metrics={
                "method": "permutation importance; Macro-F1 decrease after feature permutation",
                "sample_count": len(explain_y),
                "repeats": 3,
                "runtime_seconds": importance_seconds,
                "feature_importance": importance.to_dict(orient="records"),
                "interpretation": (
                    "Feature contribution/association in this fitted model, not causation. "
                    "Importance was measured on training rows used to fit the model and may overstate generalization."
                ),
            },
            notes=("Computed on a deterministic, bounded sample from the training partition used to fit the model; this may overstate generalization.",
                   "Permutation importance reflects model association/contribution, not causation."),
        )
        explanation_records.append(importance_record)
    except Exception as exc:
        explanation_status = "skipped_or_failed"
        warnings.append(f"Permutation importance failed: {type(exc).__name__}: {exc}")
        explanation_records.append(ExperimentRecord(
            experiment_id=f"{experiment_id}::global_importance",
            method="permutation_importance",
            algorithm=selected_algorithm,
            stage="explainability",
            status=explanation_status,
            configuration="training_partition_sample_macro_f1_3_repeats",
            test_rows=len(explain_y),
            seed=42,
            warnings=(warnings[-1],),
            notes=("No importance result was produced.",),
        ))

    local = pd.DataFrame(columns=["Feature", "Class explained", "Score change after replacement", "Score type"])
    try:
        local_started = time.perf_counter()
        row = explain_X.iloc[[0]].copy()
        predicted_class = fitted_pipeline.predict(row)[0]
        reference = X
        replacement = {}
        for column in X.columns:
            series = reference[column]
            replacement[column] = (series.median() if pd.api.types.is_numeric_dtype(series)
                                   else series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty
                                   else np.nan)
        base_score, score_type = _prediction_strength(fitted_pipeline, row, predicted_class)
        local_rows = []
        for column in list(X.columns)[:local_feature_limit]:
            altered = row.copy()
            altered.loc[:, column] = replacement[column]
            altered_score, _ = _prediction_strength(fitted_pipeline, altered, predicted_class)
            local_rows.append({
                "Feature": str(column), "Class explained": str(predicted_class),
                "Score change after replacement": base_score - altered_score,
                "Score type": score_type,
            })
        local = pd.DataFrame(local_rows).assign(
            **{"Absolute change": lambda frame: frame["Score change after replacement"].abs()}
        ).sort_values("Absolute change", ascending=False, kind="stable").drop(columns="Absolute change")
        local_seconds = time.perf_counter() - local_started
        local_record = ExperimentRecord(
            experiment_id=f"{experiment_id}::local_explanation",
            method="single_feature_ablation_sensitivity",
            algorithm=selected_algorithm,
            stage="explainability",
            status="completed",
            configuration=f"one_training_row_max_{local_feature_limit}_features",
            fit_seconds=local_seconds,
            test_rows=1,
            seed=42,
            leaderboard_metrics={
                "method": "single-feature replacement with training median/mode reference",
                "sample_count": 1,
                "features_explained": len(local),
                "runtime_seconds": local_seconds,
                "predicted_class": str(predicted_class),
                "score_type": score_type,
                "local_feature_effects": local.to_dict(orient="records"),
                "interpretation": "Sensitivity of this prediction to one-feature replacement; not causal attribution.",
            },
            notes=("Individual prediction explanation uses one training-partition row and median/mode replacements.",),
        )
    except Exception as exc:
        warnings.append(f"Local explanation failed: {type(exc).__name__}: {exc}")
        local_record = ExperimentRecord(
            experiment_id=f"{experiment_id}::local_explanation",
            method="single_feature_ablation_sensitivity",
            algorithm=selected_algorithm,
            stage="explainability",
            status="skipped_or_failed",
            configuration=f"one_training_row_max_{local_feature_limit}_features",
            warnings=(warnings[-1],),
            notes=("No local explanation result was produced.",),
        )
    explanation_records.append(local_record)
    return {
        "balance": balance,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_flagged": imbalance_flag,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion,
        "error_counts": errors,
        "importance": importance,
        "local_explanation": local,
        "records": [diagnostics_record, *explanation_records],
        "cv_runtime_seconds": cv_seconds,
        "importance_runtime_seconds": importance_seconds,
        "importance_sample_count": len(explain_y),
        "explanation_status": explanation_status,
        "warnings": warnings,
        "fairness_limitation": diagnostics_record.leaderboard_metrics["fairness_limitations"],
    }
