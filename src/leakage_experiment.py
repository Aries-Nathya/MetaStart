"""Opt-in, isolated target-leakage demonstration.

This experiment uses only sklearn's bundled Breast Cancer Wisconsin dataset. It
does not accept uploaded data, return a fitted model, or interact with MetaStart's
recommendation and model-selection functions.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.benchmark import build_pipeline
from src.evaluation_metrics import class_breakdown_rows, compute_measured_metrics
from src.experiment_results import ExperimentRecord

LEAKAGE_DEMONSTRATION_LABEL = (
    "INVALID FOR REAL-WORLD PERFORMANCE — DELIBERATE LEAKAGE DEMONSTRATION"
)
LEAKAGE_DATASET = "Breast Cancer Wisconsin Diagnostic (scikit-learn bundled)"
LEAKAGE_SPLIT_SEED = 42
LEAKAGE_TEST_SIZE = 0.25
LEAKAGE_TARGET_FEATURE = "__deliberate_target_copy__"
LEAKAGE_METRIC = "Macro-F1"


def run_leakage_demonstration(
        *, experiment_id: str = "leakage_demo_breast_cancer_seed42",
) -> tuple[list[ExperimentRecord], pd.DataFrame]:
    """Compare a valid baseline with an intentionally target-contaminated model.

    The same fixed stratified split and fixed Decision Tree configuration are used
    for both runs. In the invalid run, the target is copied into a feature before
    evaluation, including for held-out rows. This deliberately violates test
    isolation so the score demonstrates leakage and must never be treated as a
    real-world performance estimate.
    """
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.reset_index(drop=True).copy()
    features = features.reset_index(drop=True)
    train_indices, test_indices = train_test_split(
        range(len(target)), test_size=LEAKAGE_TEST_SIZE,
        random_state=LEAKAGE_SPLIT_SEED, stratify=target,
    )
    X_train = features.iloc[train_indices].reset_index(drop=True)
    X_test = features.iloc[test_indices].reset_index(drop=True)
    y_train = target.iloc[train_indices].reset_index(drop=True)
    y_test = target.iloc[test_indices].reset_index(drop=True)
    class_labels = sorted(target.astype(str).unique().tolist())

    model_parameters = {"max_depth": 4, "class_weight": "balanced", "random_state": 42}
    records: list[ExperimentRecord] = []
    summary_rows: list[dict[str, Any]] = []

    for is_leaky in (False, True):
        label = "Leaky (target copied into feature)" if is_leaky else "Valid baseline"
        if is_leaky:
            train_features = X_train.copy()
            test_features = X_test.copy()
            # This is the deliberate violation: held-out labels are explicitly
            # encoded into held-out model inputs before prediction.
            train_features[LEAKAGE_TARGET_FEATURE] = y_train.to_numpy()
            test_features[LEAKAGE_TARGET_FEATURE] = y_test.to_numpy()
        else:
            train_features, test_features = X_train, X_test

        estimator = DecisionTreeClassifier(**model_parameters)
        pipeline = build_pipeline(estimator, train_features)
        fit_started = time.perf_counter()
        pipeline.fit(train_features, y_train)
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = pipeline.predict(test_features)
        test_seconds = time.perf_counter() - predict_started
        metrics = compute_measured_metrics(
            y_test, predictions, estimator=pipeline, X=test_features, labels=class_labels,
        )
        breakdown = tuple(class_breakdown_rows(y_test, predictions, class_labels))
        status = "invalid_for_real_world" if is_leaky else "completed"
        notes = [
            LEAKAGE_DEMONSTRATION_LABEL,
            ("The target value was copied directly into a feature for both training and held-out rows. "
             "Held-out labels therefore entered the model input; this score is invalid." if is_leaky
             else "Original features only; preprocessing is fitted on the training partition."),
            ("Macro-F1 is the unweighted mean of per-class F1 on the fixed held-out partition."),
        ]
        record = ExperimentRecord(
            experiment_id=f"{experiment_id}::{ 'leaky' if is_leaky else 'valid' }",
            method="deliberate_target_leakage_demo" if is_leaky else "leakage_safe_baseline",
            algorithm=f"Decision Tree — {label}",
            stage="leakage",
            status=status,
            configuration="fixed_depth_4_decision_tree",
            primary_metric="macro_f1",
            primary_metric_value=metrics["macro_f1"],
            metrics=metrics,
            class_breakdown=breakdown,
            fit_seconds=fit_seconds,
            test_seconds=test_seconds,
            test_rows=len(y_test),
            seed=LEAKAGE_SPLIT_SEED,
            best_parameters=model_parameters,
            leaderboard_metrics={
                "dataset": LEAKAGE_DATASET,
                "split": "stratified train/test",
                "test_size": LEAKAGE_TEST_SIZE,
                "split_seed": LEAKAGE_SPLIT_SEED,
                "training_rows": len(y_train),
                "held_out_rows": len(y_test),
                "primary_metric": LEAKAGE_METRIC,
                "metric_definition": "Unweighted mean of per-class F1 on the held-out partition.",
                "target_feature": LEAKAGE_TARGET_FEATURE if is_leaky else None,
                "held_out_target_used_as_feature": is_leaky,
                "invalid_reason": (
                    "The test target is directly encoded in the model input before prediction."
                    if is_leaky else None
                ),
            },
            leak=is_leaky,
            invalid_for_real_world=is_leaky,
            notes=tuple(notes),
        )
        records.append(record)
        summary_rows.append({
            "Result": label,
            "Status": status,
            "Validity": LEAKAGE_DEMONSTRATION_LABEL if is_leaky else "Leakage-safe baseline",
            "Dataset": LEAKAGE_DATASET,
            "Split": "Stratified 75/25; seed 42",
            "Training rows": len(y_train),
            "Held-out rows": len(y_test),
            "Macro-F1": metrics["macro_f1"],
            "Accuracy": metrics["accuracy"],
            "Macro precision": metrics["macro_precision"],
            "Macro recall": metrics["macro_recall"],
            "Fit seconds": fit_seconds,
            "Test scoring seconds": test_seconds,
        })

    return records, pd.DataFrame(summary_rows)
