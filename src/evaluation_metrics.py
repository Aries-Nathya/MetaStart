"""Reusable measured classification metrics for experiments and reports.

Every metric is computed from real predictions. Values that cannot be computed
(for example ROC-AUC when a model has no predictable probability) are ``None``
and are exported as explicit missing values rather than being replaced by zero.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "micro_f1",
    "roc_auc",
    "pr_auc",
)


def compute_measured_metrics(y_true, y_pred, *, estimator=None, X=None,
                             labels: Sequence | None = None) -> dict[str, float | None]:
    """Compute the applicable classification metrics from genuine predictions.

    Macro-F1 is the primary metric; accuracy and the remaining supported metrics
    are reported alongside it. Macro and weighted averages use the one-vs-rest
    convention over the observed class labels; micro F1 aggregates over all classes.
    ROC-AUC and PR-AUC use the one-vs-rest multiclass convention and are ``None``
    when probabilities are unavailable or the calculation fails.
    """
    metrics: dict[str, float | None] = {name: None for name in METRIC_NAMES}
    true = pd.Series(y_true).astype(str).to_numpy()
    pred = pd.Series(y_pred).astype(str).to_numpy()
    metrics["accuracy"] = float(accuracy_score(true, pred))
    metrics["micro_f1"] = float(f1_score(true, pred, average="micro", zero_division=0))
    effective_labels = sorted(set(true.tolist()) | set(pred.tolist()))
    if labels is not None:
        effective_labels = sorted({str(label) for label in np.asarray(list(labels)).tolist()})
    if effective_labels:
        precision, recall, f1, support = precision_recall_fscore_support(
            true, pred, labels=effective_labels, average=None, zero_division=0)
        support_array = np.asarray(support, dtype=float)
        total_support = float(support_array.sum())
        metrics["macro_precision"] = float(np.mean(precision)) if len(precision) else None
        metrics["macro_recall"] = float(np.mean(recall)) if len(recall) else None
        metrics["macro_f1"] = float(np.mean(f1)) if len(f1) else None
        if total_support > 0:
            metrics["weighted_precision"] = float(np.dot(support_array, precision) / total_support)
            metrics["weighted_recall"] = float(np.dot(support_array, recall) / total_support)
            metrics["weighted_f1"] = float(np.dot(support_array, f1) / total_support)
    if estimator is not None and X is not None and hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=float)
        if proba.ndim == 2 and proba.shape[1] > 1 and len(effective_labels) > 1:
            model_classes = getattr(estimator, "classes_", None)
            if model_classes is not None and len(model_classes) == proba.shape[1]:
                class_columns = {str(label): index for index, label in enumerate(model_classes)}
                if all(label in class_columns for label in effective_labels):
                    proba = proba[:, [class_columns[label] for label in effective_labels]]
            multilabel = np.zeros((len(true), len(effective_labels)), dtype=int)
            for index, label in enumerate(effective_labels):
                multilabel[:, index] = (true == label).astype(int)
            try:
                metrics["roc_auc"] = float(roc_auc_score(multilabel, proba, average="macro"))
                metrics["pr_auc"] = float(average_precision_score(multilabel, proba, average="macro"))
            except Exception:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
        elif proba.ndim == 2 and proba.shape[1] == 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(true, proba[:, 0]))
                metrics["pr_auc"] = float(average_precision_score(true, proba[:, 0]))
            except Exception:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
    return metrics


def class_breakdown_rows(y_true, y_pred, labels) -> list[dict[str, Any]]:
    """Per-class precision, recall, F1, support, and confusion-derived counts."""
    true = pd.Series(y_true); pred = pd.Series(y_pred)
    ordered = sorted({str(value) for value in true.unique().tolist()}
                     | {str(value) for value in pred.unique().tolist()})
    if labels is not None:
        ordered = sorted({str(label) for label in np.asarray(list(labels)).tolist()})
    precision, recall, f1, support = precision_recall_fscore_support(
        true.astype(str).to_numpy(), pred.astype(str).to_numpy(), labels=ordered, average=None, zero_division=0)
    pred_values = pred.astype(str).to_numpy()
    rows = []
    for index, label in enumerate(ordered):
        label_key = str(label)
        actual_mask = true.astype(str).to_numpy() == label_key
        predicted_mask = pred_values == label_key
        tp = int(np.sum(actual_mask & predicted_mask))
        fp = int(np.sum(predicted_mask & ~actual_mask))
        fn = int(np.sum(actual_mask & ~predicted_mask))
        tn = int(np.sum((~actual_mask) & (~predicted_mask)))
        rows.append({
            "Class": label_key,
            "Precision": float(precision[index]),
            "Recall": float(recall[index]),
            "F1": float(f1[index]),
            "Support": int(support[index]),
            "True Positive": tp,
            "False Positive": fp,
            "False Negative": fn,
            "True Negative": tn,
        })
    return rows


def confusion_matrix_rows(y_true, y_pred, labels) -> list[dict[str, Any]]:
    """Return every observed true/predicted class pair with its count."""
    true = pd.Series(y_true).astype(str).to_numpy()
    pred = pd.Series(y_pred).astype(str).to_numpy()
    ordered = sorted({str(value) for value in pd.unique(true)} | {str(value) for value in pd.unique(pred)})
    matrix = confusion_matrix(true, pred, labels=ordered)
    rows = []
    for actual_index, actual in enumerate(ordered):
        for predicted_index, predicted in enumerate(ordered):
            rows.append({
                "True Class": str(actual),
                "Predicted Class": str(predicted),
                "Count": int(matrix[actual_index, predicted_index]),
            })
    return rows
