"""Repeated stratified CV for descriptive checks on built-in demo datasets.

This measures candidate-model stability on these datasets. Since these datasets
also appear in the benchmark corpus, it is not an independent estimate of
MetaStart's generalization to entirely new dataset identities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.benchmark import ROOT, build_pipeline, get_datasets, get_models


TARGET_DATASETS = {"iris", "wine", "breast_cancer"}


def run_repeated_validation(results_dir: str | Path = ROOT / "results",
                            n_splits: int = 5, n_repeats: int = 5):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, X, y in get_datasets():
        if dataset not in TARGET_DATASETS:
            continue
        min_class = int(y.value_counts().min())
        splits = min(n_splits, min_class)
        cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=n_repeats, random_state=42)
        for algorithm, estimator in get_models().items():
            scores = cross_validate(
                build_pipeline(estimator, X), X, y, cv=cv, n_jobs=1,
                scoring={
                    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
                    "accuracy": "accuracy",
                },
                return_train_score=False,
            )
            for fold, (macro_f1, accuracy, fit_seconds) in enumerate(zip(
                    scores["test_macro_f1"], scores["test_accuracy"], scores["fit_time"])):
                records.append({
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "repeat": fold // splits + 1,
                    "fold": fold % splits + 1,
                    "macro_f1": float(macro_f1),
                    "accuracy": float(accuracy),
                    "training_seconds": float(fit_seconds),
                })
            print(f"{dataset} | {algorithm} | macro-F1={scores['test_macro_f1'].mean():.3f} "
                  f"(SD {scores['test_macro_f1'].std(ddof=1):.3f})")
    folds = pd.DataFrame(records)
    folds.to_csv(results_dir / "repeated_cv_folds.csv", index=False)
    summary = folds.groupby(["dataset", "algorithm"]).agg(
        n_splits=("macro_f1", "size"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        macro_f1_min=("macro_f1", "min"),
        macro_f1_max=("macro_f1", "max"),
        accuracy_mean=("accuracy", "mean"),
        training_seconds_mean=("training_seconds", "mean"),
    ).reset_index()
    summary.to_csv(results_dir / "repeated_cv_summary.csv", index=False)
    return folds, summary


if __name__ == "__main__":
    run_repeated_validation()
