"""Reproducible classification benchmark and meta-dataset builder."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (load_breast_cancer, load_digits, load_iris,
                              load_wine, make_classification)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.dataset_profiler import profile_dataset

ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "k-Nearest Neighbors"]


def get_datasets():
    """Built-in datasets plus deterministic synthetic datasets for variety."""
    loaders = [load_breast_cancer, load_iris, load_wine, load_digits]
    for loader in loaders:
        bunch = loader()
        yield loader.__name__.replace("load_", ""), pd.DataFrame(bunch.data, columns=bunch.feature_names), pd.Series(bunch.target)
    for seed, (n_samples, n_features, n_classes, weights) in enumerate([
        (450, 8, 2, None), (700, 18, 3, [0.65, 0.25, 0.10]),
        (900, 30, 2, [0.85, 0.15]),
    ], start=42):
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=max(2, n_features // 2),
                                   n_redundant=n_features // 5, n_classes=n_classes,
                                   weights=weights, random_state=seed)
        yield f"synthetic_{seed}", pd.DataFrame(X), pd.Series(y)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1),
        "SVM": SVC(class_weight="balanced"),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }


def build_pipeline(model, X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []
    if numeric_cols:
        numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        transformers.append(("numeric", numeric, numeric_cols))
    if categorical_cols:
        categorical = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                    OneHotEncoder(handle_unknown="ignore"))
        transformers.append(("categorical", categorical, categorical_cols))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return make_pipeline(preprocessor, model)


def run_benchmarks(output_dir: Path | str = ROOT):
    output_dir = Path(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    records = []
    for dataset_name, X, y in get_datasets():
        min_class = int(y.value_counts().min())
        if min_class < 2:
            print(f"Skipped {dataset_name}: a class has fewer than two rows")
            continue
        cv = StratifiedKFold(n_splits=min(3, min_class), shuffle=True, random_state=42)
        features = profile_dataset(X, y)
        for name, model in get_models().items():
            pipeline = build_pipeline(model, X)
            start = time.perf_counter()
            try:
                scores = cross_validate(pipeline, X, y, cv=cv,
                                        scoring=make_scorer(f1_score, average="macro", zero_division=0),
                                        n_jobs=1, error_score="raise")
                records.append({"dataset": dataset_name, **features, "algorithm": name,
                                "macro_f1": float(scores["test_score"].mean()),
                                "training_seconds": float(scores["fit_time"].mean()),
                                "wall_seconds": float(time.perf_counter() - start)})
                print(f"{dataset_name} | {name} | macro-F1={scores['test_score'].mean():.3f}")
            except Exception as exc:
                print(f"Skipped {dataset_name}/{name}: {exc}")
    results = pd.DataFrame(records)
    if results.empty:
        raise RuntimeError("No benchmark records were produced.")
    results.to_csv(output_dir / "data/meta_dataset.csv", index=False)
    results.to_csv(output_dir / "results/benchmark_results.csv", index=False)
    print(f"Saved {len(results)} dataset-algorithm records.")
    return results


if __name__ == "__main__":
    run_benchmarks()
