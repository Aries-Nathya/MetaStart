"""Reproducible classification benchmark and meta-dataset builder."""

from __future__ import annotations

import time
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
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
from src.dataset_catalog import EXPECTED_ALGORITHMS, generate_synthetic_datasets
from src.provenance import write_dataset_provenance

ROOT = Path(__file__).resolve().parents[1]
OPENML_CACHE_DIR = ROOT / "data" / "openml_cache"
ALGORITHM_NAMES = list(EXPECTED_ALGORITHMS)
META_DATASET_COLUMNS = ["dataset", "n_rows", "n_features", "n_numeric", "n_categorical",
                        "missing_ratio", "n_classes", "imbalance_ratio", "mean_feature_mean",
                        "mean_feature_std", "algorithm", "macro_f1", "training_seconds", "wall_seconds"]
MAX_OPENML_ROWS = int(os.environ.get("METASTART_MAX_BENCHMARK_ROWS", 50_000))
MAX_OPENML_FEATURES = int(os.environ.get("METASTART_MAX_BENCHMARK_FEATURES", 500))


def validate_classification_dataset(X, y, *, min_samples_per_class=3,
                                    max_rows=MAX_OPENML_ROWS,
                                    max_features=MAX_OPENML_FEATURES):
    """Return normalized data or raise ValueError when unsuitable for 3-fold CV."""
    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()
    if X.empty or X.shape[1] == 0:
        raise ValueError("dataset has no feature rows or columns")
    if y.empty:
        raise ValueError("dataset has no target values")
    if len(X) != len(y):
        raise ValueError("feature and target row counts differ")
    keep = y.notna()
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    if X.empty or y.empty:
        raise ValueError("dataset has no rows after dropping missing targets")
    if len(X) > max_rows or X.shape[1] > max_features:
        raise ValueError(f"dataset exceeds size limit ({max_rows} rows, {max_features} features)")
    # Numeric class codes are valid, but high-cardinality numeric targets are
    # ambiguous without OpenML task metadata; reject rather than guess regression
    # labels are classes. Numeric-looking strings/categories remain valid classes.
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 50:
        raise ValueError("numeric target has more than 50 distinct values; verify an explicit classification task")
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError("classification requires at least two target classes")
    if counts.min() < min_samples_per_class:
        raise ValueError(f"each class needs at least {min_samples_per_class} samples for stratified CV")
    return X, y


def load_openml_datasets(dataset_specs, *, fetcher=None):
    """Load explicitly selected OpenML IDs; return accepted datasets and provenance rows."""
    if fetcher is None:
        from sklearn.datasets import fetch_openml
        fetcher = fetch_openml
    accepted, manifest, seen_ids, seen_names = [], [], set(), set()
    for spec in dataset_specs:
        dataset_id, name = int(spec[0]), str(spec[1]).strip()
        identity = f"openml_{dataset_id}_{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}"
        row = {"dataset": identity, "source": "OpenML", "source_id": dataset_id,
               "source_name": name, "license_status": "not reported",
               "status": "failed", "error": ""}
        try:
            if not name:
                raise ValueError("dataset name must be non-empty")
            if dataset_id in seen_ids or identity in seen_names:
                raise ValueError("duplicate OpenML dataset ID or identity in request")
            seen_ids.add(dataset_id)
            seen_names.add(identity)
            OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            bunch = fetcher(data_id=dataset_id, as_frame=True,
                            data_home=str(OPENML_CACHE_DIR))
            details = getattr(bunch, "details", {}) or {}
            reported_license = details.get("licence", details.get("license"))
            if reported_license:
                row["license_status"] = str(reported_license)
            X = bunch.data
            y = bunch.target
            X, y = validate_classification_dataset(X, y)
            accepted.append((identity, X, y))
            row["status"] = "loaded"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        manifest.append(row)
    return accepted, manifest


def get_datasets():
    """Built-in datasets plus deterministic synthetic datasets for variety."""
    loaders = [load_breast_cancer, load_iris, load_wine, load_digits]
    for loader in loaders:
        bunch = loader()
        yield loader.__name__.replace("load_", ""), pd.DataFrame(bunch.data, columns=bunch.feature_names), pd.Series(bunch.target)
    yield from generate_synthetic_datasets()


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


def run_benchmarks(output_dir: Path | str = ROOT, *, openml_datasets=None):
    output_dir = Path(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    records = []
    datasets = list(get_datasets())
    manifest = []
    if openml_datasets:
        extra, manifest = load_openml_datasets(openml_datasets)
        existing_names = {name for name, _, _ in datasets}
        for dataset_name, X, y in extra:
            if dataset_name in existing_names:
                manifest.append({"dataset": dataset_name, "source": "OpenML", "source_id": "",
                                 "source_name": dataset_name, "status": "failed",
                                 "error": "dataset identity already exists in benchmark collection"})
            else:
                datasets.append((dataset_name, X, y))
                existing_names.add(dataset_name)
    for dataset_name, X, y in datasets:
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
                for entry in manifest:
                    if entry.get("dataset") == dataset_name and entry.get("status") == "loaded":
                        entry["status"] = "benchmark_partial_failure"
                        detail = f"{name}: {type(exc).__name__}: {exc}"
                        entry["error"] = "; ".join(filter(None, [entry.get("error", ""), detail]))
                        break
    run_results = pd.DataFrame(records, columns=META_DATASET_COLUMNS)
    results = run_results.copy()
    if results.empty:
        raise RuntimeError("No benchmark records were produced.")
    # Keep historical identities that are not part of this run. Fresh measurements
    # replace matching dataset/algorithm rows; no other rows are discarded.
    output_path = output_dir / "data/meta_dataset.csv"
    if output_path.exists():
        previous = pd.read_csv(output_path)
        if list(previous.columns) != META_DATASET_COLUMNS:
            raise ValueError("Existing meta-dataset does not match the required 14-column schema")
        combined = pd.concat([previous, results], ignore_index=True)
        combined = combined.drop_duplicates(["dataset", "algorithm"], keep="last")
        results = combined[META_DATASET_COLUMNS]
    results.to_csv(output_path, index=False)
    results.to_csv(output_dir / "results/benchmark_results.csv", index=False)
    provenance = write_dataset_provenance(results, datasets, output_dir, run_results=run_results)
    incomplete = provenance[provenance.benchmark_status != "complete"]
    for _, row in incomplete.iterrows():
        print(f"Incomplete benchmark {row.dataset}: found fewer than five candidate records")
    if openml_datasets is not None:
        manifest_path = output_dir / "results/openml_manifest.csv"
        manifest_columns = ["dataset", "source", "source_id", "source_name",
                            "license_status", "status", "error"]
        new_manifest = pd.DataFrame(manifest, columns=manifest_columns)
        if manifest_path.exists():
            new_manifest = pd.concat([pd.read_csv(manifest_path), new_manifest], ignore_index=True)
            new_manifest = new_manifest.drop_duplicates(["source", "source_id"], keep="last")
        new_manifest.to_csv(manifest_path, index=False)
    print(f"Saved {len(results)} dataset-algorithm records.")
    return results


if __name__ == "__main__":
    run_benchmarks()
