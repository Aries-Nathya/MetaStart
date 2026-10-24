"""Bounded training-only baseline and hyperparameter search experiments."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score

from src.app_workflow import CV_FOLDS, CV_MAX_ROWS, CV_RANDOM_STATE
from src.benchmark import build_pipeline, get_models
from src.experiment_results import ExperimentRecord

SEARCH_TRIALS = 6
SEARCH_SEED = 42

# Small, explicit spaces bound each experiment while covering meaningful model
# choices. The same values drive grid, random, and Bayesian search.
_SPACE: dict[str, dict[str, tuple[Any, ...]]] = {
    "Logistic Regression": {
        "C": (0.1, 1.0, 10.0),
        "class_weight": (None, "balanced"),
    },
    "Decision Tree": {
        "max_depth": (None, 5, 15),
        "min_samples_leaf": (1, 3),
        "criterion": ("gini", "entropy"),
    },
    "Random Forest": {
        "max_depth": (None, 15),
        "min_samples_leaf": (1, 3),
        "max_features": ("sqrt", None),
    },
    "SVM": {
        "C": (0.5, 1.0, 5.0),
        "gamma": ("scale", "auto"),
    },
    "k-Nearest Neighbors": {
        "n_neighbors": (3, 5, 9, 15),
        "weights": ("uniform", "distance"),
        "p": (1, 2),
    },
}


def _prefixed_space(algorithm: str, X: pd.DataFrame) -> tuple[dict[str, list[Any]], str]:
    if algorithm not in _SPACE:
        raise ValueError(f"No bounded tuning space is defined for {algorithm!r}.")
    pipeline = build_pipeline(get_models()[algorithm], X)
    step_name = pipeline.steps[-1][0]
    return ({f"{step_name}__{name}": list(values)
             for name, values in _SPACE[algorithm].items()}, step_name)


def _bounded_cv_sample(X: pd.DataFrame, y: pd.Series, *, max_rows: int,
                       min_per_class: int) -> tuple[pd.DataFrame, pd.Series]:
    """Match MetaStart's deterministic CV sample without copying the full X."""
    y = pd.Series(y).reset_index(drop=True)
    if len(X) <= max_rows:
        return X.reset_index(drop=True), y
    class_counts = y.value_counts(sort=False)
    if max_rows < min_per_class * len(class_counts):
        raise ValueError("CV row budget is too small to retain the required examples from every class.")
    quotas = class_counts.to_numpy(dtype=float) / len(y) * max_rows
    allocations = np.maximum(np.floor(quotas).astype(int), min_per_class)
    counts = class_counts.to_numpy(dtype=int)
    allocations = np.minimum(allocations, counts)
    excess = int(allocations.sum() - max_rows)
    while excess > 0:
        eligible = np.flatnonzero(allocations > min_per_class)
        if not len(eligible):
            raise ValueError("Could not fit all target classes within the CV row budget.")
        index = eligible[np.argmax(allocations[eligible] - min_per_class)]
        take = min(excess, int(allocations[index] - min_per_class))
        allocations[index] -= take
        excess -= take
    remaining = int(max_rows - allocations.sum())
    while remaining > 0:
        eligible = np.flatnonzero(allocations < counts)
        if not len(eligible):
            break
        index = eligible[np.argmax(quotas[eligible] - allocations[eligible])]
        allocations[index] += 1
        remaining -= 1

    rng = np.random.default_rng(CV_RANDOM_STATE)
    selected: list[int] = []
    labels = class_counts.index.to_numpy()
    y_values = y.to_numpy()
    for label, amount in zip(labels, allocations):
        positions = np.flatnonzero(y_values == label)
        selected.extend(rng.choice(positions, size=int(amount), replace=False).tolist())
    selected_array = np.asarray(selected, dtype=int)
    rng.shuffle(selected_array)
    return (X.iloc[selected_array].reset_index(drop=True),
            y.iloc[selected_array].reset_index(drop=True))


def _optuna_space(trial, algorithm: str) -> dict[str, Any]:
    if algorithm == "Logistic Regression":
        return {
            "C": trial.suggest_categorical("C", list(_SPACE[algorithm]["C"])),
            "class_weight": trial.suggest_categorical("class_weight", list(_SPACE[algorithm]["class_weight"])),
        }
    if algorithm == "Decision Tree":
        return {
            "max_depth": trial.suggest_categorical("max_depth", list(_SPACE[algorithm]["max_depth"])),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", list(_SPACE[algorithm]["min_samples_leaf"])),
            "criterion": trial.suggest_categorical("criterion", list(_SPACE[algorithm]["criterion"])),
        }
    if algorithm == "Random Forest":
        return {
            "max_depth": trial.suggest_categorical("max_depth", list(_SPACE[algorithm]["max_depth"])),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", list(_SPACE[algorithm]["min_samples_leaf"])),
            "max_features": trial.suggest_categorical("max_features", list(_SPACE[algorithm]["max_features"])),
        }
    if algorithm == "SVM":
        return {
            "C": trial.suggest_categorical("C", list(_SPACE[algorithm]["C"])),
            "gamma": trial.suggest_categorical("gamma", list(_SPACE[algorithm]["gamma"])),
        }
    return {
        "n_neighbors": trial.suggest_categorical("n_neighbors", list(_SPACE[algorithm]["n_neighbors"])),
        "weights": trial.suggest_categorical("weights", list(_SPACE[algorithm]["weights"])),
        "p": trial.suggest_categorical("p", list(_SPACE[algorithm]["p"])),
    }


def run_tuning_experiments(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithm: str,
        *,
        cv_folds: int = CV_FOLDS,
        max_cv_rows: int = CV_MAX_ROWS,
        trials: int = SEARCH_TRIALS,
        seed: int = SEARCH_SEED,
        experiment_id: str = "uploaded",
) -> tuple[list[ExperimentRecord], pd.DataFrame]:
    """Run untuned CV, grid, random, and Optuna search on training data only.

    All methods share one deterministic, class-stratified CV sample and fixed
    folds. No held-out rows or labels are accepted by this API.
    """
    if algorithm not in get_models():
        raise ValueError(f"Unsupported candidate algorithm: {algorithm!r}")
    if trials < 1:
        raise ValueError("trials must be positive")
    minimum_class_count = int(pd.Series(y_train).value_counts().min())
    n_splits = min(int(cv_folds), minimum_class_count)
    if n_splits < 2:
        raise ValueError("Training partition needs at least two examples per class for tuning CV.")
    X_cv, y_cv = _bounded_cv_sample(
        X_train, pd.Series(y_train), max_rows=max_cv_rows, min_per_class=n_splits,
    )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CV_RANDOM_STATE)
    folds = list(splitter.split(X_cv, y_cv))
    params, step_name = _prefixed_space(algorithm, X_cv)
    model = get_models()[algorithm]
    base_pipeline = build_pipeline(model, X_cv)
    display_rows: list[dict[str, Any]] = []
    records: list[ExperimentRecord] = []

    def store(method: str, status: str, *, score: float | None = None,
              elapsed: float | None = None, best_params: dict[str, Any] | None = None,
              evaluated: int | None = None, error: Exception | None = None) -> None:
        failure = f"{type(error).__name__}: {error}" if error is not None else None
        records.append(ExperimentRecord(
            experiment_id=f"{experiment_id}::{method}::{algorithm}",
            method=method,
            algorithm=algorithm,
            stage=method,
            status=status,
            configuration="untuned" if method == "manual_baseline" else "bounded_training_cv_search",
            primary_metric_value=score,
            metrics={"macro_f1": score},
            cv_folds=n_splits,
            cv_rows=len(X_cv),
            cv_source_rows=len(X_train),
            cv_macro_f1=score,
            fit_seconds=elapsed,
            trials_evaluated=evaluated,
            best_parameters=best_params,
            notes=("Training-partition CV only; the held-out test set was not accessed. "
                   "Fit Seconds is total wall time across CV evaluations for this method.",),
            failures=(failure,) if failure else (),
        ))
        display_rows.append({
            "Experiment": method.replace("_", " ").title(),
            "Algorithm": algorithm,
            "CV Macro-F1": score,
            "CV folds": n_splits,
            "CV rows": len(X_cv),
            "Fit seconds": elapsed,
            "Trials evaluated": evaluated,
            "Best parameters": best_params,
            "Status": status,
            "Failure": failure,
        })

    # Untuned reference using the same default estimator as MetaStart's candidate.
    start = time.perf_counter()
    try:
        scores = cross_val_score(base_pipeline, X_cv, y_cv, cv=folds,
                                 scoring="f1_macro", n_jobs=1, error_score="raise")
        store("manual_baseline", "completed", score=float(np.mean(scores)),
              elapsed=time.perf_counter() - start, evaluated=1)
    except Exception as exc:
        store("manual_baseline", "failed", elapsed=time.perf_counter() - start,
              evaluated=0, error=exc)

    searches = (
        ("grid_search", GridSearchCV(
            base_pipeline, params, scoring="f1_macro", cv=folds,
            n_jobs=1, refit=False, error_score="raise")),
        ("random_search", RandomizedSearchCV(
            base_pipeline, params, n_iter=min(trials, int(np.prod([len(v) for v in params.values()]))),
            scoring="f1_macro", cv=folds, n_jobs=1, refit=False,
            random_state=seed, error_score="raise")),
    )
    for method, search in searches:
        start = time.perf_counter()
        try:
            search.fit(X_cv, y_cv)
            store(method, "completed", score=float(search.best_score_),
                  elapsed=time.perf_counter() - start,
                  best_params={key.removeprefix(f"{step_name}__"): value
                               for key, value in search.best_params_.items()},
                  evaluated=len(search.cv_results_["params"]))
        except Exception as exc:
            store(method, "failed", elapsed=time.perf_counter() - start,
                  evaluated=0, error=exc)

    start = time.perf_counter()
    try:
        import optuna

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        def objective(trial):
            chosen = _optuna_space(trial, algorithm)
            estimator = build_pipeline(get_models()[algorithm], X_cv)
            estimator.set_params(**{f"{step_name}__{key}": value
                                    for key, value in chosen.items()})
            return float(np.mean(cross_val_score(
                estimator, X_cv, y_cv, cv=folds, scoring="f1_macro",
                n_jobs=1, error_score="raise",
            )))

        study.optimize(objective, n_trials=trials, n_jobs=1, show_progress_bar=False)
        store("optuna", "completed", score=float(study.best_value),
              elapsed=time.perf_counter() - start,
              best_params=study.best_params, evaluated=len(study.trials))
    except ImportError as exc:
        store("optuna", "dependency_missing", elapsed=time.perf_counter() - start,
              evaluated=0, error=exc)
    except Exception as exc:
        store("optuna", "failed", elapsed=time.perf_counter() - start,
              evaluated=0, error=exc)

    return records, pd.DataFrame(display_rows)
