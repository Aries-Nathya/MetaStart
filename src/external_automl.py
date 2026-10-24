"""Optional external AutoML comparisons on MetaStart's existing split."""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.app_workflow import CV_FOLDS, CV_RANDOM_STATE
from src.evaluation_metrics import class_breakdown_rows, compute_measured_metrics
from src.experiment_results import ExperimentRecord

DEFAULT_TIME_BUDGET_SECONDS = 15 * 60
MAX_TIME_BUDGET_SECONDS = 15 * 60
EXTERNAL_CPU_LIMIT = 2
EXTERNAL_MEMORY_LIMIT_GB = 2
MAX_EXTERNAL_TRAIN_CELLS = 5_000_000


@dataclass
class FrameworkRun:
    framework: str
    status: str
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_model_name: str | None = None
    selected_parameters: dict[str, Any] | None = None
    selected_cv_macro_f1: float | None = None
    model: Any = None
    search_seconds: float | None = None
    error: str | None = None
    cleanup: Callable[[], None] | None = None
    model_failures: tuple[dict[str, Any], ...] = ()


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _run_flaml(X_train: pd.DataFrame, y_train: pd.Series, *,
               time_budget_seconds: int, seed: int, cv_folds: int) -> FrameworkRun:
    started = time.perf_counter()
    try:
        from flaml import AutoML
    except ImportError as exc:
        return FrameworkRun("FLAML", "missing_dependency", error=f"{type(exc).__name__}: {exc}")
    try:
        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="classification",
            metric="macro_f1",
            eval_method="cv",
            n_splits=cv_folds,
            time_budget=time_budget_seconds,
            seed=seed,
            n_jobs=EXTERNAL_CPU_LIMIT,
            n_concurrent_trials=1,
            estimator_list=["rf", "extra_tree"],
            verbose=0,
        )
        losses = getattr(automl, "best_loss_per_estimator", {}) or {}
        configurations = getattr(automl, "best_config_per_estimator", {}) or {}
        rows = []
        for learner, loss in losses.items():
            score = 1.0 - float(loss)
            rows.append({
                "Framework": "FLAML",
                "Model": str(learner),
                "CV Macro-F1": score,
                "Fit Seconds": None,
                "Parameters": _clean_value(configurations.get(learner)),
                "Framework Metrics": {"best_loss": float(loss)},
            })
        leaderboard = pd.DataFrame(rows)
        if leaderboard.empty:
            raise RuntimeError("FLAML completed without a per-estimator CV leaderboard.")
        selected_name = str(getattr(automl, "best_estimator", "unknown"))
        selected_config = _clean_value(configurations.get(selected_name))
        selected_score = (1.0 - float(losses[selected_name])
                          if selected_name in losses else None)
        if selected_score is None:
            selected_row = leaderboard.loc[leaderboard["CV Macro-F1"].idxmax()]
            selected_name = str(selected_row["Model"])
            selected_score = float(selected_row["CV Macro-F1"])
            selected_config = selected_row["Parameters"]
        return FrameworkRun(
            "FLAML", "completed", leaderboard, selected_name,
            selected_config, selected_score, automl,
            time.perf_counter() - started,
        )
    except Exception as exc:
        return FrameworkRun("FLAML", "failed", error=f"{type(exc).__name__}: {exc}",
                            search_seconds=time.perf_counter() - started)


def _run_autogluon(X_train: pd.DataFrame, y_train: pd.Series, *,
                   time_budget_seconds: int, seed: int, cv_folds: int) -> FrameworkRun:
    if sys.version_info < (3, 10) or sys.version_info >= (3, 14):
        version = ".".join(map(str, sys.version_info[:3]))
        return FrameworkRun(
            "AutoGluon", "unsupported_python",
            error=(f"AutoGluon 1.6 supports Python 3.10–3.13; "
                   f"the active runtime is Python {version}."),
        )
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        return FrameworkRun("AutoGluon", "missing_dependency", error=f"{type(exc).__name__}: {exc}")

    path = Path(tempfile.mkdtemp(prefix="metastart-autogluon-"))
    started = time.perf_counter()
    try:
        label_column = "__metastart_target__"
        train_data = X_train.copy()
        train_data[label_column] = pd.Series(y_train).reset_index(drop=True).to_numpy()
        predictor = TabularPredictor(
            label=label_column,
            problem_type="binary" if pd.Series(y_train).nunique() == 2 else "multiclass",
            eval_metric="f1_macro",
            path=str(path),
            verbosity=0,
        ).fit(
            train_data=train_data,
            time_limit=time_budget_seconds,
            presets="medium_quality",
            num_cpus=EXTERNAL_CPU_LIMIT,
            num_gpus=0,
            num_bag_folds=cv_folds,
            num_bag_sets=1,
            num_stack_levels=0,
            dynamic_stacking=False,
            fit_weighted_ensemble=False,
            memory_limit=EXTERNAL_MEMORY_LIMIT_GB,
            raise_on_no_models_fitted=False,
        )
        leaderboard = predictor.leaderboard(silent=True)
        rows = []
        for _, row in leaderboard.iterrows():
            model_name = str(row.get("model"))
            parameters = None
            try:
                parameters = _clean_value(predictor.model_hyperparameters(model_name))
            except Exception:
                pass
            row_mapping = {str(key): _clean_value(value) for key, value in row.to_dict().items()}
            rows.append({
                "Framework": "AutoGluon",
                "Model": model_name,
                "CV Macro-F1": _clean_value(row.get("score_val")),
                "Fit Seconds": _clean_value(row.get("fit_time")),
                "Parameters": parameters,
                "Framework Metrics": row_mapping,
            })
        leaderboard = pd.DataFrame(rows)
        if leaderboard.empty:
            raise RuntimeError("AutoGluon completed without a leaderboard.")
        failure_rows: tuple[dict[str, Any], ...] = ()
        try:
            failures = predictor.model_failures()
            failure_rows = tuple({
                "Model": str(row.get("model", "unknown")),
                "Exception": str(row.get("exc_type", "Exception")),
                "Details": str(row.get("exc_str", ""))[:500],
                "Fit Seconds": _clean_value(row.get("total_time")),
            } for _, row in failures.iterrows())
        except Exception:
            # Older compatible AutoGluon releases may not expose model_failures.
            failure_rows = ()
        selected_name = str(predictor.model_best)
        selected = leaderboard.loc[leaderboard["Model"].eq(selected_name)]
        selected_score = (float(selected.iloc[0]["CV Macro-F1"])
                          if not selected.empty and pd.notna(selected.iloc[0]["CV Macro-F1"])
                          else None)
        selected_parameters = (selected.iloc[0]["Parameters"] if not selected.empty else None)
        return FrameworkRun(
            "AutoGluon", "completed", leaderboard, selected_name,
            selected_parameters, selected_score, predictor,
            time.perf_counter() - started,
            cleanup=lambda: shutil.rmtree(path, ignore_errors=True),
            model_failures=failure_rows,
        )
    except Exception as exc:
        shutil.rmtree(path, ignore_errors=True)
        return FrameworkRun("AutoGluon", "failed", error=f"{type(exc).__name__}: {exc}",
                            search_seconds=time.perf_counter() - started)


def run_external_automl_comparison(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        *,
        time_budget_seconds: int = DEFAULT_TIME_BUDGET_SECONDS,
        seed: int = CV_RANDOM_STATE,
        cv_folds: int = CV_FOLDS,
        experiment_id: str = "uploaded",
        progress_callback: Callable[[str, str], None] | None = None,
) -> tuple[list[ExperimentRecord], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run FLAML and AutoGluon; held-out data is only used after each search."""
    if not 1 <= int(time_budget_seconds) <= MAX_TIME_BUDGET_SECONDS:
        raise ValueError(f"time_budget_seconds must be between 1 and {MAX_TIME_BUDGET_SECONDS}.")
    if cv_folds < 2:
        raise ValueError("External AutoML comparison requires at least two CV folds.")
    runners = (_run_flaml, _run_autogluon)
    records: list[ExperimentRecord] = []
    summaries: list[dict[str, Any]] = []
    leaderboards: dict[str, pd.DataFrame] = {}

    for runner in runners:
        framework = "FLAML" if runner is _run_flaml else "AutoGluon"
        if progress_callback is not None:
            progress_callback(framework, "running")
        if len(X_train) * X_train.shape[1] > MAX_EXTERNAL_TRAIN_CELLS:
            run = FrameworkRun(
                framework, "resource_skipped",
                error=(f"External AutoML is limited to {MAX_EXTERNAL_TRAIN_CELLS:,} training cells; "
                       f"this training partition has {len(X_train) * X_train.shape[1]:,}."),
            )
        else:
            try:
                run = runner(X_train, y_train, time_budget_seconds=int(time_budget_seconds),
                             seed=seed, cv_folds=cv_folds)
            except Exception as exc:
                run = FrameworkRun(framework, "failed", error=f"{type(exc).__name__}: {exc}")

        heldout_metrics: dict[str, float | None] = {}
        heldout_breakdown: tuple[dict[str, Any], ...] = ()
        test_seconds = None
        heldout_error = None
        if run.status == "completed" and run.model is not None:
            started = time.perf_counter()
            try:
                predictions = run.model.predict(X_test)
                test_seconds = time.perf_counter() - started
                classes = sorted(pd.Series(y_train).astype(str).unique().tolist())
                heldout_metrics = compute_measured_metrics(
                    y_test, predictions, estimator=run.model, X=X_test, labels=classes,
                )
                heldout_breakdown = tuple(class_breakdown_rows(y_test, predictions, classes))
            except Exception as exc:
                heldout_error = f"{type(exc).__name__}: {exc}"
            finally:
                if run.cleanup is not None:
                    run.cleanup()
        elif run.cleanup is not None:
            run.cleanup()

        leaderboard = run.leaderboard.copy()
        leaderboards[framework] = leaderboard
        if run.status == "completed" and run.model is None:
            effective_status = "selected_model_missing"
        elif run.status == "completed" and heldout_error is not None:
            effective_status = "heldout_failed"
        else:
            effective_status = run.status
        error = heldout_error or run.error

        for _, entry in leaderboard.iterrows():
            score = entry.get("CV Macro-F1")
            model_name = str(entry.get("Model", "unknown"))
            raw_metrics = entry.get("Framework Metrics", {})
            if not isinstance(raw_metrics, dict):
                raw_metrics = {"leaderboard_entry": _clean_value(raw_metrics)}
            records.append(ExperimentRecord(
                experiment_id=f"{experiment_id}::external::{framework}::{model_name}",
                method=f"{framework}_automl",
                algorithm=f"{framework}: {model_name}",
                stage="external_automl",
                status="completed" if pd.notna(score) else "cv_score_missing",
                configuration="framework_selected_configuration",
                primary_metric_value=float(score) if pd.notna(score) else None,
                metrics={"macro_f1": float(score) if pd.notna(score) else None},
                cv_folds=cv_folds,
                cv_rows=len(X_train),
                cv_source_rows=len(X_train),
                cv_macro_f1=float(score) if pd.notna(score) else None,
                fit_seconds=(float(entry["Fit Seconds"])
                             if pd.notna(entry.get("Fit Seconds")) else None),
                trials_evaluated=len(leaderboard),
                best_parameters=(entry.get("Parameters")
                                 if isinstance(entry.get("Parameters"), dict) else None),
                leaderboard_metrics=raw_metrics,
                notes=(f"{framework}-reported training CV Macro-F1. "
                       "Held-out rows were not used for this leaderboard score.",),
            ))

        for failure in run.model_failures:
            reason = f"{failure['Exception']}: {failure['Details']}"
            records.append(ExperimentRecord(
                experiment_id=(f"{experiment_id}::external::{framework}::"
                               f"failed::{failure['Model']}"),
                method=f"{framework}_automl",
                algorithm=f"{framework}: {failure['Model']}",
                stage="external_automl",
                status="model_failed",
                configuration="framework_default_configuration",
                fit_seconds=failure.get("Fit Seconds"),
                cv_folds=cv_folds,
                cv_rows=len(X_train),
                cv_source_rows=len(X_train),
                leaderboard_metrics={"failure": failure},
                failures=(reason,),
                notes=("AutoGluon attempted this model but did not produce a leaderboard score.",),
            ))

        summary_score = run.selected_cv_macro_f1
        records.append(ExperimentRecord(
            experiment_id=f"{experiment_id}::external::{framework}::summary",
            method=f"{framework}_automl",
            algorithm=framework,
            stage="external_automl",
            status=effective_status,
            configuration="selected_by_framework_macro_f1_cv",
            primary_metric_value=summary_score,
            metrics={"macro_f1": summary_score},
            cv_folds=cv_folds if summary_score is not None else None,
            cv_rows=len(X_train) if summary_score is not None else None,
            cv_source_rows=len(X_train) if summary_score is not None else None,
            cv_macro_f1=summary_score,
            fit_seconds=run.search_seconds,
            test_seconds=test_seconds,
            test_rows=len(X_test) if heldout_metrics else None,
            seed=seed,
            trials_evaluated=len(leaderboard) if not leaderboard.empty else None,
            best_parameters=run.selected_parameters,
            leaderboard_metrics={
                "time_budget_seconds": int(time_budget_seconds),
                "leaderboard_entries": len(leaderboard),
                "requested_seed": seed,
                "framework_seed": seed if framework == "FLAML" else 0,
                "seed_note": ("FLAML seed set by this run." if framework == "FLAML"
                              else "AutoGluon Tabular uses its documented default model seed (0); no global seed argument is exposed by its stable predictor API."),
                "failed_models": list(run.model_failures),
            },
            notes=((f"Search used the training partition only. Total search runtime was "
                    f"{run.search_seconds:.3f} seconds." if run.search_seconds is not None
                    else "Search did not complete."),),
            failures=(error,) if error else (),
        ))
        if heldout_metrics:
            records.append(ExperimentRecord(
                experiment_id=f"{experiment_id}::external::{framework}::heldout",
                method=f"{framework}_automl",
                algorithm=f"{framework}: {run.selected_model_name}",
                stage="external_automl_held_out_test",
                status="completed",
                configuration="framework_cv_selected_model",
                primary_metric_value=heldout_metrics.get("macro_f1"),
                metrics=heldout_metrics,
                class_breakdown=heldout_breakdown,
                cv_folds=cv_folds,
                cv_rows=len(X_train),
                cv_source_rows=len(X_train),
                cv_macro_f1=run.selected_cv_macro_f1,
                fit_seconds=run.search_seconds,
                test_seconds=test_seconds,
                test_rows=len(X_test),
                seed=seed if framework == "FLAML" else None,
                best_parameters=run.selected_parameters,
                notes=("Held-out metrics were computed after framework search and model selection. "
                       "Test labels did not enter AutoML training or leaderboard selection.",),
            ))
        summaries.append({
            "Framework": framework,
            "Status": effective_status,
            "Selected model": run.selected_model_name,
            "Training CV Macro-F1": summary_score,
            "Held-out Macro-F1": heldout_metrics.get("macro_f1"),
            "Held-out Accuracy": heldout_metrics.get("accuracy"),
            "Search seconds": run.search_seconds,
            "Test scoring seconds": test_seconds,
            "Leaderboard entries": len(leaderboard),
            "Time budget seconds": int(time_budget_seconds),
            "Details": error,
            "Limitations": "; ".join(
                f"{failure['Model']}: {failure['Exception']} — {failure['Details']}"
                for failure in run.model_failures
            ) or None,
        })
        if progress_callback is not None:
            progress_callback(framework, effective_status)

    return records, pd.DataFrame(summaries), leaderboards
