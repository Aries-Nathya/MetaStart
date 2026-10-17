"""Testable input validation and candidate evaluation for the Streamlit app."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import BytesIO
from io import TextIOWrapper
import numbers
import os
import time
import warnings
from typing import BinaryIO, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.benchmark import ALGORITHM_NAMES
from sklearn.svm import SVC


def _positive_setting(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class UploadLimits:
    """Centralized upload and parsed-table safety limits."""

    max_rows: int = _positive_setting("METASTART_MAX_UPLOAD_ROWS", 750_000)
    max_file_bytes: int = _positive_setting("METASTART_MAX_UPLOAD_MB", 200) * 1024 * 1024
    max_features: int = _positive_setting("METASTART_MAX_UPLOAD_FEATURES", 100)
    max_cells: int = _positive_setting("METASTART_MAX_UPLOAD_CELLS", 40_000_000)


UPLOAD_LIMITS = UploadLimits()
MAX_UPLOAD_ROWS = UPLOAD_LIMITS.max_rows
MAX_UPLOAD_BYTES = UPLOAD_LIMITS.max_file_bytes
MAX_UPLOAD_FEATURES = UPLOAD_LIMITS.max_features
MAX_UPLOAD_CELLS = UPLOAD_LIMITS.max_cells
MIN_SAMPLES_PER_CLASS = 4
MAX_TARGET_CLASSES = 50
CV_MAX_ROWS = 5_000
CV_FOLDS = 3
CV_RANDOM_STATE = 42

# Full-data refit guards. CV still evaluates all candidates on the same
# deterministic 5,000-row cap; these limits only govern the final refit.
MAX_FULL_FIT_ROWS = {
    "Logistic Regression": 750_000,
    "Decision Tree": 500_000,
    "Random Forest": 250_000,
    "SVM": 20_000,
    "k-Nearest Neighbors": 750_000,
}
MAX_KNN_DISTANCE_PAIRS = 25_000_000


class UploadValidationError(ValueError):
    """An actionable CSV or classification-data validation error."""


@dataclass(frozen=True)
class PreparedClassificationData:
    X: pd.DataFrame
    y: pd.Series
    dropped_target_rows: int
    dropped_all_missing_features: tuple[str, ...]


@dataclass(frozen=True)
class CandidateEvaluation:
    selection_results: pd.DataFrame
    test_result: pd.DataFrame
    selected_algorithm: str | None
    fitted_models: dict[str, object]
    failures: tuple[tuple[str, str, str], ...]
    class_metrics: pd.DataFrame
    error_patterns: pd.DataFrame
    boundary_diagnostic: pd.DataFrame
    cv_source_rows: int
    cv_rows: int
    cv_n_splits: int


def _stratified_cv_sample(X: pd.DataFrame, y: pd.Series, *, max_rows: int,
                          min_per_class: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic class-stratified CV sample, or all rows if under budget."""
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    if len(X) <= max_rows:
        return X, y
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
        idx = eligible[np.argmax(allocations[eligible] - min_per_class)]
        take = min(excess, int(allocations[idx] - min_per_class))
        allocations[idx] -= take
        excess -= take
    remaining = int(max_rows - allocations.sum())
    while remaining > 0:
        eligible = np.flatnonzero(allocations < counts)
        if not len(eligible):
            break
        idx = eligible[np.argmax(quotas[eligible] - allocations[eligible])]
        allocations[idx] += 1
        remaining -= 1

    rng = np.random.default_rng(random_state)
    selected = []
    labels = class_counts.index.to_numpy()
    y_values = y.to_numpy()
    for label, amount in zip(labels, allocations):
        positions = np.flatnonzero(y_values == label)
        selected.extend(rng.choice(positions, size=int(amount), replace=False).tolist())
    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)
    return X.iloc[selected].reset_index(drop=True), y.iloc[selected].reset_index(drop=True)


def _full_fit_resource_reason(algorithm: str, train_rows: int, test_rows: int) -> str | None:
    row_limit = MAX_FULL_FIT_ROWS.get(algorithm)
    if row_limit is not None and train_rows > row_limit:
        if algorithm == "SVM":
            return (f"Full-data refit skipped: the RBF SVM full-fit limit is {row_limit:,} training rows; "
                    f"this dataset has {train_rows:,}. Kernel SVM cost can grow sharply with row count.")
        if algorithm == "Random Forest":
            return (f"Full-data refit skipped: the Random Forest safety limit is {row_limit:,} training rows; "
                    f"this dataset has {train_rows:,}, and the current model uses 100 trees.")
        return (f"Full-data refit skipped: the configured {algorithm} limit is {row_limit:,} training rows; "
                f"this dataset has {train_rows:,}.")
    if algorithm == "k-Nearest Neighbors":
        distance_pairs = train_rows * test_rows
        if distance_pairs > MAX_KNN_DISTANCE_PAIRS:
            return ("Full-data refit/prediction skipped: k-NN would compare approximately "
                    f"{distance_pairs:,} train/test row pairs, above the "
                    f"{MAX_KNN_DISTANCE_PAIRS:,}-pair processing budget.")
    return None


def recommendation_download_table(ranking: pd.DataFrame) -> pd.DataFrame:
    """Format the real production predictions as a separate ranked CSV table."""
    required = {"algorithm", "predicted_macro_f1"}
    if not required.issubset(ranking.columns):
        raise ValueError(f"Recommendation ranking is missing columns: {sorted(required - set(ranking.columns))}")
    if len(ranking) != len(ALGORITHM_NAMES) or ranking["algorithm"].duplicated().any():
        raise ValueError("A full recommendation download requires exactly one row for each supported algorithm.")
    if set(ranking["algorithm"]) != set(ALGORITHM_NAMES):
        raise ValueError("Recommendation ranking does not contain the five supported algorithms.")
    ordered = ranking.sort_values("predicted_macro_f1", ascending=False, kind="stable")
    return pd.DataFrame({
        "Rank": range(1, len(ordered) + 1),
        "Algorithm": ordered["algorithm"].to_numpy(),
        "Predicted Macro-F1": ordered["predicted_macro_f1"].astype(float).to_numpy(),
    })


def load_uploaded_csv(source: bytes | bytearray | BinaryIO, *,
                      limits: UploadLimits = UPLOAD_LIMITS) -> pd.DataFrame:
    """Parse once, with file, row, width, and cell limits checked before large allocation."""
    if isinstance(source, (bytes, bytearray)):
        stream = BytesIO(source)
        size = len(source)
    elif hasattr(source, "read") and hasattr(source, "seek"):
        stream = source
        try:
            reported_size = getattr(source, "size", None)
            if reported_size is not None:
                size = int(reported_size)
            else:
                position = stream.tell()
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(position)
        except (OSError, AttributeError) as exc:
            raise UploadValidationError("Could not determine the uploaded file size safely.") from exc
    elif hasattr(source, "read"):
        raw = source.read(limits.max_file_bytes + 1)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if len(raw) > limits.max_file_bytes:
            raise UploadValidationError(
                f"The CSV exceeds the {limits.max_file_bytes // (1024 * 1024)} MiB upload limit."
            )
        stream = BytesIO(raw)
        size = len(raw)
    else:
        raise UploadValidationError("Choose a CSV file to upload.")
    if size <= 0:
        raise UploadValidationError("The uploaded file is empty or has no CSV header.")
    if size > limits.max_file_bytes:
        raise UploadValidationError(
            f"The CSV is {size / (1024 * 1024):.1f} MiB; the configured limit is "
            f"{limits.max_file_bytes / (1024 * 1024):.1f} MiB ({limits.max_file_bytes:,} bytes). "
            "Increase METASTART_MAX_UPLOAD_MB "
            "and Streamlit's server.maxUploadSize together if this file is trusted."
        )
    stream.seek(0)
    try:
        text_stream = TextIOWrapper(stream, encoding="utf-8-sig", newline="")
        try:
            header = next(csv.reader(text_stream, strict=True))
        finally:
            text_stream.detach()
        if not header or all(not str(name).strip() for name in header):
            raise UploadValidationError("The uploaded file has no usable CSV header.")
        if len(set(header)) != len(header):
            raise UploadValidationError("CSV column names must be unique.")
        if len(header) < 2:
            raise UploadValidationError("The CSV needs at least one feature column and one target column.")
        feature_count = len(header) - 1
        if feature_count > limits.max_features:
            raise UploadValidationError(
                f"The CSV has {feature_count:,} feature columns; the configured limit is "
                f"{limits.max_features:,} (METASTART_MAX_UPLOAD_FEATURES)."
            )
        row_limit = min(limits.max_rows, limits.max_cells // len(header))
    except UnicodeDecodeError as exc:
        raise UploadValidationError(
            "The CSV text encoding could not be read. Save the file as UTF-8 and upload it again."
        ) from exc
    except StopIteration as exc:
        raise UploadValidationError("The uploaded file is empty or has no CSV header.") from exc
    except csv.Error as exc:
        raise UploadValidationError(f"The CSV header is malformed: {exc}") from exc
    stream.seek(0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                stream, nrows=row_limit + 1, low_memory=False, index_col=False,
                on_bad_lines="error", na_values=["?"],
            )
    except pd.errors.EmptyDataError as exc:
        raise UploadValidationError("The uploaded file is empty or has no CSV header.") from exc
    except pd.errors.ParserError as exc:
        raise UploadValidationError(
            "The file is not a well-formed CSV. Check that each row has the same number of fields."
        ) from exc
    except pd.errors.ParserWarning as exc:
        raise UploadValidationError(
            f"Malformed CSV: {exc}. Check that each row has the same number of fields."
        ) from exc
    except UnicodeDecodeError as exc:
        raise UploadValidationError(
            "The CSV text encoding could not be read. Save the file as UTF-8 and upload it again."
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise UploadValidationError(f"Could not read the uploaded CSV: {exc}") from exc
    if len(frame) > row_limit and row_limit == limits.max_rows:
        raise UploadValidationError(
            f"The CSV contains more than the configured {limits.max_rows:,}-row limit "
            "(METASTART_MAX_UPLOAD_ROWS). Increase the limit only if the file-size and cell budgets are also safe."
        )
    if len(frame) > row_limit:
        raise UploadValidationError(
            f"This CSV width exceeds the configured {limits.max_cells:,}-cell processing budget "
            f"(METASTART_MAX_UPLOAD_CELLS); at this width, at most {row_limit:,} rows are allowed."
        )
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise UploadValidationError(f"CSV column names must be unique; duplicate(s): {duplicates}.")
    return frame


def _is_missing_scalar(value) -> bool:
    result = pd.isna(value)
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _validate_scalar_column(series: pd.Series, name: str) -> None:
    if (pd.api.types.is_complex_dtype(series.dtype)
            or pd.api.types.is_datetime64_any_dtype(series.dtype)
            or pd.api.types.is_timedelta64_dtype(series.dtype)):
        raise UploadValidationError(
            f"Column '{name}' uses an unsupported data type. Convert it to numeric values or text."
        )
    if (pd.api.types.is_numeric_dtype(series.dtype)
            or (pd.api.types.is_string_dtype(series.dtype)
                and not pd.api.types.is_object_dtype(series.dtype))
            or pd.api.types.is_bool_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)):
        return
    for value in series:
        if _is_missing_scalar(value):
            continue
        if not np.isscalar(value) or not isinstance(
                value, (str, bool, numbers.Real, np.number)):
            raise UploadValidationError(
                f"Column '{name}' contains unsupported values. Use numeric values or text categories."
            )
        if isinstance(value, (complex, np.complexfloating)):
            raise UploadValidationError(
                f"Column '{name}' contains complex numbers, which are not supported."
            )


def _as_category_text(value):
    return np.nan if _is_missing_scalar(value) else str(value)


def prepare_classification_data(
        frame: pd.DataFrame, target: str | None, *,
        max_rows: int = MAX_UPLOAD_ROWS,
        max_features: int = MAX_UPLOAD_FEATURES,
        max_cells: int = MAX_UPLOAD_CELLS) -> PreparedClassificationData:
    """Validate and normalize upload data before splitting or fitting estimators."""
    if not isinstance(frame, pd.DataFrame):
        raise UploadValidationError("The uploaded data must be a CSV table.")
    if frame.empty or len(frame.columns) < 2:
        raise UploadValidationError(
            "The CSV needs rows and at least one feature column plus one target column."
        )
    if frame.columns.duplicated().any():
        raise UploadValidationError("CSV column names must be unique before choosing a target.")
    if target is None or target not in frame.columns:
        raise UploadValidationError("Choose an existing target column from the uploaded CSV.")
    if len(frame) > max_rows:
        raise UploadValidationError(
            f"The CSV has {len(frame):,} rows, exceeding the row limit of {max_rows:,}."
        )
    feature_count = len(frame.columns) - 1
    if feature_count > max_features:
        raise UploadValidationError(
            f"The CSV has {feature_count:,} feature columns; the current limit is {max_features:,}."
        )
    if frame.shape[0] * frame.shape[1] > max_cells:
        raise UploadValidationError(
            f"The table has {frame.shape[0] * frame.shape[1]:,} cells, exceeding the configured "
            f"processing budget of {max_cells:,} (METASTART_MAX_UPLOAD_CELLS)."
        )

    y_raw = frame[target]
    _validate_scalar_column(y_raw, target)
    if pd.api.types.is_numeric_dtype(y_raw.dtype):
        y_raw = y_raw.replace([np.inf, -np.inf], np.nan)
    before_target_filter = len(frame)
    keep = y_raw.notna()
    if bool(keep.all()):
        X = frame.drop(columns=[target])
        y = y_raw
    else:
        X = frame.loc[keep].drop(columns=[target])
        y = y_raw.loc[keep]
    if not isinstance(X.index, pd.RangeIndex) or X.index.start != 0 or X.index.step != 1:
        X = X.reset_index(drop=True)
    if not isinstance(y.index, pd.RangeIndex) or y.index.start != 0 or y.index.step != 1:
        y = y.reset_index(drop=True)
    dropped_target_rows = before_target_filter - len(y)
    if y.empty:
        raise UploadValidationError("The selected target has no usable values.")

    if not pd.api.types.is_numeric_dtype(y.dtype):
        y = y.map(_as_category_text)
    class_counts = y.value_counts(dropna=True)
    if len(class_counts) < 2:
        raise UploadValidationError("The target must contain at least two distinct classes.")
    if len(class_counts) > MAX_TARGET_CLASSES:
        raise UploadValidationError(
            f"The target has {len(class_counts)} classes; the current limit is {MAX_TARGET_CLASSES}. "
            "If this is a regression target, this app currently supports classification only."
        )
    if class_counts.min() < MIN_SAMPLES_PER_CLASS:
        raise UploadValidationError(
            f"Each target class needs at least {MIN_SAMPLES_PER_CLASS} rows for a reliable stratified split."
        )

    for column in X.columns:
        _validate_scalar_column(X[column], str(column))
    numeric_columns = X.select_dtypes(include=np.number).columns
    if len(numeric_columns):
        numeric_values = X.loc[:, numeric_columns].to_numpy(copy=False)
        if np.isinf(numeric_values).any():
            X.loc[:, numeric_columns] = X.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
    categorical_columns = X.select_dtypes(exclude=np.number).columns
    for column in categorical_columns:
        if X[column].dtype == object:
            non_missing = X[column].dropna()
            if not non_missing.map(lambda value: isinstance(value, str)).all():
                X[column] = X[column].map(_as_category_text)
    empty_features = tuple(str(name) for name in X.columns[X.isna().all()])
    if empty_features:
        X = X.drop(columns=list(empty_features))
    if X.empty or X.shape[1] == 0:
        raise UploadValidationError(
            "No usable feature columns remain after removing columns with only missing values."
        )
    return PreparedClassificationData(X, y, dropped_target_rows, empty_features)


def evaluate_ranked_candidates(ranking: pd.DataFrame, X_train: pd.DataFrame,
                               y_train: pd.Series, X_test: pd.DataFrame,
                               y_test: pd.Series, *, model_factory,
                               pipeline_builder,
                               progress_callback: Callable[[int, int, str], None] | None = None,
                               cv_splits: int = 5, random_state: int = 42
                               ) -> CandidateEvaluation:
    """Choose via training-only CV, then evaluate the selected fit once on holdout."""
    required = {"algorithm", "predicted_macro_f1"}
    if not required.issubset(ranking.columns):
        raise ValueError(f"Recommendation ranking is missing columns: {sorted(required - set(ranking.columns))}")
    if ranking.empty:
        return CandidateEvaluation(pd.DataFrame(), pd.DataFrame(), None, {}, (),
                                   pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0, 0)

    ranked = ranking.sort_values("predicted_macro_f1", ascending=False, kind="stable")
    candidate_names = [name for name in ranked["algorithm"].tolist() if name in ALGORITHM_NAMES]
    if not candidate_names:
        return CandidateEvaluation(pd.DataFrame(), pd.DataFrame(), None, {}, (),
                                   pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0, 0)
    predictions = ranked.set_index("algorithm")["predicted_macro_f1"]
    models = model_factory()
    min_class_count = int(pd.Series(y_train).value_counts().min())
    n_splits = min(CV_FOLDS, int(cv_splits), min_class_count)
    if n_splits < 2:
        raise ValueError("Training partition needs at least two examples per class for stratified cross-validation.")
    cv_source_rows = len(X_train)
    X_cv, y_cv = _stratified_cv_sample(
        X_train, y_train, max_rows=CV_MAX_ROWS, min_per_class=n_splits,
        random_state=CV_RANDOM_STATE,
    )
    cv_rows = len(X_cv)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_splits = list(splitter.split(X_cv, y_cv))
    selection_rows, failures, class_rows, error_rows = [], [], [], []
    for index, name in enumerate(candidate_names):
        if progress_callback is not None:
            progress_callback(index, len(candidate_names) + 2, f"Cross-validating {name}")
        try:
            if name not in models:
                raise KeyError(f"No candidate estimator is registered for '{name}'.")
            pipeline = pipeline_builder(models[name], X_cv)
            cv = cross_validate(
                pipeline, X_cv, y_cv, cv=fold_splits, scoring="f1_macro",
                return_train_score=False, error_score="raise", n_jobs=1,
                return_estimator=True,
            )
            oof = np.empty(cv_rows, dtype=y_cv.to_numpy().dtype)
            for estimator, (_, valid_indices) in zip(cv["estimator"], fold_splits):
                oof[valid_indices] = estimator.predict(X_cv.iloc[valid_indices])
            labels = pd.unique(y_cv)
            precision, recall, class_f1, support = precision_recall_fscore_support(
                y_cv, oof, labels=labels, zero_division=0,
            )
            for label, p, r, f, count in zip(labels, precision, recall, class_f1, support):
                class_rows.append({"Algorithm": name, "Class": str(label), "Precision": p,
                                   "Recall": r, "F1": f, "Support": int(count)})
            confusion = pd.crosstab(
                pd.Series(y_cv.to_numpy(), name="True class"),
                pd.Series(oof, name="Predicted class"), dropna=False,
            )
            for actual in confusion.index:
                for predicted in confusion.columns:
                    count = int(confusion.loc[actual, predicted])
                    if actual != predicted and count:
                        error_rows.append({"Algorithm": name, "True class": str(actual),
                                           "Predicted class": str(predicted), "Count": count})
            selection_rows.append({
                "Algorithm": name,
                "Predicted Macro-F1": float(predictions.loc[name]),
                "CV Macro-F1 Mean": float(np.mean(cv["test_score"])),
                "CV Macro-F1 Std": float(np.std(cv["test_score"], ddof=1)) if n_splits > 1 else 0.0,
                "CV Fit Seconds Mean": float(np.mean(cv["fit_time"])),
            })
        except Exception as exc:
            failures.append((name, "cross-validation", f"{type(exc).__name__}: {exc}"))

    columns = ["Algorithm", "Predicted Macro-F1", "CV Macro-F1 Mean",
               "CV Macro-F1 Std", "CV Fit Seconds Mean"]
    selection_frame = pd.DataFrame(selection_rows, columns=columns)
    if not selection_frame.empty:
        selection_frame = selection_frame.sort_values(
            "CV Macro-F1 Mean", ascending=False, kind="stable").reset_index(drop=True)

    # Diagnostic only: this linear-kernel comparison never enters selection.
    boundary_rows = []
    if "SVM" in models:
        try:
            if progress_callback is not None:
                progress_callback(len(candidate_names), len(candidate_names) + 2,
                                  "Running diagnostic-only linear SVM comparison")
            linear_pipeline = pipeline_builder(
                SVC(kernel="linear", class_weight="balanced"), X_cv,
            )
            linear_cv = cross_validate(
                linear_pipeline, X_cv, y_cv, cv=fold_splits, scoring="f1_macro",
                return_train_score=False, error_score="raise", n_jobs=1,
            )
            boundary_rows.append({
                "Diagnostic": "Linear-kernel SVM (diagnostic only)",
                "CV Macro-F1 Mean": float(np.mean(linear_cv["test_score"])),
                "CV Macro-F1 Std": float(np.std(linear_cv["test_score"], ddof=1)),
                "CV Fit Seconds Mean": float(np.mean(linear_cv["fit_time"])),
                "Status": "Completed; not eligible for selection.",
            })
        except Exception as exc:
            failures.append(("Linear-kernel SVM", "diagnostic", f"{type(exc).__name__}: {exc}"))
            boundary_rows.append({"Diagnostic": "Linear-kernel SVM (diagnostic only)",
                                  "CV Macro-F1 Mean": np.nan, "CV Macro-F1 Std": np.nan,
                                  "CV Fit Seconds Mean": np.nan,
                                  "Status": f"Diagnostic failed: {type(exc).__name__}: {exc}"})

    fitted: dict[str, object] = {}
    selected_name = None
    elapsed = None
    for fit_index, row in enumerate(selection_frame.itertuples(index=False)):
        name = row.Algorithm
        if progress_callback is not None:
            progress_callback(len(candidate_names) + 1, len(candidate_names) + 2,
                              f"Fitting CV-selected {name}")
        try:
            resource_reason = _full_fit_resource_reason(name, len(X_train), len(X_test))
            if resource_reason:
                failures.append((name, "resource limit", resource_reason))
                continue
            pipeline = pipeline_builder(models[name], X_train)
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            selected_name = name
            fitted[name] = pipeline
            break
        except Exception as exc:
            failures.append((name, "final fit", f"{type(exc).__name__}: {exc}"))

    test_result = pd.DataFrame()
    if selected_name is not None:
        try:
            y_pred = fitted[selected_name].predict(X_test)
            test_result = pd.DataFrame([{
                "Selected Algorithm": selected_name,
                "Selection CV Macro-F1": float(selection_frame.loc[
                    selection_frame["Algorithm"].eq(selected_name), "CV Macro-F1 Mean"].iloc[0]),
                "Held-out Test Accuracy": float(accuracy_score(y_test, y_pred)),
                "Held-out Test Macro-F1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "Training seconds": elapsed,
            }])
        except Exception as exc:
            failures.append((selected_name, "held-out evaluation", f"{type(exc).__name__}: {exc}"))
    return CandidateEvaluation(
        selection_frame, test_result, selected_name, fitted, tuple(failures),
        pd.DataFrame(class_rows, columns=["Algorithm", "Class", "Precision", "Recall", "F1", "Support"]),
        pd.DataFrame(error_rows, columns=["Algorithm", "True class", "Predicted class", "Count"]),
        pd.DataFrame(boundary_rows), cv_source_rows, cv_rows, n_splits,
    )
