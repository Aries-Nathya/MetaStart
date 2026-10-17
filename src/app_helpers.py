"""Presentation helpers for the Streamlit application."""

from __future__ import annotations

import pandas as pd

from src.benchmark import ALGORITHM_NAMES
from src.dataset_profiler import META_FEATURES


PRODUCTION_PROFILE_LABELS = {
    "n_rows": "Training rows",
    "n_features": "Features",
    "n_numeric": "Numeric features",
    "n_categorical": "Categorical features",
    "missing_ratio": "Missing-value ratio",
    "n_classes": "Target classes",
    "imbalance_ratio": "Class imbalance ratio",
    "mean_feature_mean": "Mean numeric feature mean",
    "mean_feature_std": "Mean numeric feature standard deviation",
}


def production_profile_table(profile: dict[str, float]) -> pd.DataFrame:
    """Build the profile table for model A without exposing experimental fields."""
    missing = [feature for feature in META_FEATURES if feature not in profile]
    if missing:
        raise ValueError(f"Profile is missing production model features: {missing}")
    return pd.DataFrame({
        "Characteristic": [PRODUCTION_PROFILE_LABELS[feature] for feature in META_FEATURES],
        "Value": [profile[feature] for feature in META_FEATURES],
    })


CV_CLOSE_TOLERANCE = 0.01

ALGORITHM_GUIDANCE = {
    "Logistic Regression": (
        "Fast, relatively compact linear baseline; numeric scaling is applied by the pipeline.",
        "A linear boundary may miss nonlinear relationships; in the current candidate setup it uses class balancing.",
    ),
    "Decision Tree": (
        "Can represent nonlinear splits and does not rely on numeric feature scaling.",
        "A single tree can be sensitive to the sample and may overfit, especially as feature count grows.",
    ),
    "Random Forest": (
        "Ensemble of trees can capture nonlinear patterns and feature interactions; scaling is not needed by the trees.",
        "Less directly interpretable than one tree and generally more computational work than a linear model.",
    ),
    "SVM": (
        "The RBF SVM can represent nonlinear decision boundaries; the pipeline scales numeric features.",
        "Can require more computation as row count grows and is sensitive to feature scaling; categories become one-hot columns.",
    ),
    "k-Nearest Neighbors": (
        "Can model local, nonlinear class structure without fitting a complex global boundary.",
        "Distance calculations are sensitive to scale and irrelevant dimensions; prediction cost grows with training-set size. The current configuration does not use class weights.",
    ),
}


def _fmt_profile_number(profile: dict[str, float], key: str, digits: int = 0) -> str:
    value = profile.get(key)
    if value is None or not pd.notna(value):
        return "unavailable"
    return f"{value:,.{digits}f}"


def _dataset_evidence(profile: dict[str, float]) -> str:
    rows = _fmt_profile_number(profile, "n_rows")
    features = _fmt_profile_number(profile, "n_features")
    classes = _fmt_profile_number(profile, "n_classes")
    numeric = _fmt_profile_number(profile, "n_numeric")
    categorical = _fmt_profile_number(profile, "n_categorical")
    missing = profile.get("missing_ratio")
    imbalance = profile.get("imbalance_ratio")
    missing_text = f"{missing:.1%} missing feature cells" if missing is not None and pd.notna(missing) else "missingness unavailable"
    imbalance_text = (
        f"largest-to-smallest class count ratio {imbalance:.2f}"
        if imbalance is not None and pd.notna(imbalance) else "class balance unavailable"
    )
    return (
        f"Training partition: {rows} rows, {features} features, {classes} classes; "
        f"{numeric} numeric and {categorical} categorical features; {missing_text}; {imbalance_text}."
    )


def _suitability_note(algorithm: str, profile: dict[str, float]) -> str:
    rows = profile.get("n_rows")
    features = profile.get("n_features")
    numeric = profile.get("n_numeric", 0.0)
    categorical = profile.get("n_categorical", 0.0)
    missing = profile.get("missing_ratio", 0.0)
    imbalance = profile.get("imbalance_ratio", 1.0)
    evidence = _dataset_evidence(profile)
    preprocessing = []
    if numeric:
        preprocessing.append("numeric fields are median-imputed and standardized")
    if categorical:
        preprocessing.append("categorical fields are most-frequent-imputed and one-hot encoded")
    if not preprocessing:
        preprocessing.append("no numeric or categorical preprocessing fields were identified")
    prep_text = "; ".join(preprocessing)

    if algorithm == "Logistic Regression":
        context = "A linear boundary is a useful low-cost comparison for this feature representation; the profile alone cannot show whether the classes are linearly separable."
    elif algorithm == "Decision Tree":
        context = "Nonlinear feature splits can use the observed feature set without scaling; this profile does not establish whether a single tree will generalize well."
    elif algorithm == "Random Forest":
        context = "Tree ensembles can use nonlinear interactions among these features; their added flexibility should be judged by the training-only CV score."
    elif algorithm == "SVM":
        context = "The numeric scaling step is relevant to SVM distance and margin calculations; larger row counts can increase fit cost."
    else:
        context = "Scaled numeric values and one-hot categories form the distance space; many features can make neighborhood distances less informative."

    if rows is not None and pd.notna(rows) and algorithm in {"SVM", "k-Nearest Neighbors", "Random Forest"}:
        if rows > 5000:
            context += " The training partition is large enough that runtime is a consideration."
        elif rows <= 500:
            context += " The training partition is small, so CV estimates may vary with individual examples."
    if features is not None and pd.notna(features) and features > 50 and algorithm == "k-Nearest Neighbors":
        context += " The feature dimension is high for a distance-based method."
    if missing and missing > 0:
        context += " The measured missing cells are handled by the pipeline imputers."
    if categorical:
        context += " Categorical fields are represented through one-hot encoding in the same pipeline."
    if imbalance is not None and imbalance >= 3:
        if algorithm == "k-Nearest Neighbors":
            context += f" The observed imbalance ratio is {imbalance:.2f}; this candidate has no class_weight setting in the current configuration."
        elif algorithm != "Random Forest":
            context += f" The observed imbalance ratio is {imbalance:.2f}; this candidate is configured with class balancing."
        else:
            context += f" The observed imbalance ratio is {imbalance:.2f}; this Random Forest is configured with class balancing."
    shape = (
        f"Training-profile diagnostics: median absolute numeric skewness "
        f"{_fmt_profile_number(profile, 'median_abs_numeric_skewness', 3)}, median absolute excess "
        f"kurtosis {_fmt_profile_number(profile, 'median_abs_numeric_excess_kurtosis', 3)}, mean absolute "
        f"numeric correlation {_fmt_profile_number(profile, 'mean_abs_numeric_correlation', 3)}, and "
        f"high-correlation pair fraction {_fmt_profile_number(profile, 'fraction_numeric_pairs_abs_corr_ge_0_8', 3)}. "
        "These are descriptive feature summaries; they do not establish class separability."
    )
    if categorical:
        shape += (
            f" Categorical diagnostics: mean normalized entropy "
            f"{_fmt_profile_number(profile, 'mean_normalized_categorical_entropy', 3)} and mean cardinality "
            f"{_fmt_profile_number(profile, 'mean_categorical_cardinality', 2)}."
        )
    return f"{evidence} {context} {shape} Pipeline: {prep_text}."


def dataset_diagnostics_table(profile: dict[str, float], y_train: pd.Series) -> pd.DataFrame:
    """Display measured training-partition characteristics and class proportions."""
    descriptions = [
        ("Rows", "n_rows", 0), ("Features", "n_features", 0),
        ("Classes", "n_classes", 0), ("Numeric features", "n_numeric", 0),
        ("Categorical features", "n_categorical", 0),
        ("Missing feature cells", "missing_ratio", 3),
        ("Largest/smallest class ratio", "imbalance_ratio", 2),
        ("Normalized class entropy", "normalized_class_entropy", 3),
        ("Median absolute numeric skewness", "median_abs_numeric_skewness", 3),
        ("Median absolute excess kurtosis", "median_abs_numeric_excess_kurtosis", 3),
        ("Mean absolute numeric correlation", "mean_abs_numeric_correlation", 3),
        ("Numeric pairs with |correlation| ≥ 0.8", "fraction_numeric_pairs_abs_corr_ge_0_8", 3),
        ("Mean normalized categorical entropy", "mean_normalized_categorical_entropy", 3),
        ("Mean categorical cardinality", "mean_categorical_cardinality", 2),
    ]
    rows = [{"Characteristic": label,
             "Value": _fmt_profile_number(profile, key, digits)}
            for label, key, digits in descriptions]
    counts = pd.Series(y_train).value_counts(dropna=False)
    for label, count in counts.items():
        rows.append({"Characteristic": f"Class {label} (count; share)",
                     "Value": f"{int(count):,}; {count / len(y_train):.1%}"})
    return pd.DataFrame(rows)


def feature_structure_tables(X_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize per-feature distributions and strongest numeric correlations."""
    feature_rows = []
    numeric = X_train.select_dtypes(include="number").replace([float("inf"), -float("inf")], float("nan"))
    categorical = X_train.select_dtypes(exclude="number")
    for column in X_train.columns:
        series = X_train[column]
        missing = float(series.isna().mean())
        if column in numeric:
            valid = numeric[column].dropna()
            feature_rows.append({
                "Feature": str(column), "Type": "Numeric", "Missing share": missing,
                "Distinct values": int(valid.nunique()),
                "Mean": float(valid.mean()) if len(valid) else None,
                "Median": float(valid.median()) if len(valid) else None,
                "Standard deviation": float(valid.std()) if len(valid) > 1 else 0.0,
                "Skewness": float(valid.skew()) if len(valid) > 2 else None,
                "Excess kurtosis": float(valid.kurt()) if len(valid) > 3 else None,
            })
        else:
            valid = categorical[column].dropna()
            feature_rows.append({
                "Feature": str(column), "Type": "Categorical", "Missing share": missing,
                "Distinct values": int(valid.nunique()), "Mean": None, "Median": None,
                "Standard deviation": None, "Skewness": None, "Excess kurtosis": None,
            })
    correlations = numeric.corr(method="pearson").abs()
    correlation_rows = []
    if correlations.shape[1] > 1:
        for left_index, left in enumerate(correlations.columns):
            for right in correlations.columns[left_index + 1:]:
                value = correlations.loc[left, right]
                if pd.notna(value):
                    correlation_rows.append({"Feature 1": str(left), "Feature 2": str(right),
                                             "Absolute Pearson correlation": float(value)})
    correlation_table = pd.DataFrame(correlation_rows, columns=[
        "Feature 1", "Feature 2", "Absolute Pearson correlation",
    ]).sort_values("Absolute Pearson correlation", ascending=False, kind="stable").head(10)
    return pd.DataFrame(feature_rows), correlation_table


def build_algorithm_comparison(
        ranking: pd.DataFrame,
        selection_results: pd.DataFrame,
        profile: dict[str, float],
        failures: tuple[tuple[str, str, str], ...] = (),
        selected_algorithm: str | None = None,
        full_training_seconds: float | None = None,
        evaluation_error: str | None = None,
        class_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Combine predictions, training-only CV evidence, status, and fixed guidance."""
    if not {"algorithm", "predicted_macro_f1"}.issubset(ranking.columns):
        raise ValueError("Ranking requires algorithm and predicted_macro_f1 columns.")
    if (len(ranking) != len(ALGORITHM_NAMES) or ranking["algorithm"].duplicated().any()
            or set(ranking["algorithm"]) != set(ALGORITHM_NAMES)):
        raise ValueError("Comparison requires one ranking row for each of the five supported algorithms.")
    ranked = ranking.sort_values("predicted_macro_f1", ascending=False, kind="stable").reset_index(drop=True)
    cv_by_algorithm = (
        selection_results.set_index("Algorithm").to_dict("index")
        if not selection_results.empty and "Algorithm" in selection_results else {}
    )
    failures_by_algorithm: dict[str, list[tuple[str, str]]] = {}
    for name, phase, message in failures:
        failures_by_algorithm.setdefault(name, []).append((phase, message))

    records = []
    for rank, row in ranked.iterrows():
        name = row["algorithm"]
        cv = cv_by_algorithm.get(name)
        problem_list = failures_by_algorithm.get(name, [])
        cv_problem = next((message for phase, message in problem_list if phase == "cross-validation"), None)
        fit_problem = next((message for phase, message in problem_list if phase == "final fit"), None)
        resource_problem = next((message for phase, message in problem_list if phase == "resource limit"), None)
        heldout_problem = next((message for phase, message in problem_list if phase == "held-out evaluation"), None)
        if cv is None and evaluation_error:
            cv_status = f"CV unavailable: {evaluation_error}"
        elif cv is None and cv_problem:
            cv_status = f"CV failed: {cv_problem}"
        elif cv is None:
            cv_status = "Not evaluated: CV result is unavailable; measured performance is unknown."
        elif heldout_problem:
            cv_status = f"CV completed; selected-model held-out evaluation failed: {heldout_problem}"
        elif fit_problem:
            cv_status = f"CV completed; full-training fit failed and fallback was used: {fit_problem}"
        elif resource_problem:
            cv_status = f"CV completed; full-training refit skipped by resource limit: {resource_problem}"
        else:
            cv_status = "CV completed."
        cv_value = cv.get("CV Macro-F1 Mean") if cv is not None else None
        cv_std = cv.get("CV Macro-F1 Std") if cv is not None else None
        cv_fit_time = cv.get("CV Fit Seconds Mean") if cv is not None else None
        is_selected = name == selected_algorithm
        records.append({
            "Predicted rank": rank + 1,
            "Algorithm": name,
            "Predicted Macro-F1": float(row["predicted_macro_f1"]),
            "Mean training-only CV Macro-F1": cv_value,
            "CV Macro-F1 standard deviation": cv_std,
            "CV mean fit seconds per fold": cv_fit_time,
            "Selected full-training fit seconds": full_training_seconds if is_selected else None,
            "CV status": cv_status,
            "Selected by CV rule": "Yes" if is_selected else "No",
            "General strengths": ALGORITHM_GUIDANCE[name][0],
            "General trade-offs": ALGORITHM_GUIDANCE[name][1],
            "Dataset-specific suitability evidence": _suitability_note(name, profile),
        })
    if not selection_results.empty:
        by_algorithm = selection_results.set_index("Algorithm")
        for record in records:
            name = record["Algorithm"]
            if name not in by_algorithm.index:
                record["Dataset-specific suitability evidence"] += " This candidate has no successful CV result, so its dataset-specific performance is unknown."
                continue
            score = float(by_algorithm.loc[name, "CV Macro-F1 Mean"])
            std = float(by_algorithm.loc[name, "CV Macro-F1 Std"])
            record["Dataset-specific suitability evidence"] += (
                f" On this training-only CV, observed mean Macro-F1 was {score:.3f} "
                f"(fold standard deviation {std:.3f}); the standard deviation describes fold variation, "
                "not a significance test."
            )
            if selected_algorithm in by_algorithm.index and name != selected_algorithm:
                winner_score = float(by_algorithm.loc[selected_algorithm, "CV Macro-F1 Mean"])
                delta = winner_score - score
                record["Dataset-specific suitability evidence"] += (
                    f" Difference from selected {selected_algorithm}: {delta:+.3f} Macro-F1 "
                    "(selected minus this candidate)."
                )
            if class_metrics is not None and not class_metrics.empty:
                rows = class_metrics[class_metrics["Algorithm"].eq(name)]
                if not rows.empty:
                    weakest = rows.sort_values("Recall", kind="stable").iloc[0]
                    record["Dataset-specific suitability evidence"] += (
                        f" Lowest out-of-fold class recall was {float(weakest['Recall']):.3f} "
                        f"for class {weakest['Class']} (support {int(weakest['Support'])}); "
                        "this points to a class-specific error pattern in the CV sample."
                    )
    return pd.DataFrame(records)


def build_selection_rationale(
        selection_results: pd.DataFrame,
        failures: tuple[tuple[str, str, str], ...],
        selected_algorithm: str | None,
        prediction_ranks: pd.DataFrame,
        evaluation_error: str | None = None,
        boundary_diagnostic: pd.DataFrame | None = None) -> dict[str, str | list[str]]:
    """Explain the actual CV selection/fallback rule without consulting test scores."""
    cv_by_algorithm = (
        selection_results.set_index("Algorithm").to_dict("index")
        if not selection_results.empty and "Algorithm" in selection_results else {}
    )
    sorted_predictions = prediction_ranks.sort_values(
        "predicted_macro_f1", ascending=False, kind="stable").reset_index(drop=True)
    predicted = pd.DataFrame({
        "Predicted rank": range(1, len(sorted_predictions) + 1),
        "Predicted Macro-F1": sorted_predictions["predicted_macro_f1"].to_numpy(),
        "algorithm": sorted_predictions["algorithm"].to_numpy(),
    }).set_index("algorithm")
    failures_by_algorithm: dict[str, list[tuple[str, str]]] = {}
    for name, phase, message in failures:
        failures_by_algorithm.setdefault(name, []).append((phase, message))
    if selected_algorithm is None:
        if evaluation_error:
            why_selected = (
                f"Candidate evaluation was unavailable: {evaluation_error}. No candidate was selected; "
                "held-out test metrics were not used to choose a model."
            )
        elif cv_by_algorithm:
            why_selected = (
                f"Training-only CV completed for {len(cv_by_algorithm)} candidate(s), but none completed a usable "
                "full-training fit. No candidate was selected and no held-out test metrics were used to choose one."
            )
        else:
            why_selected = (
                "No candidate completed training-only CV, so no candidate was selected. "
                "Held-out test metrics were not used to choose a model."
            )
    else:
        selected_cv = cv_by_algorithm.get(selected_algorithm)
        if selected_cv is None:
            why_selected = (
                f"{selected_algorithm} is the returned fitted candidate, but its CV score is unavailable. "
                "Selection could not be explained from CV results; test metrics were not used for selection."
            )
        else:
            score = float(selected_cv["CV Macro-F1 Mean"])
            higher = [
                (name, float(values["CV Macro-F1 Mean"]))
                for name, values in cv_by_algorithm.items()
                if float(values["CV Macro-F1 Mean"]) > score and name != selected_algorithm
            ]
            predicted_rank = int(predicted.loc[selected_algorithm, "Predicted rank"])
            predicted_score = float(predicted.loc[selected_algorithm, "Predicted Macro-F1"])
            successful_scores = [float(values["CV Macro-F1 Mean"]) for values in cv_by_algorithm.values()]
            nearest_gap = (score - sorted(successful_scores, reverse=True)[1]
                           if len(successful_scores) > 1 and score == max(successful_scores) else 0.0)
            winner_std = float(selected_cv.get("CV Macro-F1 Std", 0.0))
            closeness = (
                f" Its gap over the next-highest CV mean was {nearest_gap:.3f}; "
                "the observed fold standard deviation is descriptive and is not a significance test."
            )
            if not higher:
                why_selected = (
                    f"{selected_algorithm} was selected because it had the highest available mean training-only "
                    f"CV Macro-F1 ({score:.3f}, fold standard deviation {winner_std:.3f}) among the evaluated candidates "
                    f"and its full-training fit succeeded.{closeness} "
                    f"Production model A predicted rank {predicted_rank} (predicted Macro-F1 {predicted_score:.3f}); "
                    "that prediction ranked recommendations but did not determine measured-model selection."
                )
            else:
                failed_higher = [
                    (name, value) for name, value in higher
                    if any(phase in {"final fit", "resource limit"}
                           for phase, _ in failures_by_algorithm.get(name, []))
                ]
                if failed_higher:
                    failures_text = "; ".join(
                        f"{name} (CV {value:.3f}, {value - score:+.3f} vs selected) could not be used for full-data evaluation"
                        for name, value in failed_higher
                    )
                    why_selected = (
                        f"{selected_algorithm} (MetaStart predicted rank {predicted_rank}, predicted Macro-F1 "
                        f"{predicted_score:.3f}) was selected with mean training-only CV Macro-F1 {score:.3f} "
                        f"after higher-scoring candidate(s) {failures_text}. The workflow falls back in descending "
                        "CV order when a full-training fit fails. MetaStart's predicted rank did not select the measured model."
                    )
                else:
                    why_selected = (
                        f"{selected_algorithm} was returned with mean training-only CV Macro-F1 {score:.3f}, "
                        "but another evaluated CV result was higher. The available status does not explain that ordering; "
                        "no test metrics were used in this rationale."
                    )

        heldout_problem = next((
            message for phase, message in failures_by_algorithm.get(selected_algorithm, [])
            if phase == "held-out evaluation"
        ), None)
        if heldout_problem:
            why_selected += (
                f" Its held-out evaluation later failed ({heldout_problem}); this failure did not affect CV selection."
            )

    why_not = []
    for name in ALGORITHM_NAMES:
        if name == selected_algorithm:
            continue
        cv = cv_by_algorithm.get(name)
        problems = failures_by_algorithm.get(name, [])
        cv_failure = next((message for phase, message in problems if phase == "cross-validation"), None)
        fit_failure = next((message for phase, message in problems if phase == "final fit"), None)
        resource_failure = next((message for phase, message in problems if phase == "resource limit"), None)
        if cv_failure:
            reason = f"{name}: CV result unavailable because cross-validation failed ({cv_failure}); no score-based comparison is justified."
        elif cv is None and evaluation_error:
            reason = f"{name}: CV score unavailable because candidate evaluation did not complete ({evaluation_error}); no score-based comparison is justified."
        elif cv is None:
            reason = f"{name}: no CV score is available; its measured performance is unknown and cannot be compared here."
        elif fit_failure:
            reason = f"{name}: mean training-only CV Macro-F1 was {float(cv['CV Macro-F1 Mean']):.3f}, but its full-training fit failed ({fit_failure}); the workflow continued to the next candidate by CV order."
        elif resource_failure:
            reason = (f"{name}: mean training-only CV Macro-F1 was {float(cv['CV Macro-F1 Mean']):.3f}, "
                      f"but full-data fitting or prediction exceeded the documented resource guard ({resource_failure}); "
                      "the workflow continued to the next candidate by CV order.")
        elif selected_algorithm is None:
            reason = f"{name}: CV mean was {float(cv['CV Macro-F1 Mean']):.3f}, but no candidate completed a usable final fit."
        else:
            delta = float(cv_by_algorithm[selected_algorithm]["CV Macro-F1 Mean"]) - float(cv["CV Macro-F1 Mean"])
            if abs(delta) <= CV_CLOSE_TOLERANCE:
                reason = f"{name}: CV mean was {float(cv['CV Macro-F1 Mean']):.3f}, within {CV_CLOSE_TOLERANCE:.3f} Macro-F1 of the selected candidate; this small observed difference does not establish that it is meaningfully worse."
            elif delta > 0:
                reason = f"{name}: CV mean was lower than the selected candidate by {delta:.3f} Macro-F1 ({float(cv['CV Macro-F1 Mean']):.3f} vs {float(cv_by_algorithm[selected_algorithm]['CV Macro-F1 Mean']):.3f}); this is an observed score difference, not a significance test."
            else:
                reason = f"{name}: CV mean was higher by {-delta:.3f} Macro-F1, but the CV-order fallback did not return it. Inspect its CV/final-fit status above; held-out scores are not used to explain the selection."
        if cv is not None and selected_algorithm in cv_by_algorithm:
            selected_row = cv_by_algorithm[selected_algorithm]
            candidate_score = float(cv["CV Macro-F1 Mean"])
            winner_score = float(selected_row["CV Macro-F1 Mean"])
            time_note = ""
            if "CV Fit Seconds Mean" in cv and "CV Fit Seconds Mean" in selected_row:
                difference = float(cv["CV Fit Seconds Mean"]) - float(selected_row["CV Fit Seconds Mean"])
                time_note = f" Its mean CV fit time was {difference:+.3f}s per fold relative to the selected candidate."
            reason += time_note
        why_not.append(f"{reason} General trade-off (context only): {ALGORITHM_GUIDANCE[name][1]}")

    if boundary_diagnostic is not None and not boundary_diagnostic.empty:
        row = boundary_diagnostic.iloc[0]
        if pd.notna(row.get("CV Macro-F1 Mean")) and "SVM" in cv_by_algorithm:
            rbf = float(cv_by_algorithm["SVM"]["CV Macro-F1 Mean"])
            linear = float(row["CV Macro-F1 Mean"])
            why_selected += (
                f" A diagnostic-only linear-kernel SVM scored {linear:.3f} versus the candidate RBF SVM "
                f"at {rbf:.3f} on the same training-only folds (linear minus RBF {linear-rbf:+.3f}). "
                "This is evidence about these two configured pipelines on these folds, not proof of linear separability; "
                "the diagnostic did not participate in selection."
            )
        else:
            why_selected += " The linear-versus-RBF diagnostic was unavailable, so no boundary-shape conclusion is supported."
    return {"selected": why_selected, "alternatives": why_not}
