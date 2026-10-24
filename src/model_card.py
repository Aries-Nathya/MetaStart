"""Build a model card from the current MetaStart experiment records."""

from __future__ import annotations

from datetime import date
import json
from typing import Any, Iterable, Mapping

from src.experiment_results import ExperimentRecord

MODEL_CARD_MIME_TYPE = "text/markdown"


def _display(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "Not measured"
    try:
        if bool(value != value):  # NaN and pandas.NA
            return "Not measured"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float,)):
        return f"{value:.{digits}f}"
    return str(value)


def _find(records: list[ExperimentRecord], stage: str, algorithm: str | None,
          *, completed_only: bool = True) -> ExperimentRecord | None:
    if not algorithm:
        return None
    return next((record for record in records
                 if record.stage == stage and record.algorithm == algorithm
                 and not record.invalid_for_real_world
                 and (not completed_only or record.status == "completed")), None)


def _metric(record: ExperimentRecord | None, name: str) -> str:
    if record is None:
        return "Not measured"
    value = record.metrics.get(name)
    if value is None and name == record.primary_metric:
        value = record.primary_value
    return _display(value)


def _class_table(record: ExperimentRecord | None) -> str:
    if record is None or not record.class_breakdown:
        return "Not available in this run."
    output = ["| Class | Precision | Recall | F1 | Support |",
              "|---|---:|---:|---:|---:|"]
    for item in record.class_breakdown:
        output.append(
            f"| {item.get('Class', 'Unknown')} | {_display(item.get('Precision'))} "
            f"| {_display(item.get('Recall'))} | {_display(item.get('F1'))} "
            f"| {_display(item.get('Support'), digits=0)} |"
        )
    return "\n".join(output)


def _parameters_block(parameters: Any) -> str:
    if parameters is None:
        return "Not available in this run."
    return "```json\n" + json.dumps(parameters, indent=2, sort_keys=True, default=str) + "\n```"


def _feature_findings(record: ExperimentRecord | None) -> str:
    if record is None:
        return "Not run in this session."
    details = record.leaderboard_metrics or {}
    features = details.get("feature_importance", [])
    if record.status != "completed" or not features:
        reason = "; ".join(record.warnings) or "Importance values were not recorded."
        return f"Status: `{record.status}`. {reason}"
    lines = [
        f"Method: {details.get('method', record.method)}; "
        f"sample size: {_display(details.get('sample_count'), digits=0)}; "
        f"runtime: {_display(details.get('runtime_seconds'))} seconds.",
        "",
        "Top measured features (Macro-F1 decrease after permutation):",
    ]
    lines.extend(
        f"- `{item.get('Feature', 'Unknown')}`: {_display(item.get('Importance Mean'))} "
        f"(SD {_display(item.get('Importance Std'))})"
        for item in features[:10]
    )
    if record.warnings:
        lines.extend(["", "Warnings: " + "; ".join(record.warnings)])
    return "\n".join(lines)


def _local_findings(record: ExperimentRecord | None) -> str:
    if record is None:
        return "Not run in this session."
    details = record.leaderboard_metrics or {}
    effects = details.get("local_feature_effects", [])
    if record.status != "completed":
        return f"Status: `{record.status}`. " + ("; ".join(record.warnings) or "No result recorded.")
    lines = [
        f"Method: {details.get('method', record.method)}; one prediction; "
        f"runtime: {_display(details.get('runtime_seconds'))} seconds; "
        f"score type: {details.get('score_type', 'Not available')}.",
        "",
    ]
    if effects:
        lines.append("Largest measured feature-replacement sensitivities:")
        lines.extend(
            f"- `{item.get('Feature', 'Unknown')}`: score change "
            f"{_display(item.get('Score change after replacement'))}"
            for item in effects[:5]
        )
    else:
        lines.append("No local feature effects were recorded.")
    return "\n".join(lines)


def _optional_experiments(records: list[ExperimentRecord], stages: set[str],
                          *, framework_only: bool = False) -> str:
    selected = [record for record in records
                if record.stage in stages and not record.invalid_for_real_world]
    if framework_only:
        selected = [record for record in selected
                    if record.algorithm in {"FLAML", "AutoGluon"}
                    or (record.stage == "external_automl_held_out_test"
                        and record.algorithm.startswith(("FLAML:", "AutoGluon:")))]
    if not selected:
        return "Not run in this session."
    rows = ["| Stage | Method / algorithm | Status | Macro-F1 | Runtime (s) | Rows / folds | Budget / trials |",
            "|---|---|---|---:|---:|---:|---:|"]
    for record in selected:
        score = record.metrics.get("macro_f1")
        runtime = record.fit_seconds
        sample = (f"{record.cv_rows or 'Not measured'} / {record.cv_folds or 'Not measured'}"
                  if record.stage not in {"held_out_test", "external_automl_held_out_test"}
                  else _display(record.test_rows, digits=0))
        budget = record.leaderboard_metrics.get("time_budget_seconds")
        budget_text = (_display(budget, digits=0) if budget is not None
                       else _display(record.trials_evaluated, digits=0))
        rows.append(
            f"| {record.stage} | {record.method} / {record.algorithm} | `{record.status}` "
            f"| {_display(score)} | {_display(runtime)} | {sample} | {budget_text} |"
        )
    return "\n".join(rows)


def build_model_card(records: Iterable[ExperimentRecord],
                     metadata: Mapping[str, Any] | None = None) -> str:
    """Render Markdown from current-run records and supplied run metadata.

    Leakage-stage records are never used for production metrics. Every selected
    score comes from its explicitly named record stage; missing values are shown
    as ``Not measured`` instead of being inferred or substituted.
    """
    records = list(records)
    metadata = dict(metadata or {})
    selected = metadata.get("selected_algorithm")
    recommendation = _find(records, "recommendation", selected)
    cv = _find(records, "cv", selected)
    heldout = _find(records, "held_out_test", selected)
    bias = _find(records, "bias", selected) or cv
    global_record = next((record for record in records
                          if record.stage == "explainability"
                          and record.algorithm == selected
                          and record.method == "permutation_importance"), None)
    local_record = next((record for record in records
                         if record.stage == "explainability"
                         and record.algorithm == selected
                         and record.method == "single_feature_ablation_sensitivity"), None)

    distribution = metadata.get("class_distribution")
    if distribution:
        class_lines = []
        for item in distribution:
            proportion = item.get("proportion")
            share = f" ({float(proportion):.1%})" if proportion is not None else " (proportion not measured)"
            class_lines.append(f"- `{item.get('class', 'Unknown')}`: {item.get('count', 'Not measured')}{share}")
        distribution_text = "\n".join(class_lines)
    else:
        distribution_text = "Not available in this run."

    ratio = metadata.get("imbalance_ratio")
    imbalance_threshold = metadata.get("imbalance_ratio_threshold", 3.0)
    if ratio is None:
        imbalance = f"Not measured; this app flags largest/smallest class ratios at or above {imbalance_threshold:g}:1."
    else:
        flag = metadata.get("imbalance_flagged")
        result = "flagged" if flag is True else "not flagged" if flag is False else "status not recorded"
        imbalance = (f"Training-partition largest/smallest class ratio: {_display(ratio, digits=2)}:1; "
                     f"{result} at the {imbalance_threshold:g}:1 criterion.")

    external_text = _optional_experiments(
        records, {"external_automl", "external_automl_held_out_test"}, framework_only=True,
    )
    tuning_text = _optional_experiments(
        records, {"manual_baseline", "grid_search", "random_search", "optuna"},
    )
    leakage_present = any(record.stage == "leakage" for record in records)
    leakage_note = (
        "A separate leakage demonstration is included in experiment exports. It is excluded from all model-card "
        "performance metrics and is **INVALID FOR REAL-WORLD PERFORMANCE — DELIBERATE LEAKAGE DEMONSTRATION**."
        if leakage_present else "No leakage experiment was included in this run's records."
    )
    parameters = metadata.get("selected_model_parameters")
    cv_folds = cv.cv_folds if cv else metadata.get("cv_folds")
    cv_rows = cv.cv_rows if cv else metadata.get("cv_rows")
    cv_source_rows = cv.cv_source_rows if cv else metadata.get("cv_source_rows")
    seed = cv.seed if cv else metadata.get("random_seed")
    fit_runtime = (metadata.get("training_seconds") if metadata.get("training_seconds") is not None
                   else heldout.fit_seconds if heldout else None)
    status = "completed" if recommendation or cv or heldout else "No selected model record available"

    return f"""# Model Card — {metadata.get('model_name', 'MetaStart selected classification model')}

- **Model version:** {metadata.get('model_version', 'Not separately versioned')}
- **Created:** {metadata.get('card_created_at', date.today().isoformat())}
- **Application:** {metadata.get('application_version', 'MetaStart IA-3 research prototype; version not separately tagged')}
- **Current run ID:** {metadata.get('run_id', 'Not available')}
- **Selected model status:** {status}

## Purpose and intended users

Intended for exploratory classification research and educational use by project evaluators and data-science users. It is not validated for clinical, safety-critical, or other high-impact decisions and is not presented as a deployed service.

## Dataset and preprocessing

- **Dataset name:** {metadata.get('dataset_name', 'Not available')}
- **Source:** {metadata.get('dataset_source', 'Not available')}
- **Rows × features:** {_display(metadata.get('dataset_rows'), digits=0)} × {_display(metadata.get('dataset_features'), digits=0)}
- **Target column:** {metadata.get('target', 'Not available')}
- **Class distribution (training partition):**
{distribution_text}
- **Imbalance finding:** {imbalance}
- **Preprocessing:** numeric median imputation and standard scaling; categorical most-frequent imputation and one-hot encoding (`handle_unknown=ignore`); transformations are fitted within the model pipeline.
- **Dataset provenance limitation:** user-uploaded filenames are recorded, but their original source/license is not independently verified.

## Split and reproducibility

- **Train/test strategy:** {metadata.get('split_strategy', 'Stratified 75/25 split, seed 42')}
- **Training / held-out rows:** {_display(metadata.get('training_rows'), digits=0)} / {_display(metadata.get('held_out_rows'), digits=0)}
- **CV:** {_display(cv_folds, digits=0)} stratified folds; {_display(cv_rows, digits=0)} sampled training rows from {_display(cv_source_rows, digits=0)}; seed {_display(seed, digits=0)}
- **Environment:** {metadata.get('environment', 'Not recorded')}
- **Sampling limits:** {metadata.get('sampling_limits', 'CV uses a deterministic stratified sample capped at 5,000 training rows; larger uploads are subject to application resource guards.')}

## Selected model and fitted configuration

- **Model type:** Supervised classification pipeline
- **Algorithm:** {selected or 'Not selected'}
- **Fitted estimator parameters:**

{_parameters_block(parameters)}

## MetaStart recommendation — predicted, not measured

- **Predicted Macro-F1:** {_metric(recommendation, 'macro_f1')}
- **Status:** `{recommendation.status if recommendation else 'Not available'}`

This value is the meta-recommender's estimate. It is not the selected model's measured cross-validation or held-out performance.

## Selected model cross-validation — training data only

- **Measured CV Macro-F1 mean:** {_metric(cv, 'macro_f1')}
- **CV Macro-F1 standard deviation:** {_display(cv.cv_macro_f1_std if cv else None)}
- **Mean fit runtime per fold:** {_display(cv.cv_fit_seconds_mean if cv else None)} seconds
- **Selected model full-training fit runtime:** {_display(fit_runtime)} seconds
- **Per-class out-of-fold performance:**

{_class_table(bias)}

## Selected model held-out evaluation

- **Held-out accuracy:** {_metric(heldout, 'accuracy')}
- **Held-out Macro-F1:** {_metric(heldout, 'macro_f1')}
- **Held-out rows:** {_display(heldout.test_rows if heldout else metadata.get('held_out_rows'), digits=0)}
- **Held-out prediction runtime:** Not measured by the current candidate-evaluation workflow.

The held-out partition is evaluated after CV selection and does not select or tune the model.

## Explainability

**Global explanation — permutation importance:** {_feature_findings(global_record)}

**Individual prediction — feature replacement sensitivity:** {_local_findings(local_record)}

These methods report model association or sensitivity, not causation. The global importance sample comes from training data used to fit the selected model and may overstate generalization. SHAP was not used for this explanation run.

## Optional tuning experiments

{tuning_text}

## External AutoML comparisons (not the selected production model)

{external_text}

## Limitations, bias, and resource constraints

- Class imbalance and class-wise metrics describe outcome classes; they do not establish demographic or group fairness.
- Sensitive attributes and appropriate subgroup evidence were not supplied. Demographic fairness is therefore not established, and target classes are not assumed to be protected groups.
- This research prototype has finite benchmark coverage. A benchmark score does not guarantee performance on another dataset, population, or deployment setting.
- Upload size, feature count, full-fit, and CV sampling safeguards apply. Limits for this run: {_display(metadata.get('resource_limits'))}.
- Hyperparameter search and optional external AutoML outcomes are training-partition experiments; they do not replace the selected model's separate held-out evaluation.
- No model card claims clinical validation, deployment approval, or broad real-world generalization.

## Leakage integrity

{leakage_note}

**Benchmark performance does not guarantee real-world performance.**
"""
