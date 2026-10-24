"""Streamlit interface for the MetaStart classification prototype."""

from __future__ import annotations

import hashlib
import platform
from io import BytesIO

import joblib
import pandas as pd
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split

from src.app_helpers import (build_algorithm_comparison, build_selection_rationale,
                             dataset_diagnostics_table, feature_structure_tables,
                             production_profile_table)
from src.app_workflow import (CV_MAX_ROWS, MAX_FULL_FIT_ROWS, MAX_UPLOAD_BYTES,
                              MAX_UPLOAD_CELLS, MAX_UPLOAD_FEATURES, MAX_UPLOAD_ROWS,
                              UploadValidationError, evaluate_ranked_candidates,
                              load_uploaded_csv, prepare_classification_data,
                              recommendation_download_table)
from src.benchmark import ALGORITHM_NAMES, build_pipeline, get_models
from src.advanced_experiments import run_tuning_experiments
from src.bias_explainability import (IMBALANCE_RATIO_THRESHOLD,
                                     class_balance_diagnostics,
                                     run_bias_explainability)
from src.dataset_profiler import drop_serialized_index_columns, profile_dataset
from src.experiment_results import (ExperimentRecord, build_experiment_downloads,
                                    build_recommendation_records)
from src.external_automl import (DEFAULT_TIME_BUDGET_SECONDS,
                                 MAX_TIME_BUDGET_SECONDS,
                                 run_external_automl_comparison)
from src.leakage_experiment import (LEAKAGE_DATASET, LEAKAGE_DEMONSTRATION_LABEL,
                                    LEAKAGE_SPLIT_SEED, LEAKAGE_TEST_SIZE,
                                    run_leakage_demonstration)
from src.model_card import MODEL_CARD_MIME_TYPE, build_model_card
from src.recommender import recommend_algorithms


st.set_page_config(page_title="MetaStart AutoML", page_icon="🧭", layout="wide")
st.title("MetaStart AutoML")
st.caption("A meta-learning research prototype for classification algorithm recommendation.")
st.warning("Recommendations are estimates from a small benchmark collection. Candidate selection uses cross-validation on the training partition; held-out test metrics are reported only for the selected candidate.")

upload = st.file_uploader("Upload a classification dataset (CSV)", type=["csv"])
if upload:
    st.caption(
        f"Uploaded file: {upload.size / (1024 * 1024):.1f} MiB. "
        f"Application file limit: {MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MiB."
    )
    try:
        frame = load_uploaded_csv(upload)
    except UploadValidationError as exc:
        st.error(str(exc))
    else:
        st.caption(f"Parsed CSV: {len(frame):,} rows × {len(frame.columns):,} columns.")
        frame, removed_index_columns = drop_serialized_index_columns(frame)
        if removed_index_columns:
            st.info("Removed serialized row index column(s): " + ", ".join(removed_index_columns))
        target = None
        if frame.empty or frame.shape[1] < 2:
            st.error("The CSV must have rows and at least one feature plus a target column.")
        else:
            target = st.selectbox(
                "Choose the target column",
                [None, *frame.columns],
                format_func=lambda value: "Select a target column" if value is None else str(value),
            )
        if st.button("Recommend, train and evaluate", type="primary"):
            try:
                prepared = prepare_classification_data(frame, target)
            except UploadValidationError as exc:
                st.error(str(exc))
            else:
                if prepared.dropped_target_rows:
                    st.info(f"Removed {prepared.dropped_target_rows:,} row(s) with a missing target.")
                if prepared.dropped_all_missing_features:
                    st.info("Removed feature column(s) containing only missing values: "
                            + ", ".join(prepared.dropped_all_missing_features))
                frame = None
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        prepared.X, prepared.y, test_size=0.25, random_state=42,
                        stratify=prepared.y)
                except ValueError as exc:
                    st.error(f"Could not create a stratified train/test split. Add more examples per class: {exc}")
                else:
                    del prepared
                    st.info(
                        f"Data split: {len(X_train):,} training rows for profiling and CV; "
                        f"{len(X_test):,} held-out test rows reserved for the selected model only."
                    )
                    try:
                        profile = profile_dataset(X_train, y_train)
                        ranking = recommend_algorithms(X_train, y_train, profile=profile)
                    except FileNotFoundError:
                        st.error("Production meta-model A was not found. Restore models/meta_model.joblib before starting the app.")
                    except Exception as exc:
                        st.error(f"Could not generate recommendations from production model A: {type(exc).__name__}: {exc}")
                    else:
                        profile_table = production_profile_table(profile)
                        st.subheader("Training dataset profile — production model features")
                        st.dataframe(profile_table, hide_index=True, width="stretch")
                        st.subheader("MetaStart ranking — predicted Macro-F1")
                        st.caption("This complete five-algorithm ranking is predicted by production model A. Scores are estimates, not measured test performance.")
                        st.bar_chart(ranking.set_index("algorithm")["predicted_macro_f1"],
                                     horizontal=True, x_label="Predicted Macro-F1", width="stretch")
                        st.dataframe(ranking[["algorithm", "predicted_macro_f1"]],
                                     hide_index=True, width="stretch")
                        recommendation_table = recommendation_download_table(ranking)
                        st.download_button(
                            "Download full recommendation ranking (CSV)",
                            recommendation_table.to_csv(index=False),
                            "metastart_recommendation_ranking.csv", "text/csv",
                        )

                        st.subheader("Candidate selection — training-partition cross-validation")
                        st.caption("The highest mean stratified CV Macro-F1 selects the measured candidate. The reserved test labels are not used to rank or select candidates.")

                        progress = st.progress(0)
                        status = st.empty()

                        def update_progress(index, total, algorithm):
                            status.info(f"{algorithm} ({index + 1} of {total})…")
                            progress.progress(index / total)

                        evaluation = None
                        try:
                            evaluation = evaluate_ranked_candidates(
                                ranking, X_train, y_train, X_test, y_test,
                                model_factory=get_models, pipeline_builder=build_pipeline,
                                progress_callback=update_progress,
                            )
                        except Exception as exc:
                            status.empty()
                            evaluation_error = f"{type(exc).__name__}: {exc}"
                            st.error(f"Candidate evaluation could not start: {evaluation_error}")
                            selection_results = pd.DataFrame()
                            failures = ()
                            selected_algorithm = None
                            full_training_seconds = None
                        else:
                            progress.progress(1.0)
                            status.success("Training-only cross-validation and selected-model evaluation finished.")
                            for algorithm, phase, error in evaluation.failures:
                                st.warning(f"{algorithm} failed during {phase}: {error}")
                            evaluation_error = None
                            selection_results = evaluation.selection_results
                            failures = evaluation.failures
                            selected_algorithm = evaluation.selected_algorithm
                            st.caption(
                                f"All five candidates use the same stratified {evaluation.cv_n_splits}-fold splits. "
                                f"CV used {evaluation.cv_rows:,} of {evaluation.cv_source_rows:,} training rows "
                                "(deterministic class-stratified sample; maximum 5,000 rows). "
                                "Out-of-fold class metrics below use those same CV rows."
                            )
                            full_training_seconds = (
                                float(evaluation.test_result.iloc[0]["Training seconds"])
                                if not evaluation.test_result.empty else None
                            )

                            if evaluation.selection_results.empty:
                                st.error("No candidate completed training-partition cross-validation. The recommendation ranking is available, but there is no selected model or held-out result.")
                            else:
                                st.dataframe(evaluation.selection_results, hide_index=True, width="stretch")
                                st.subheader("Training-partition diagnostics")
                                st.dataframe(dataset_diagnostics_table(profile, y_train),
                                             hide_index=True, width="stretch")
                                st.caption(
                                    "Distribution, correlation, and entropy summaries describe the training data. "
                                    "Correlation does not demonstrate class separability. Preprocessing effects were "
                                    "not separately ablated; every candidate uses the same fold-fitted imputation, "
                                    "scaling, and categorical encoding pipeline."
                                )
                                feature_table, correlation_table = feature_structure_tables(X_train)
                                with st.expander("Per-feature distributions and missingness"):
                                    st.dataframe(feature_table, hide_index=True, width="stretch")
                                with st.expander("Strongest numeric feature correlations"):
                                    if correlation_table.empty:
                                        st.info("No valid numeric feature pairs are available for correlation summaries.")
                                    else:
                                        st.dataframe(correlation_table, hide_index=True, width="stretch")
                                    st.caption("Absolute Pearson correlations are descriptive associations between features; they do not measure class separation or causation.")
                                st.subheader("Class-wise out-of-fold performance")
                                if evaluation.class_metrics.empty:
                                    st.info("No class-wise CV results are available because candidates did not complete CV.")
                                else:
                                    st.dataframe(evaluation.class_metrics, hide_index=True, width="stretch")
                                st.subheader("Out-of-fold error patterns")
                                if evaluation.error_patterns.empty:
                                    st.info("No misclassified out-of-fold examples were observed for successful candidates.")
                                else:
                                    st.dataframe(evaluation.error_patterns, hide_index=True, width="stretch")
                                st.subheader("Linear versus RBF SVM diagnostic")
                                if evaluation.boundary_diagnostic.empty:
                                    st.info("The diagnostic-only kernel comparison was not available.")
                                else:
                                    st.dataframe(evaluation.boundary_diagnostic, hide_index=True, width="stretch")
                                    st.caption(
                                        "This diagnostic compares two SVM kernel configurations on the same training-only "
                                        "folds. It is not included in candidate selection and does not prove the dataset "
                                        "is linearly separable."
                                    )
                                if evaluation.selected_algorithm is None:
                                    st.error("Cross-validation completed, but no candidate could be fitted successfully on the full training partition. No held-out metrics or model download are available.")
                                elif evaluation.test_result.empty:
                                    st.error(f"{evaluation.selected_algorithm} was selected by training-only CV, but held-out evaluation failed. No other candidate was selected using test results.")
                                else:
                                    st.success(f"Selected by training-only CV: {evaluation.selected_algorithm} (mean CV Macro-F1 {evaluation.test_result.iloc[0]['Selection CV Macro-F1']:.3f}).")
                                    st.subheader("Held-out test metrics — selected candidate only")
                                    st.caption("This reserved split was evaluated once after selection. Test metrics did not choose the candidate.")
                                    st.dataframe(evaluation.test_result, hide_index=True, width="stretch")
                                    st.download_button(
                                        "Download evaluation results",
                                        evaluation.test_result.to_csv(index=False),
                                        "metastart_results.csv", "text/csv",
                                    )
                                    model_buffer = BytesIO()
                                    joblib.dump(evaluation.fitted_models[evaluation.selected_algorithm], model_buffer)
                                    st.download_button(
                                        "Download selected trained model", model_buffer.getvalue(),
                                        "metastart_selected_model.joblib", "application/octet-stream",
                                    )

                        st.subheader("Algorithm Comparison")
                        st.caption("The comparison combines MetaStart's predicted ranking, training-only CV outcomes, observed CV fit times, and the measured training-partition profile. Held-out test metrics are not used in these explanations.")
                        comparison = build_algorithm_comparison(
                            ranking, selection_results, profile, failures,
                            selected_algorithm, full_training_seconds, evaluation_error,
                            evaluation.class_metrics if evaluation is not None else None,
                        )
                        summary_columns = [
                            "Predicted rank", "Algorithm", "Predicted Macro-F1",
                            "Mean training-only CV Macro-F1", "CV Macro-F1 standard deviation",
                            "CV mean fit seconds per fold", "Selected full-training fit seconds",
                            "CV status", "Selected by CV rule",
                        ]
                        st.dataframe(comparison[summary_columns], hide_index=True, width="stretch")
                        for _, algorithm_row in comparison.iterrows():
                            with st.expander(
                                    f"Rank {int(algorithm_row['Predicted rank'])}: {algorithm_row['Algorithm']} — strengths and fit for this dataset"):
                                st.markdown(f"**General strengths:** {algorithm_row['General strengths']}")
                                st.markdown(f"**Trade-offs:** {algorithm_row['General trade-offs']}")
                                st.markdown(f"**Dataset-specific suitability:** {algorithm_row['Dataset-specific suitability evidence']}")

                        rationale = build_selection_rationale(
                            selection_results, failures, selected_algorithm, ranking,
                            evaluation_error,
                            evaluation.boundary_diagnostic if evaluation is not None else None,
                        )
                        st.subheader("Why This Algorithm Was Selected")
                        st.write(rationale["selected"])
                        st.subheader("Why Not the Other Algorithms")
                        for reason in rationale["alternatives"]:
                            st.markdown(f"- {reason}")

                        experiment_records = build_recommendation_records(
                            ALGORITHM_NAMES, ranking, selection_results, failures,
                            selected_algorithm=selected_algorithm,
                            full_training_seconds=full_training_seconds,
                            test_result=(evaluation.test_result if evaluation is not None
                                         else pd.DataFrame()),
                            class_metrics=(evaluation.class_metrics if evaluation is not None
                                           else pd.DataFrame()),
                            cv_source_rows=(evaluation.cv_source_rows if evaluation is not None else 0),
                            cv_rows=(evaluation.cv_rows if evaluation is not None else 0),
                            cv_n_splits=(evaluation.cv_n_splits if evaluation is not None else 0),
                            test_rows=len(X_test) if selected_algorithm is not None else 0,
                            dataset_label=f"uploaded:{target}",
                        )
                        upload_signature = hashlib.sha256(upload.getbuffer()).hexdigest()
                        # Optional experiment results belong to both this upload and
                        # the currently selected fitted algorithm. Including the
                        # selection prevents a previous model's tuning/AutoML records
                        # from appearing in a new model card after the selector changes.
                        advanced_key = f"{upload_signature}:{target}:{selected_algorithm or 'no_selection'}"
                        with st.expander("Manual baseline and hyperparameter searches", expanded=False):
                            st.caption(
                                "These bounded searches use the selected algorithm and the training partition only. "
                                "They share MetaStart’s deterministic, stratified CV sample; held-out rows are not used."
                            )
                            if st.button(
                                "Run manual baseline, Grid Search, Random Search, and Optuna",
                                disabled=selected_algorithm is None,
                                key="run_advanced_searches",
                            ):
                                with st.spinner("Running the untuned baseline and bounded CV searches…"):
                                    advanced_records, advanced_summary = run_tuning_experiments(
                                        X_train, y_train, selected_algorithm,
                                        cv_folds=(evaluation.cv_n_splits if evaluation is not None else 3),
                                        experiment_id=advanced_key,
                                    )
                                st.session_state["advanced_experiment_results"] = {
                                    "key": advanced_key,
                                    "records": advanced_records,
                                    "summary": advanced_summary,
                                }
                            saved_advanced = st.session_state.get("advanced_experiment_results")
                            if saved_advanced and saved_advanced.get("key") == advanced_key:
                                experiment_records.extend(saved_advanced["records"])
                                st.dataframe(saved_advanced["summary"], hide_index=True, width="stretch")
                            else:
                                st.info("Run the searches to add their measured results to the report downloads.")

                        with st.expander("External AutoML framework comparison", expanded=False):
                            st.caption(
                                "FLAML and AutoGluon receive the same training partition and Macro-F1 objective. "
                                "The held-out test partition is used only after each framework finishes search. "
                                "Optional packages are loaded only when this comparison runs."
                            )
                            external_budget = st.slider(
                                "Time budget per framework (minutes)",
                                min_value=1,
                                max_value=MAX_TIME_BUDGET_SECONDS // 60,
                                value=DEFAULT_TIME_BUDGET_SECONDS // 60,
                                step=1,
                                key="external_automl_time_budget_minutes",
                            )
                            external_budget_seconds = int(external_budget * 60)
                            external_key = f"{advanced_key}:{external_budget_seconds}"
                            if st.button("Run external AutoML comparison", key="run_external_automl"):
                                with st.status("Starting external AutoML comparison…", expanded=True) as progress_status:
                                    def external_progress(framework, status):
                                        if status == "running":
                                            progress_status.update(
                                                label=f"{framework} is searching (up to {external_budget} min)…",
                                                state="running",
                                            )
                                        else:
                                            progress_status.update(
                                                label=f"{framework}: {status.replace('_', ' ')}",
                                                state="running",
                                            )

                                    external_records, external_summary, external_leaderboards = (
                                        run_external_automl_comparison(
                                            X_train, y_train, X_test, y_test,
                                            time_budget_seconds=external_budget_seconds,
                                            seed=42,
                                            cv_folds=(evaluation.cv_n_splits
                                                      if evaluation is not None and evaluation.cv_n_splits >= 2
                                                      else 3),
                                            experiment_id=external_key,
                                            progress_callback=external_progress,
                                        )
                                    )
                                    progress_status.update(
                                        label="External AutoML runs finished; check each framework status below.",
                                        state="complete",
                                    )
                                st.session_state["external_automl_results"] = {
                                    "key": external_key,
                                    "records": external_records,
                                    "summary": external_summary,
                                    "leaderboards": external_leaderboards,
                                }
                            saved_external = st.session_state.get("external_automl_results")
                            if saved_external and saved_external.get("key") == external_key:
                                experiment_records.extend(saved_external["records"])
                                st.dataframe(saved_external["summary"], hide_index=True, width="stretch")
                                for framework, leaderboard in saved_external["leaderboards"].items():
                                    st.markdown(f"**{framework} leaderboard**")
                                    if leaderboard.empty:
                                        st.info("No leaderboard entries were produced.")
                                    else:
                                        st.dataframe(leaderboard, hide_index=True, width="stretch")
                            else:
                                st.info(
                                    "External results are opt-in. Choose a per-framework budget and run the comparison; "
                                    "unavailable frameworks will be reported with their actual status."
                                )

                        with st.expander("Deliberate data-leakage demonstration", expanded=False):
                            st.error(LEAKAGE_DEMONSTRATION_LABEL)
                            st.write(
                                "This isolated demonstration uses scikit-learn’s bundled Breast Cancer dataset, "
                                "not the uploaded data or the production recommender. The valid baseline uses the "
                                "original features and training-fitted preprocessing. The deliberately leaky run "
                                "copies the target into a feature for both training and held-out rows. That gives "
                                "the model the answer during evaluation, so its score is invalid for real-world use."
                            )
                            if st.button(
                                "Run leakage demonstration",
                                key="run_leakage_demonstration",
                            ):
                                with st.spinner("Running the isolated leakage demonstration…"):
                                    leakage_records, leakage_summary = run_leakage_demonstration()
                                st.session_state["leakage_demonstration_results"] = {
                                    "records": leakage_records,
                                    "summary": leakage_summary,
                                }
                            saved_leakage = st.session_state.get("leakage_demonstration_results")
                            if saved_leakage:
                                leakage_summary = saved_leakage["summary"]
                                result_columns = st.columns(2)
                                for column, (_, result_row) in zip(
                                        result_columns, leakage_summary.iterrows()):
                                    with column:
                                        st.markdown(f"### {result_row['Result']}")
                                        if result_row["Status"] == "invalid_for_real_world":
                                            st.error(LEAKAGE_DEMONSTRATION_LABEL)
                                        else:
                                            st.success("Leakage-safe comparison baseline")
                                        st.metric("Held-out Macro-F1", f"{result_row['Macro-F1']:.4f}")
                                        st.metric("Held-out accuracy", f"{result_row['Accuracy']:.4f}")
                                        st.caption(
                                            f"Macro precision {result_row['Macro precision']:.4f} · "
                                            f"Macro recall {result_row['Macro recall']:.4f} · "
                                            f"Fit {result_row['Fit seconds']:.3f}s · "
                                            f"Test scoring {result_row['Test scoring seconds']:.3f}s"
                                        )
                                st.dataframe(leakage_summary, hide_index=True, width="stretch")
                                experiment_records.extend(saved_leakage["records"])
                            else:
                                st.info(
                                    "This experiment is opt-in. It runs only when you press the button and uses "
                                    "a fixed built-in dataset, independent of your upload."
                                )

                        with st.expander("Bias evaluation and model explainability", expanded=False):
                            st.caption(
                                "Optional diagnostics use training-partition data and the already CV-selected fitted model. "
                                "They do not refit or reselect the production candidate, and never use held-out test labels."
                            )
                            st.write(
                                "Class imbalance is flagged when the largest class has at least three times the "
                                "smallest class count. Class-wise metrics use a separate stratified out-of-fold "
                                "evaluation with a configurable row cap. Global permutation importance also has "
                                "a configurable sample cap and is measured on rows used to fit the selected model, "
                                "so it may overstate generalization. The local explanation perturbs one training-row prediction. "
                                "Permutation importance is used instead of SHAP because it works with the existing "
                                "mixed-type preprocessing pipeline and all supported estimator families without "
                                "optional model-specific explainers. Importance indicates model association or "
                                "sensitivity, not causation."
                            )
                            bias_cv_limit = st.slider(
                                "Maximum training rows for out-of-fold class metrics",
                                min_value=300, max_value=3_000, value=3_000, step=300,
                                key="bias_cv_sample_limit",
                            )
                            explanation_sample_limit = st.slider(
                                "Maximum training rows for permutation importance",
                                min_value=25, max_value=200, value=200, step=25,
                                key="explanation_sample_limit",
                            )
                            bias_key = (
                                f"{advanced_key}:{selected_algorithm or 'no_selection'}:"
                                f"{bias_cv_limit}:{explanation_sample_limit}"
                            )
                            if st.button(
                                "Run bias evaluation and explanations",
                                key="run_bias_explainability",
                                disabled=(selected_algorithm is None or evaluation is None
                                          or selected_algorithm not in evaluation.fitted_models),
                            ):
                                try:
                                    with st.spinner("Computing bounded training-only diagnostics and explanations…"):
                                        bias_result = run_bias_explainability(
                                            X_train, y_train,
                                            selected_algorithm=selected_algorithm,
                                            fitted_pipeline=evaluation.fitted_models[selected_algorithm],
                                            model_factory=get_models,
                                            pipeline_builder=build_pipeline,
                                            experiment_id=f"{bias_key}:explainability",
                                            cv_sample_limit=bias_cv_limit,
                                            explanation_sample_limit=explanation_sample_limit,
                                        )
                                except Exception as exc:
                                    bias_result = {
                                        "records": [ExperimentRecord(
                                            experiment_id=f"{bias_key}:explainability::diagnostics",
                                            method="training_partition_bias_explainability",
                                            algorithm=selected_algorithm or "unavailable",
                                            stage="bias",
                                            status="skipped_or_failed",
                                            warnings=(f"{type(exc).__name__}: {exc}",),
                                            notes=("Opt-in diagnostics failed; production workflow remains available.",),
                                        )],
                                        "failure": f"{type(exc).__name__}: {exc}",
                                    }
                                st.session_state["bias_explainability_results"] = {
                                    "key": bias_key, "result": bias_result,
                                }
                            saved_bias = st.session_state.get("bias_explainability_results")
                            if saved_bias and saved_bias.get("key") == bias_key:
                                bias_result = saved_bias["result"]
                                if "failure" in bias_result:
                                    st.warning(f"Diagnostics were skipped: {bias_result['failure']}")
                                    experiment_records.extend(bias_result["records"])
                                else:
                                    st.markdown(
                                        f"**Class imbalance ratio:** {bias_result['imbalance_ratio']:.2f}:1 — "
                                        f"{'flagged' if bias_result['imbalance_flagged'] else 'not flagged'} "
                                        "(flag threshold ≥ 3:1)."
                                    )
                                    st.dataframe(bias_result["balance"], hide_index=True, width="stretch")
                                    st.caption(bias_result["fairness_limitation"])
                                    st.markdown("**Class-wise out-of-fold performance (training partition)**")
                                    st.dataframe(bias_result["class_metrics"], hide_index=True, width="stretch")
                                    bias_record = next(
                                        record for record in bias_result["records"] if record.stage == "bias"
                                    )
                                    st.caption(
                                        f"Method: stratified {bias_record.cv_folds} fold out-of-fold estimates; "
                                        f"{bias_record.cv_rows:,} "
                                        f"training rows; runtime {bias_result['cv_runtime_seconds']:.3f}s. "
                                        "Classes marked below the class mean have lower observed F1 in this run."
                                    )
                                    st.markdown("**Confusion matrix (rows: actual; columns: predicted)**")
                                    matrix_view = bias_result["confusion_matrix"].pivot(
                                        index="True Class", columns="Predicted Class", values="Count"
                                    ).fillna(0).astype(int)
                                    st.dataframe(matrix_view, width="stretch")
                                    st.markdown("**False-positive / false-negative counts by class pair**")
                                    if bias_result["error_counts"].empty:
                                        st.info("No out-of-fold classification errors were observed.")
                                    else:
                                        st.dataframe(bias_result["error_counts"], hide_index=True, width="stretch")
                                    st.markdown("**Global feature importance — permutation importance**")
                                    if bias_result["importance"].empty:
                                        st.warning("Global feature importance was unavailable.")
                                    else:
                                        st.dataframe(bias_result["importance"].head(20), hide_index=True, width="stretch")
                                        strongest = bias_result["importance"].iloc[0]
                                        st.write(
                                            f"On this sample, permuting **{strongest['Feature']}** "
                                            f"{'reduced' if strongest['Importance Mean'] >= 0 else 'increased'} Macro-F1 "
                                            f"by {abs(strongest['Importance Mean']):.4f} on average. Larger positive "
                                            "reductions indicate greater reliance for this fitted model; this does not "
                                            "show that the feature causes the outcome."
                                        )
                                    st.caption(
                                        f"Method: permutation importance using Macro-F1, "
                                        f"{bias_result['importance_sample_count']} training rows, 3 repeats; "
                                        f"runtime {bias_result['importance_runtime_seconds']:.3f}s. "
                                        "Positive values indicate score reduction after permutation; this is not causal evidence."
                                    )
                                    st.markdown("**One individual prediction — single-feature replacement sensitivity**")
                                    if bias_result["local_explanation"].empty:
                                        st.info("A local explanation was unavailable for this fitted model.")
                                    else:
                                        st.dataframe(bias_result["local_explanation"], hide_index=True, width="stretch")
                                        local_record = next(
                                            record for record in bias_result["records"]
                                            if record.method == "single_feature_ablation_sensitivity"
                                        )
                                        st.caption(
                                            f"Method: one training-partition prediction; "
                                            f"{local_record.leaderboard_metrics['features_explained']} features; "
                                            f"runtime {local_record.leaderboard_metrics['runtime_seconds']:.3f}s. "
                                            "Changes compare the original class score with a median/mode replacement."
                                        )
                                    for warning in bias_result["warnings"]:
                                        st.warning(warning)
                                    experiment_records.extend(bias_result["records"])
                            else:
                                st.info("These diagnostics run only when requested and do not affect model selection.")

                        report_metadata = {
                            "dataset": f"uploaded:{target}",
                            "target": str(target),
                            "training_rows": len(X_train),
                            "held_out_rows": len(X_test),
                            "selected_algorithm": selected_algorithm,
                            "separate_leakage_demonstration": ({
                                "dataset": LEAKAGE_DATASET,
                                "split": "stratified train/test",
                                "test_size": LEAKAGE_TEST_SIZE,
                                "seed": LEAKAGE_SPLIT_SEED,
                                "label": LEAKAGE_DEMONSTRATION_LABEL,
                            } if st.session_state.get("leakage_demonstration_results") else None),
                        }
                        model_card_text = None
                        fitted_selected = (
                            evaluation.fitted_models.get(selected_algorithm)
                            if evaluation is not None and selected_algorithm is not None else None
                        )
                        if fitted_selected is not None:
                            try:
                                balance_table, imbalance_ratio, imbalance_flagged = (
                                    class_balance_diagnostics(y_train)
                                )
                                fitted_estimator = fitted_selected.steps[-1][1]
                                model_parameters = fitted_estimator.get_params(deep=False)
                                report_metadata.update({
                                    "model_name": f"MetaStart {selected_algorithm} classifier",
                                    "model_version": "IA-3 research prototype; unversioned",
                                    "application_version": "MetaStart IA-3 research prototype",
                                    "card_created_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                    "run_id": advanced_key,
                                    "dataset_name": upload.name,
                                    "dataset_source": "User-uploaded CSV; source provenance not independently verified",
                                    "dataset_rows": len(X_train) + len(X_test),
                                    "dataset_features": X_train.shape[1],
                                    "class_distribution": [{
                                        "class": str(row["Class"]),
                                        "count": int(row["Count"]),
                                        "proportion": float(row["Proportion"]),
                                    } for _, row in balance_table.iterrows()],
                                    "imbalance_ratio": imbalance_ratio,
                                    "imbalance_flagged": imbalance_flagged,
                                    "imbalance_ratio_threshold": IMBALANCE_RATIO_THRESHOLD,
                                    "split_strategy": "Stratified 75/25 train/test split; random_state=42",
                                    "random_seed": 42,
                                    "cv_folds": evaluation.cv_n_splits,
                                    "cv_rows": evaluation.cv_rows,
                                    "cv_source_rows": evaluation.cv_source_rows,
                                    "selected_model_parameters": model_parameters,
                                    "training_seconds": full_training_seconds,
                                    "environment": (
                                        f"Python {platform.python_version()}; scikit-learn {sklearn.__version__}; "
                                        f"pandas {pd.__version__}"
                                    ),
                                    "sampling_limits": (
                                        f"Training CV sample capped at {CV_MAX_ROWS:,} rows; this run used "
                                        f"{evaluation.cv_rows:,} of {evaluation.cv_source_rows:,} training rows."
                                    ),
                                    "resource_limits": {
                                        "max_upload_rows": MAX_UPLOAD_ROWS,
                                        "max_upload_features": MAX_UPLOAD_FEATURES,
                                        "max_upload_cells": MAX_UPLOAD_CELLS,
                                        "max_upload_bytes": MAX_UPLOAD_BYTES,
                                        "max_full_fit_rows_for_selected_algorithm": (
                                            MAX_FULL_FIT_ROWS.get(selected_algorithm)
                                        ),
                                        "cv_max_rows": CV_MAX_ROWS,
                                    },
                                })
                                model_card_text = build_model_card(experiment_records, report_metadata)
                            except Exception as exc:
                                st.warning(
                                    f"The model card could not be generated: {type(exc).__name__}: {exc}. "
                                    "Existing experiment downloads remain available."
                                )

                        experiment_downloads = build_experiment_downloads(
                            experiment_records,
                            metadata=report_metadata,
                            model_card_text=model_card_text,
                        )
                        st.subheader("Experiment results export")
                        st.download_button(
                            "Download comprehensive experiment CSV",
                            experiment_downloads["experiment_results.csv"],
                            "metastart_experiment_results.csv", "text/csv",
                        )
                        st.download_button(
                            "Download detailed metrics CSV",
                            experiment_downloads["experiment_detailed_metrics.csv"],
                            "metastart_experiment_detailed_metrics.csv", "text/csv",
                        )
                        st.download_button(
                            "Download experiment JSON report",
                            experiment_downloads["experiment_report.json"],
                            "metastart_experiment_report.json", "application/json",
                        )
                        st.download_button(
                            "Download experiment results bundle (ZIP)",
                            experiment_downloads["experiment_results_bundle.zip"],
                            "metastart_experiment_results_bundle.zip", "application/zip",
                        )
                        if model_card_text is not None:
                            st.download_button(
                                "Download model card (Markdown)",
                                experiment_downloads["model_card.md"],
                                "metastart_model_card.md", MODEL_CARD_MIME_TYPE,
                            )
                        elif fitted_selected is None:
                            st.info("A model card is available after a candidate is selected and fitted successfully.")
else:
    st.info("Upload a CSV to begin. The target column must contain classification labels.")

st.divider()
st.caption("MetaStart AutoML | Research prototype | Macro-F1 estimates are not guaranteed test scores.")
