"""Streamlit interface for the MetaStart classification prototype."""

from __future__ import annotations

from io import BytesIO

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.app_helpers import (build_algorithm_comparison, build_selection_rationale,
                             dataset_diagnostics_table, feature_structure_tables,
                             production_profile_table)
from src.app_workflow import (MAX_UPLOAD_BYTES, UploadValidationError, evaluate_ranked_candidates,
                              load_uploaded_csv, prepare_classification_data,
                              recommendation_download_table)
from src.benchmark import build_pipeline, get_models
from src.dataset_profiler import drop_serialized_index_columns, profile_dataset
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
else:
    st.info("Upload a CSV to begin. The target column must contain classification labels.")

st.divider()
st.caption("MetaStart AutoML | Research prototype | Macro-F1 estimates are not guaranteed test scores.")
