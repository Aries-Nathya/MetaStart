"""Streamlit interface for the MetaStart classification prototype."""

from __future__ import annotations

from io import BytesIO
import time

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.benchmark import build_pipeline, get_models
from src.dataset_profiler import drop_serialized_index_columns, profile_dataset
from src.recommender import recommend_algorithms


st.set_page_config(page_title="MetaStart AutoML", page_icon="🧭", layout="wide")
st.title("MetaStart AutoML")
st.caption("A meta-learning research prototype for classification algorithm recommendation.")
st.warning("Recommendations are estimates from a small benchmark collection. Reported test metrics are measured on your uploaded data's held-out split.")

upload = st.file_uploader("Upload a classification dataset (CSV)", type=["csv"])
if upload:
    try:
        frame = pd.read_csv(upload)
        frame, removed_index_columns = drop_serialized_index_columns(frame)
        if removed_index_columns:
            st.info("Removed serialized row index column(s): " + ", ".join(removed_index_columns))
        if frame.empty or frame.shape[1] < 2:
            st.error("The CSV must have rows and at least one feature plus a target column.")
            st.stop()
        target = st.selectbox("Choose the target column", frame.columns)
        if st.button("Recommend, train and evaluate", type="primary"):
            clean = frame.dropna(subset=[target]).copy()
            X = clean.drop(columns=[target])
            y = clean[target]
            if X.empty or y.nunique() < 2 or y.value_counts().min() < 2:
                st.error("Need at least two target classes with two or more rows per class.")
                st.stop()
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y)
            except ValueError as exc:
                st.error(f"Could not create a stratified train/test split: {exc}")
                st.stop()
            try:
                ranking = recommend_algorithms(X_train, y_train)
            except FileNotFoundError:
                st.error("Meta-model not found. Run `python -m src.benchmark` and `python -m src.meta_learner` first.")
                st.stop()
            profile = profile_dataset(X_train, y_train)
            profile_labels = {
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
            profile_table = pd.DataFrame({
                "Characteristic": [profile_labels[key] for key in profile],
                "Value": list(profile.values()),
            })
            st.subheader("Training dataset profile")
            st.dataframe(profile_table, hide_index=True, width="stretch")
            st.subheader("MetaStart ranking — predicted Macro-F1")
            st.caption("The ranking combines the training-data profile with historical benchmark results. Predicted scores are estimates; the held-out test results below are measured separately.")
            st.bar_chart(ranking.set_index("algorithm")[["predicted_macro_f1"]],
                         horizontal=True, x_label="Predicted Macro-F1", width="stretch")
            st.dataframe(ranking[["algorithm", "predicted_macro_f1"]], hide_index=True, width="stretch")
            candidates = ranking.head(3)["algorithm"].tolist()
            if "Random Forest" not in candidates:
                candidates.append("Random Forest")
            results = []
            fitted = {}
            models = get_models()
            candidate_order = list(dict.fromkeys(candidates))
            progress = st.progress(0)
            for i, name in enumerate(candidate_order):
                pipeline = build_pipeline(models[name], X_train)
                start = time.perf_counter()
                pipeline.fit(X_train, y_train)
                elapsed = time.perf_counter() - start
                predictions = pipeline.predict(X_test)
                results.append({"Algorithm": name,
                                "Predicted Macro-F1": float(ranking.loc[ranking.algorithm == name, "predicted_macro_f1"].iloc[0]),
                                "Test Accuracy": accuracy_score(y_test, predictions),
                                "Test Macro-F1": f1_score(y_test, predictions, average="macro", zero_division=0),
                                "Training seconds": elapsed,
                                "Selection role": "Fixed baseline" if name == "Random Forest" else "MetaStart candidate"})
                fitted[name] = pipeline
                progress.progress((i + 1) / len(candidate_order))
            result_frame = pd.DataFrame(results).sort_values("Test Macro-F1", ascending=False)
            st.subheader("Measured held-out test results")
            st.caption("Test metrics are measured on the reserved 25% split. They do not affect the recommendation ranking.")
            st.dataframe(result_frame, hide_index=True, width="stretch")
            best_name = result_frame.iloc[0]["Algorithm"]
            st.success(f"Best measured candidate on this split: {best_name} (Test Macro-F1 {result_frame.iloc[0]['Test Macro-F1']:.3f})")
            st.download_button("Download evaluation results", result_frame.to_csv(index=False),
                               "metastart_results.csv", "text/csv")
            model_buffer = BytesIO()
            joblib.dump(fitted[best_name], model_buffer)
            st.download_button("Download best trained model", model_buffer.getvalue(),
                               "metastart_best_model.joblib", "application/octet-stream")
    except Exception as exc:
        st.error(f"Could not read or process the CSV: {exc}")
else:
    st.info("Upload a CSV to begin. The target column must contain classification labels.")

st.divider()
st.caption("MetaStart AutoML | Research prototype | Macro-F1 estimates are not guaranteed test scores.")
