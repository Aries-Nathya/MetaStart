# MetaStart AutoML

**A Meta-Learning Framework for Intelligent Machine Learning Algorithm Recommendation**

MetaStart profiles a classification dataset and recommends which of five candidate algorithms may perform well. The Streamlit app also selects one candidate using cross-validation on the training partition, then reports that candidate's metrics on a reserved test partition. MetaStart is a final-year research prototype; its small benchmark does not establish broad real-world generalization.

## Run the application

Run these commands from the repository root. The app uses the existing production model at `models/meta_model.joblib`; ordinary startup does not regenerate benchmark data, train a model, or run the research evaluation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. On Windows, activate the environment with `.venv\\Scripts\\activate` before installing requirements and starting Streamlit.

The currently available environment used for verification is Python 3.14.7, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1, Streamlit 1.65.0, Matplotlib 3.11.2, joblib 1.6.0, and pytest 9.1.1. `requirements.txt` declares minimum versions rather than a fully locked environment; reproducible installs across platforms may resolve different newer versions.

## Use the app

1. Upload a UTF-8 CSV with a header, one or more feature columns, and a classification target column.
2. Select the target column and choose **Recommend, train and evaluate**.
3. Review the dataset profile, the complete five-algorithm MetaStart ranking, training-only cross-validation results, and the selected model's held-out test metrics.

The default upload safeguards support up to **750,000 rows**, **200 MiB per CSV**, **100 feature columns**, and **40 million table cells**. These defaults accommodate Forest CoverType (581,012 rows and 54 input features) within the configured cell budget. Limits are centralized in `src/app_workflow.py` and can be adjusted with `METASTART_MAX_UPLOAD_ROWS`, `METASTART_MAX_UPLOAD_MB`, `METASTART_MAX_UPLOAD_FEATURES`, and `METASTART_MAX_UPLOAD_CELLS`. The file-size limit must not exceed Streamlit's `server.maxUploadSize` in `.streamlit/config.toml`; both default to 200 MiB. Increase limits only when the machine has enough memory for the full training and test partitions and model fitting.

The target must have 2–50 classes and at least four usable examples in each class. Rows with a missing target are removed. `?` and standard CSV missing markers are treated as missing. Feature columns that are entirely missing are removed; if no usable features remain, the app reports an error. Numeric and categorical columns are supported. Unsupported values, malformed CSV, duplicate column names, one-class targets, and undersized classes receive actionable errors. Serialized row-index columns are detected and removed. CSV parsing is performed once after the header, file size, width, and configured row/cell budgets are checked; row counts are bounded before the full table is materialized.

For larger datasets, the complete training partition is profiled once and the held-out test partition remains separate. Candidate cross-validation uses a deterministic class-stratified sample of at most 5,000 training rows with three folds; the UI reports both the full training-row count and actual CV sample size. The CV sample does not include held-out test rows. The selected candidate is refitted on the full training partition only if its full-data resource guard permits it. Current refit guards are 750,000 rows for Logistic Regression and k-NN, 500,000 for Decision Tree, 250,000 for Random Forest, 20,000 for RBF SVM, and 25 million train/test distance pairs for k-NN. A candidate exceeding a guard is reported and the workflow tries the next candidate by training-only CV order; it does not silently sample or change that candidate's fit method. This means some models may not be available for full-data evaluation on a large upload.

Numeric missing values are imputed with the training-fold median and scaled. Categorical missing values are imputed with the training-fold most frequent value and one-hot encoded. These transformations are part of the estimator pipeline, so each cross-validation fold fits preprocessing using only that fold's training data.

## Scores, selection, and downloads

The app keeps three kinds of score distinct:

- **Predicted Macro-F1** is the estimate from production meta-model A. It ranks all five supported algorithms and is not a measurement on the uploaded data.
- **Training-partition CV Macro-F1** is measured for all five algorithms in stratified cross-validation on the training partition, using the same deterministic sample (at most 5,000 rows) and three folds. Selection uses mean CV Macro-F1, with candidate-specific failures reported. Full-data resource guards can prevent an otherwise CV-ranked candidate from being refitted; the workflow then tries the next CV-ranked candidate.
- **Held-out test metrics** are computed once for the CV-selected model after it is fitted on the full training partition. The test scores are not used to choose the winner. They describe this reserved split and should not be used as an unbiased estimate after further model or workflow changes.

The app provides three separate downloads:

- **Full recommendation ranking** (`metastart_recommendation_ranking.csv`): all five algorithms, rank 1–5, and their actual predicted Macro-F1 scores.
- **Evaluation results** (`metastart_results.csv`): the selected algorithm, its training-only CV score, its one held-out test result, and training time.
- **Selected trained model** (`metastart_selected_model.joblib`): the selected fitted preprocessing-and-classifier pipeline.

The recommendation CSV is available when the ranking succeeds even if candidate evaluation later fails. The CV results, held-out result, and fitted model are separate from that ranking.

## Model and benchmark status

Production **model A** is the original nine-feature meta-model saved at `models/meta_model.joblib`. The app inference path uses those same nine features. The 15-feature **model B** is retained as a research reference in `models/feature_experiments/feature_abc_v1_20261014/`; the app does not load it. The experimental features and repeated-seed comparisons have not replaced model A.

The checked-in benchmark files contain 60 dataset–algorithm records for 12 identities, with five algorithm records per identity:

- 4 bundled scikit-learn datasets: Breast Cancer, Digits, Iris, and Wine.
- 6 deterministic synthetic datasets from the local synthetic generator family.
- 2 OpenML-prefixed identities whose source provenance remains unverified. Their records are retained for audit, but provenance filtering excludes them from meta-model training and the leave-one-dataset-identity-out evaluation.

Thus, 10 identities and 50 records are eligible for training and evaluation. Six eligible identities are synthetic; they are useful for controlled engineering checks but are not independent real-world evidence. The manifest is `results/dataset_provenance.csv`; the 14-column meta-dataset is `data/meta_dataset.csv` and benchmark results are in `results/benchmark_results.csv`.

The production recommendation estimates are learned from a small and compositionally limited collection. Four eligible identities are real bundled datasets and six share one synthetic generator family. Treat recommendation performance and evaluation intervals as descriptive and preliminary, not proof of generalization to new real-world datasets.

## Architecture

| Component | Role |
| --- | --- |
| `src/dataset_profiler.py` | Shared dataset profile and meta-feature calculations. |
| `src/dataset_catalog.py` | Approved benchmark identities and deterministic synthetic dataset specifications. |
| `src/benchmark.py` | Built-in dataset collection, optional explicit OpenML loader, preprocessing pipelines, and benchmark generation. |
| `src/provenance.py` | Dataset provenance checks and filtering of eligible benchmark records. |
| `src/meta_learner.py` | Production meta-model construction and training from eligible records. |
| `src/recommender.py` | Loads the saved model and returns the five-algorithm predicted ranking. |
| `src/app_workflow.py` | Configurable upload safeguards, validation, deterministic CV sampling, per-candidate resource guards, and final test workflow. |
| `src/app_helpers.py` | Profile, feature-distribution, correlation, and evidence-led algorithm explanation tables. |
| `app.py` | Streamlit upload, ranking, candidate selection, measured results, and downloads. |
| `src/evaluation.py` | Leave-one-dataset-identity-out offline evaluation and summary artifacts. |
| `tests/` | Unit and workflow regression tests. |

Offline evaluation holds out complete dataset identities, keeping all five algorithm rows for a dataset together. However, benchmark meta-features are profiled from each complete dataset before its algorithm scores are cross-validated. This evaluates dataset-level recommendation using an available dataset profile; it is not a held-out-row prediction estimate. The ten eligible identities, six synthetic identities, and four bundled real datasets are too few for strong claims about generalization.

## Tests

With the virtual environment active:

```bash
python -m pytest -q
```

Or run the project's environment directly:

```bash
.venv/bin/python -m pytest -q
```

## Research artifact generation

These commands are **not** part of app setup or startup. They write to the repository's default artifact paths and can replace existing benchmark, model, or evaluation outputs. Before running them, preserve the current artifacts and inspect the script's output-path options; use an isolated checkout or explicit temporary output directory when supported.

```bash
python -m src.benchmark
python -m src.meta_learner
python -m src.evaluation
python -m src.repeated_validation
```

The benchmark command writes the meta-dataset and benchmark results; meta-model training writes `models/meta_model.joblib`; evaluation writes the report, summary, and plot. Do not run these commands merely to launch or demonstrate the app.
