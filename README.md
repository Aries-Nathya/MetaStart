# MetaStart AutoML

**MetaStart AutoML: A Meta-Learning Framework for Intelligent Machine Learning Algorithm Recommendation**

MetaStart learns from benchmark experiments on prior datasets and estimates which classification algorithms are promising for a new dataset. It is a small, reproducible research prototype, not a claim of established research novelty.

## Architecture and workflow

**Offline:** collect benchmark datasets → calculate dataset meta-features → cross-validate five candidate models → save `data/meta_dataset.csv` → train a score regressor → evaluate by holding out complete dataset identities.

**Online:** upload a CSV → select the target → split into train/test → profile the training portion → rank candidate algorithms → fit the top three plus a fixed Random Forest baseline → report measured held-out scores and downloadable artifacts.

## Stack

Python, pandas, NumPy, scikit-learn, joblib and Streamlit. The benchmark includes Breast Cancer Wisconsin, Iris, Wine, Digits, and three deterministic synthetic classification datasets. Candidates are Logistic Regression, Decision Tree, Random Forest, SVM and k-Nearest Neighbors.

Meta-features include row and feature counts, numeric/categorical counts, missingness, class count and imbalance, and summary statistics of numeric columns. Imputation, one-hot encoding and scaling are fitted inside each CV training fold or online training split to avoid preprocessing leakage.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the workflow

Run from the project root:

```bash
python -m src.benchmark
python -m src.meta_learner
python -m src.evaluation
python -m src.repeated_validation
streamlit run app.py
```

The benchmark writes `data/meta_dataset.csv` and `results/benchmark_results.csv`. Training writes `models/meta_model.joblib`. Evaluation writes per-dataset results, a dataset-bootstrap summary CSV, and a regret chart in `results/`. Repeated validation writes fold-level and per-algorithm summaries for Iris, Wine, and Breast Cancer. In the dashboard, upload a classification CSV, choose the target, and click **Recommend, train and evaluate**. Sequential serialized CSV index columns are detected and removed. Predicted Macro-F1 rankings are labelled separately from measured held-out test results. Download the results CSV and the best measured model from that split.

To run the automated checks, use `python -m pytest`.

## Evaluation protocol

The research evaluation uses leave-one-dataset-out folds: every algorithm result for the held-out dataset is excluded from meta-model training. It reports selected macro-F1, regret from the best candidate, top-1/top-3 hits, prediction MAE and ranking Spearman correlation. Comparators are a fixed Random Forest, the historically best average algorithm selected from training datasets, and expected random choice estimated by repeated draws. The random baseline's standard deviation is included per held-out dataset.

The evaluation is small and noisy. The fixed candidate set, synthetic data, limited meta-features, and small number of dataset identities constrain conclusions. Bootstrap intervals resample the seven held-out dataset identities and are descriptive, not strong inferential evidence. Repeated validation on Iris, Wine, and Breast Cancer describes model stability on known benchmark datasets; it is not an independent test of MetaStart's generalization to new dataset identities. Strong research claims would need more diverse public datasets, repeated outer dataset-level splits, careful runtime accounting, and a literature review. Online scores also depend on one random-seeded holdout and should not be treated as a final unbiased estimate after inspecting candidates.

## Project structure

```text
app.py                     Streamlit dashboard
src/dataset_profiler.py    Shared dataset meta-features
src/benchmark.py           Dataset collection and model benchmark
src/meta_learner.py        Regressor training and artifact saving
src/recommender.py         Algorithm ranking for new data
src/evaluation.py          Held-out dataset baseline evaluation
data/                      Generated meta-dataset
models/                    Trained model artifacts
results/                   Benchmark and evaluation outputs
tests/                     Automated checks
```
