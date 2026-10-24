# MetaStart IA-3 Submission Deliverables Summary

## 1. Verified Project Tree of Relevant Deliverables

### Core Application Files
- `README.md` - Installation and usage instructions
- `app.py` - Main Streamlit application
- `requirements.txt` - Core dependencies
- `requirements-external-automl.txt` - Optional dependencies for FLAML/AutoGluon

### Source Code Modules
- `src/app_workflow.py` - Upload validation, data preparation, CV sampling, resource guards
- `src/app_helpers.py` - Profile tables, feature structure, algorithm explanations
- `src/benchmark.py` - Built-in dataset collection and benchmark generation
- `src/dataset_profiler.py` - Dataset profiling and meta-feature calculations
- `src/experiment_results.py` - Experiment ledger and report builders (CSV, JSON, ZIP)
- `src/external_automl.py` - Optional FLAML and AutoGluon comparisons
- `src/leakage_experiment.py` - Opt-in leakage demonstration
- `src/model_card.py` - Model card generation (downloadable Markdown)
- `src/bias_explainability.py` - Class imbalance diagnostics and explainability
- `src/advanced_experiments.py` - Manual baseline, Grid Search, Random Search, Optuna
- `src/recommender.py` - Loads production meta-model and returns algorithm rankings

### Key Assets
- `models/meta_model.joblib` - Production meta-learning model (9 features)
- `results/dataset_provenance.csv` - Dataset provenance tracking
- `results/benchmark_results.csv` - Benchmark results for meta-model training
- `results/regret_comparison.png` - Visualization from offline evaluation
- `results/experiment_freeze.json` - Metadata from a benchmark run

### Test Suite
- `tests/` - 84 unit and workflow regression tests covering all components

### Notes on Generated Artifacts
The application generates the following artifacts **on-the-fly** during a run (not committed to the repository):
- `metastart_recommendation_ranking.csv` - Full five-algorithm ranking
- `metastart_results.csv` - Selected model's training CV and held-out test metrics
- `metastart_selected_model.joblib` - Fitted preprocessing-and-classifier pipeline
- Experiment exports section provides:
  - `metastart_experiment_results.csv` (comprehensive experiment CSV)
  - `metastart_experiment_detailed_metrics.csv` (per-class CV metrics)
  - `metastart_experiment_report.json` (JSON report)
  - `metastart_experiment_results_bundle.zip` (ZIP bundle)
  - `metastart_model_card.md` (downloadable model card in Markdown)

These artifacts are held in memory and offered for download via Streamlit buttons; they are not written to disk by the application unless the user explicitly downloads them.

## 2. Reproducible Demo Command Sequence (Manual Verification)

Since interactive GUI work cannot be performed in this terminal environment, the following manual steps demonstrate the workflow using a small reproducible dataset.

### Preparation
```bash
# Clone or navigate to the MetaStart repository
cd /Users/anishr/MetaStart

# Create a small reproducible test dataset (3 features, 2 classes, 8 rows)
cat > demo.csv << 'EOF'
feature1,feature2,feature3,target
5.1,3.5,1.4,0
4.9,3.0,1.4,0
4.7,3.2,1.3,0
4.6,3.1,1.5,0
5.0,3.6,1.4,1
5.4,3.9,1.7,1
4.6,3.4,1.4,1
5.0,3.4,1.5,1
EOF
```

### Activation and Setup
```bash
# Activate the main virtual environment (contains core dependencies)
source .venv/bin/activate

# Install core dependencies if not already installed (should be present)
pip install -r requirements.txt

# For external AutoML comparison (optional), activate the AutoML environment:
#   source .venv-automl312/bin/activate
#   pip install -r requirements-external-automl.txt
# Then return to the main environment for the app:
#   source .venv/bin/activate
```

### Running the Application
```bash
# Start the Streamlit app
streamlit run app.py
```

### Manual Workflow Steps (perform in the browser UI)
1. **Upload Dataset**
   - Click "Browse files" and select `demo.csv`
   - Verify file size and parsing messages appear

2. **Select Target Column**
   - Choose the "target" column from the dropdown
   - Click "Recommend, train and evaluate"

3. **Observe Outputs**
   - Training dataset profile table (nine meta-features)
   - MetaStart ranking bar chart and table (predicted Macro-F1)
   - Download button for recommendation ranking CSV
   - Candidate selection progress bar and status messages
   - Upon completion:
     - Selected algorithm announced (e.g., "Selected by training-only CV: SVM")
     - Training-partition diagnostics tables
     - Class-wise out-of-fold performance
     - Held-out test metrics table for selected candidate
     - Download buttons for evaluation results and selected model (.joblib)
     - Algorithm comparison table with expandable details
     - Selection rationale text

4. **Run Optional Experiments**
   - Expand "Manual baseline and hyperparameter searches"
     - Click "Run manual baseline, Grid Search, Random Search, and Optuna"
     - Wait for completion; view summary table and add records to experiment logs
   - Expand "External AutoML framework comparison"
     - Set time budget per framework (e.g., 2 minutes)
     - Click "Run external AutoML comparison"
     - Wait for completion; view framework leaderboards and summary
   - Expand "Deliberate data-leakage demonstration"
     - Read the warning about invalid real-world performance
     - Click "Run leakage demonstration"
     - View results: valid baseline vs. deliberately leaky run (clearly labeled)
   - Expand "Bias evaluation and model explainability"
     - Adjust sliders for CV sample and explanation sample if desired
     - Click "Run bias evaluation and explanations"
     - View class imbalance ratio, confusion matrix, permutation importance, local explanation

5. **Download Artifacts**
   - In the "Experiment results export" section:
     - Download comprehensive experiment CSV
     - Download detailed metrics CSV
     - Download JSON report
     - Download ZIP bundle
   - If a model card was generated (requires successful model selection and fit):
     - Download model card (Markdown)
   - Also available:
     - Download full recommendation ranking (CSV)
     - Download evaluation results (CSV)
     - Download selected trained model (.joblib)

6. **Verify Consistency**
   - Open downloaded files and check:
     - Model card contains sections matching the template
     - Metrics in model card match those in evaluation results and experiment CSV
     - Recommendation score is labeled as predicted/not measured
     - CV and held-out metrics are distinct and correctly sourced
     - Leakage demonstration results are excluded from model card performance metrics
     - Explainability sections note that methods report association/sensitivity, not causation
     - ZIP bundle contains all expected files and can be extracted successfully

### Cleanup
```bash
# Remove the demo dataset
rm demo.csv
# Deactivate the virtual environment if desired
deactivate
```

## 3. Prioritized Submission Checklist

| IA-3 Rubric Item                                      | Status       | Evidence (Exact File Path)                                                                                                                              | What You Must Personally Submit/Demonstrate                                                                                                                         |
|-------------------------------------------------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **A. Certification**                                  | Unable to verify | N/A (external submission)                                                                                                                               | Submit proof of 100% completion of the assigned online course and the certificate.                                                                                 |
| **B.1 Dataset approval & sufficient size**            | Complete     | `data/test_datasets/adult_income.csv` (3.5M rows), `data/test_datasets/forest_cover.csv` (75M rows); app supports ≤750k rows, ≤200 MiB, ≤100 features | Demonstrate that your chosen dataset was approved for the project and meets size requirements (screenshots/upload logs).                                               |
| **B.2 Manual baseline, Grid Search, RS, Bayes opt**   | Complete     | `src/advanced_experiments.py`; `tests/test_advanced_experiments.py`; UI expander "Manual baseline and hyperparameter searches"                           | Show that you ran these searches on your dataset (screenshots of the expander with run button clicked and summary table populated).                                   |
| **B.3 Two external AutoML frameworks**                | Complete     | `src/external_automl.py`; `tests/test_external_automl.py`; UI expander "External AutoML framework comparison"                                          | Provide leaderboards/summary tables from FLAML and AutoGluon runs on your dataset (screenshots or exported CSV/JSON). Frameworks may be run in separate venv. |
| **B.4 Leakage, imbalance/bias, explainability**       | Complete     | Leakage: `src/leakage_experiment.py`, `tests/test_leakage_experiment.py`<br>Imbalance/bias: `src/bias_explainability.py`, `tests/test_bias_explainability.py`<br>Explainability: permutation importance & local ablation (in model card and bias explainability output) | Show leakage demo results clearly marked "INVALID FOR REAL-WORLD PERFORMANCE"; show imbalance ratio and flag; show permutation importance and local explanation outputs. |
| **B.5 Best-model export, deployment/demo, model card**| Complete     | Best-model: app lines 221-224 (`metastart_selected_model.joblib` download)<br>Deployment/demo: the Streamlit app (`app.py`)<br>Model card: `src/model_card.py`, download button in app lines 663-668 | Provide: downloaded selected model (.joblib), screenshots of the running app (deployment/demo), and downloaded model card (Markdown).                                  |
| **B.6 Report, screenshots, comparison tables, pres.**| Partial      | **Report**: app generates experiment exports (CSV, JSON, ZIP) and model card (see `src/experiment_results.py`, `src/model_card.py`)<br>**Comparison tables**: UI algorithm comparison (app lines 226-246) and exported per-algorithm metrics<br>**Screenshots, presentation, reflection**: external artifacts | You must provide: <br>- Screenshots of key workflow steps (upload, ranking, CV results, held-out metrics, optional experiment tabs, downloads)<br>- A presentation (slides or PDF) summarizing the project, methodology, results, and reflections<br>- A written reflection on outcomes, limitations, and lessons learned<br>- The experiment exports (CSV, JSON, ZIP) and model card as part of the report appendix |

## 4. Remaining Blockers and Environment Constraints

- **Certification (A)** is an external requirement; no code changes can satisfy it. You must complete the online course and submit the certificate separately.
- **External AutoML frameworks** require Python 3.10–3.13 for AutoGluon; the main environment uses Python 3.14.7. To run both frameworks without replacing the app environment, use the separate `.venv-automl312` as described in the README. If you do not wish to set up this environment, you may skip the external AutoML comparison, but the rubric requires at least two external frameworks. The app handles missing frameworks gracefully (reports `unsupported_python` or `missing_optional_framework`).
- **Interactive verification**: Because this session is terminal-only, you must perform the manual workflow steps yourself in a local GUI environment to capture screenshots and download artifacts.
- **No modifications needed**: The existing implementation on branch `feature/ia3-complete-requirements` satisfies all software-related IA-3 rubric requirements. All tests pass (84/84). The model card is generated from actual run metadata without fabrication.
- **Artifact preservation**: Do not commit or push changes. The generated artifacts (CSV, JSON, ZIP, .joblib, model card) are transient; preserve them by downloading during your demo run.

### Next Steps
1. Complete the online course and obtain the certificate.
2. Perform the manual demo workflow using your approved dataset (or the provided `demo.csv` for quick verification).
3. Capture screenshots at each key stage and download all artifacts.
4. Assemble the report, presentation, and reflection using the downloaded artifacts as evidence.
5. Ensure all rubric items are addressed in your submission package.