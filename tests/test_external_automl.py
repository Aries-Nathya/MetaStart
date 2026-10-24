import io
import json
import sys
import zipfile

import numpy as np
import pandas as pd

import src.external_automl as external
from src.evaluation_metrics import compute_measured_metrics
from src.experiment_results import build_experiment_downloads


class _FakePredictor:
    classes_ = np.array(["no", "yes"])

    def __init__(self, predicted):
        self.predicted = np.asarray(predicted)
        self.predicted_X = None

    def predict(self, X):
        self.predicted_X = X
        return self.predicted

    def predict_proba(self, X):
        return np.tile([0.25, 0.75], (len(X), 1))


def _dataset():
    X_train = pd.DataFrame({"x": [0, 1, 2, 3]})
    y_train = pd.Series(["no", "yes", "no", "yes"])
    X_test = pd.DataFrame({"x": [4, 5]})
    y_test = pd.Series(["no", "yes"])
    return X_train, y_train, X_test, y_test


def test_external_results_use_matching_train_test_split_and_export_leaderboards(monkeypatch):
    X_train, y_train, X_test, y_test = _dataset()
    models = {}
    calls = []

    def fake_runner(framework, score):
        def run(train_X, train_y, *, time_budget_seconds, seed, cv_folds):
            calls.append((framework, train_X, train_y, time_budget_seconds, seed, cv_folds))
            model = _FakePredictor(["no", "yes"])
            models[framework] = model
            leaderboard = pd.DataFrame([{
                "Framework": framework,
                "Model": f"{framework}Model",
                "CV Macro-F1": score,
                "Fit Seconds": 1.25,
                "Parameters": {"depth": 4},
                "Framework Metrics": {"score_val": score, "fit_time": 1.25},
            }])
            run = external.FrameworkRun(
                framework, "completed", leaderboard, f"{framework}Model",
                {"depth": 4}, score, model, 5.0,
            )
            if framework == "AutoGluon":
                run.model_failures = ({
                    "Model": "UnavailableModel", "Exception": "ImportError",
                    "Details": "optional package missing", "Fit Seconds": 0.1,
                },)
            return run
        return run

    monkeypatch.setattr(external, "_run_flaml", fake_runner("FLAML", 0.7))
    monkeypatch.setattr(external, "_run_autogluon", fake_runner("AutoGluon", 0.8))
    records, summary, leaderboards = external.run_external_automl_comparison(
        X_train, y_train, X_test, y_test,
        time_budget_seconds=30, seed=13, cv_folds=2,
    )

    assert all(call[1] is X_train and call[2] is y_train for call in calls)
    assert all(call[3:] == (30, 13, 2) for call in calls)
    assert all(model.predicted_X is X_test for model in models.values())
    assert summary["Status"].eq("completed").all()
    assert summary["Held-out Macro-F1"].tolist() == [1.0, 1.0]
    assert leaderboards["FLAML"].loc[0, "CV Macro-F1"] == 0.7
    assert leaderboards["AutoGluon"].loc[0, "CV Macro-F1"] == 0.8

    downloads = build_experiment_downloads(records, {"time_budget_seconds": 30})
    flat = pd.read_csv(io.BytesIO(downloads["experiment_results.csv"]))
    detailed = pd.read_csv(io.BytesIO(downloads["experiment_detailed_metrics.csv"]))
    report = json.loads(downloads["experiment_report.json"])
    assert set(flat["Stage"]) == {"external_automl", "external_automl_held_out_test"}
    assert flat.loc[flat["Stage"].eq("external_automl"), "Framework Metrics"].notna().all()
    failed = flat.loc[flat["Status"].eq("model_failed")].iloc[0]
    assert failed["Algorithm"] == "AutoGluon: UnavailableModel"
    assert pd.isna(failed["CV Macro F1"])
    assert len(detailed) == 4
    assert any(row["leaderboard_metrics"].get("score_val") == 0.7
               for row in report["experiments"])
    with zipfile.ZipFile(io.BytesIO(downloads["experiment_results_bundle.zip"])) as archive:
        assert archive.testzip() is None
        assert pd.read_csv(io.BytesIO(archive.read("experiment_results.csv"))).shape[0] == len(records)


def test_missing_optional_frameworks_are_reported_without_fabricated_scores(monkeypatch):
    X_train, y_train, X_test, y_test = _dataset()
    monkeypatch.setitem(sys.modules, "flaml", None)
    monkeypatch.setattr(external.sys, "version_info", (3, 13, 0))
    monkeypatch.setitem(sys.modules, "autogluon", None)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", None)

    records, summary, leaderboards = external.run_external_automl_comparison(
        X_train, y_train, X_test, y_test, time_budget_seconds=30,
    )

    assert summary["Status"].tolist() == ["missing_dependency", "missing_dependency"]
    assert summary["Training CV Macro-F1"].isna().all()
    assert summary["Held-out Macro-F1"].isna().all()
    assert all(leaderboard.empty for leaderboard in leaderboards.values())
    assert all(record.status == "missing_dependency" for record in records)
    assert all(record.primary_value is None for record in records)


def test_measured_metrics_handle_numeric_class_labels_for_external_predictions():
    metrics = compute_measured_metrics(
        [0, 1, 0, 1], [0, 1, 0, 1], labels=["0", "1"],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_autogluon_runner_uses_train_only_and_numeric_memory_limit(monkeypatch):
    X_train, y_train, X_test, y_test = _dataset()
    captured = {}

    class FakeTabularPredictor:
        model_best = "RF_BAG_L1"

        def __init__(self, **kwargs):
            captured["predictor_kwargs"] = kwargs

        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs
            return self

        def leaderboard(self, *, silent):
            assert silent is True
            return pd.DataFrame([{
                "model": "RF_BAG_L1", "score_val": 0.8, "fit_time": 1.0,
            }])

        def model_hyperparameters(self, model):
            return {"RF": {"n_estimators": 10}}

        def model_failures(self):
            return pd.DataFrame(columns=["model", "exc_type", "exc_str", "total_time"])

    monkeypatch.setattr(external.sys, "version_info", (3, 12, 0))
    monkeypatch.setitem(sys.modules, "autogluon", __import__("types").ModuleType("autogluon"))
    fake_tabular = __import__("types").ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    run = external._run_autogluon(
        X_train, y_train, time_budget_seconds=30, seed=42, cv_folds=3,
    )

    assert run.status == "completed"
    train_data = captured["fit_kwargs"]["train_data"]
    assert train_data.shape[0] == len(X_train)
    pd.testing.assert_frame_equal(train_data.iloc[:, :-1], X_train.reset_index(drop=True))
    assert train_data.iloc[:, -1].tolist() == y_train.reset_index(drop=True).tolist()
    assert "memory_limit" in captured["fit_kwargs"]
    assert captured["fit_kwargs"]["memory_limit"] == external.EXTERNAL_MEMORY_LIMIT_GB
    assert isinstance(captured["fit_kwargs"]["memory_limit"], (int, float))
    assert run.selected_cv_macro_f1 == 0.8
