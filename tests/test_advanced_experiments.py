import pandas as pd
from sklearn.datasets import load_iris

from src.advanced_experiments import _bounded_cv_sample, run_tuning_experiments
from src.app_workflow import CV_MAX_ROWS, CV_RANDOM_STATE, _stratified_cv_sample


def test_manual_grid_random_and_optuna_searches_produce_training_cv_records():
    dataset = load_iris(as_frame=True)
    records, summary = run_tuning_experiments(
        dataset.data, pd.Series(dataset.target), "Random Forest", trials=2,
        experiment_id="test-iris",
    )

    assert [record.method for record in records] == [
        "manual_baseline", "grid_search", "random_search", "optuna",
    ]
    assert all(record.status == "completed" for record in records)
    assert all(0.0 <= record.primary_value <= 1.0 for record in records)
    assert all(record.cv_folds == 3 and record.cv_rows == len(dataset.data) for record in records)
    assert records[0].configuration == "untuned"
    assert records[2].trials_evaluated == 2
    assert records[3].trials_evaluated == 2
    assert set(summary["Experiment"]) == {
        "Manual Baseline", "Grid Search", "Random Search", "Optuna",
    }


def test_advanced_sampler_matches_workflow_sample_without_full_frame_copy():
    X = pd.DataFrame({"feature": range(CV_MAX_ROWS + 20)})
    y = pd.Series(["a"] * 4000 + ["b"] * (CV_MAX_ROWS + 20 - 4000))

    expected_X, expected_y = _stratified_cv_sample(
        X, y, max_rows=CV_MAX_ROWS, min_per_class=3, random_state=CV_RANDOM_STATE,
    )
    sampled_X, sampled_y = _bounded_cv_sample(
        X, y, max_rows=CV_MAX_ROWS, min_per_class=3,
    )

    pd.testing.assert_frame_equal(sampled_X, expected_X)
    pd.testing.assert_series_equal(sampled_y, expected_y)
