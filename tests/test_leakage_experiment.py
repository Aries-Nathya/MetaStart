import io
import json
import zipfile

import pandas as pd
import pytest

import src.app_workflow as app_workflow
import src.recommender as recommender
from src.leakage_experiment import (
    LEAKAGE_DEMONSTRATION_LABEL,
    run_leakage_demonstration,
)
from src.experiment_results import build_experiment_downloads


def test_leakage_demo_is_reproducible_and_separate_from_production_selection(monkeypatch):
    def production_path_must_not_run(*args, **kwargs):
        pytest.fail("The isolated leakage demo called the production recommendation path.")

    monkeypatch.setattr(recommender, "recommend_algorithms", production_path_must_not_run)
    monkeypatch.setattr(app_workflow, "evaluate_ranked_candidates", production_path_must_not_run)
    first_records, first_summary = run_leakage_demonstration()
    second_records, second_summary = run_leakage_demonstration()

    assert first_summary["Macro-F1"].tolist() == second_summary["Macro-F1"].tolist()
    assert [record.stage for record in first_records] == ["leakage", "leakage"]
    assert all(record.method != "recommendation" for record in first_records)
    assert all(record.algorithm.startswith("Decision Tree") for record in first_records)
    assert all(record.test_rows == 143 and record.seed == 42 for record in first_records)
    assert all(record.fit_seconds is not None and record.fit_seconds >= 0 for record in first_records)
    assert all(record.test_seconds is not None and record.test_seconds >= 0 for record in first_records)
    assert first_records[0].status == "completed"
    assert first_records[0].leak is False
    assert first_records[0].invalid_for_real_world is False
    assert first_records[1].status == "invalid_for_real_world"
    assert first_records[1].leak is True
    assert first_records[1].invalid_for_real_world is True
    assert first_records[1].metrics["macro_f1"] == 1.0
    assert first_records[1].leaderboard_metrics["held_out_target_used_as_feature"] is True
    assert LEAKAGE_DEMONSTRATION_LABEL in " ".join(first_records[1].notes)
    assert "selected_algorithm" not in first_summary.columns


def test_leakage_flags_and_measured_values_are_present_in_all_exports():
    records, summary = run_leakage_demonstration()
    downloads = build_experiment_downloads(
        records,
        {"leakage_demo": True, "label": LEAKAGE_DEMONSTRATION_LABEL},
    )

    flat = pd.read_csv(io.BytesIO(downloads["experiment_results.csv"]))
    assert flat["Stage"].tolist() == ["leakage", "leakage"]
    assert flat["Status"].tolist() == ["completed", "invalid_for_real_world"]
    assert flat["Leak"].tolist() == [False, True]
    assert flat["Invalid for Real-World"].tolist() == [False, True]
    assert flat.loc[1, "Macro F1"] == summary.loc[1, "Macro-F1"] == 1.0
    assert flat["Test Scoring Seconds"].notna().all()
    assert LEAKAGE_DEMONSTRATION_LABEL in flat.loc[1, "Notes"]

    detailed = pd.read_csv(io.BytesIO(downloads["experiment_detailed_metrics.csv"]))
    assert set(detailed["Stage"]) == {"leakage"}
    assert set(detailed.loc[detailed["Status"].eq("invalid_for_real_world"), "Invalid for Real-World"]) == {True}

    report = json.loads(downloads["experiment_report.json"])
    leaky = report["experiments"][1]
    assert leaky["invalid_for_real_world"] is True
    assert leaky["leak"] is True
    assert leaky["leaderboard_metrics"]["held_out_target_used_as_feature"] is True
    assert leaky["metrics"]["macro_f1"] == 1.0

    with zipfile.ZipFile(io.BytesIO(downloads["experiment_results_bundle.zip"])) as archive:
        assert archive.testzip() is None
        zipped = pd.read_csv(io.BytesIO(archive.read("experiment_results.csv")))
        assert zipped.loc[1, "Status"] == "invalid_for_real_world"
