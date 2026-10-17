"""Versioned A/B/C experiment for MetaStart profile feature sets.

This experiment never rewrites the frozen benchmark, baseline model, or baseline
evaluation. Outputs are written under a new experiment ID and existing output
directories are rejected to prevent accidental replacement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from src.benchmark import ALGORITHM_NAMES, ROOT, get_datasets
from src.dataset_profiler import (ADDITIONAL_META_FEATURES, META_FEATURES,
                                  PROFILE_FEATURES, profile_dataset)
from src.meta_learner import build_meta_model
from src.provenance import filter_trainable_records


EXPERIMENT_ID = "feature_abc_v1_20261014"
NUMERIC_EXTENSION_FEATURES = [
    "normalized_class_entropy",
    "log10_samples_per_feature",
    "median_abs_numeric_skewness",
    "median_abs_numeric_excess_kurtosis",
    "mean_abs_numeric_correlation",
    "fraction_numeric_pairs_abs_corr_ge_0_8",
]
FEATURE_SETS = {
    "A": {"name": "original9", "features": list(META_FEATURES)},
    "B": {"name": "numeric15",
          "features": [*META_FEATURES, *NUMERIC_EXTENSION_FEATURES]},
    "C": {"name": "full17", "features": list(PROFILE_FEATURES)},
}
RANDOM_TRIALS = 1000
BOOTSTRAP_REPLICATES = 5000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profiles_for_eligible_identities(identities: list[str]) -> pd.DataFrame:
    profiles = {}
    for name, X, y in get_datasets():
        if name in identities:
            profiles[name] = profile_dataset(X, y)
    missing = set(identities) - set(profiles)
    if missing:
        raise ValueError(f"Eligible dataset identities lack reproducible local inputs: {sorted(missing)}")
    result = pd.DataFrame.from_dict(profiles, orient="index").loc[identities]
    if list(result.columns) != PROFILE_FEATURES:
        raise ValueError("Profile feature order differs from the declared 17-feature contract")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Profile matrix contains non-finite feature values")
    return result


def _evaluate_feature_set(name: str, features: list[str], data: pd.DataFrame,
                          identities: list[str], random_baselines: dict[str, tuple[float, float]],
                          random_state: int = 42):
    rows, candidate_rows = [], []
    for heldout in identities:
        train = data[data.dataset != heldout].copy()
        test = data[data.dataset == heldout].copy()
        model = build_meta_model(random_state=random_state)
        model.fit(train[features + ["algorithm"]], train["macro_f1"])
        test["predicted_macro_f1"] = model.predict(test[features + ["algorithm"]])
        test = test.set_index("algorithm").loc[ALGORITHM_NAMES].reset_index()
        ranked = test.sort_values("predicted_macro_f1", ascending=False)
        chosen = ranked.iloc[0]
        actual_ranked = test.sort_values("macro_f1", ascending=False)
        best_score = float(test.macro_f1.max())
        historical_algorithm = train.groupby("algorithm").macro_f1.mean().idxmax()
        historical_score = float(test.loc[test.algorithm == historical_algorithm,
                                           "macro_f1"].iloc[0])
        fixed_score = float(test.loc[test.algorithm == "Random Forest", "macro_f1"].iloc[0])
        random_mean, random_sd = random_baselines[heldout]
        predicted_top3 = set(ranked.algorithm.head(3))
        actual_best_algorithm = actual_ranked.iloc[0].algorithm
        rows.append({
            "experiment": name,
            "dataset": heldout,
            "selected_algorithm": chosen.algorithm,
            "selected_f1": float(chosen.macro_f1),
            "best_f1": best_score,
            "meta_regret": best_score - float(chosen.macro_f1),
            "meta_top1_hit": int(float(chosen.macro_f1) == best_score),
            "meta_top3_hit": int(actual_best_algorithm in predicted_top3),
            "fixed_rf_f1": fixed_score,
            "fixed_rf_regret": best_score - fixed_score,
            "historical_algorithm": historical_algorithm,
            "historical_f1": historical_score,
            "historical_regret": best_score - historical_score,
            "random_expected_f1": random_mean,
            "random_regret": best_score - random_mean,
            "random_f1_sd": random_sd,
            "prediction_mae": float(mean_absolute_error(test.macro_f1,
                                                         test.predicted_macro_f1)),
            "ranking_spearman": float(spearmanr(test.macro_f1,
                                                 test.predicted_macro_f1).statistic),
        })
        for _, candidate in test.iterrows():
            candidate_rows.append({
                "experiment": name,
                "dataset": heldout,
                "algorithm": candidate.algorithm,
                "observed_macro_f1": float(candidate.macro_f1),
                "predicted_macro_f1": float(candidate.predicted_macro_f1),
            })
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def _summarize(reports: dict[str, pd.DataFrame], identities: list[str]):
    metrics = ["meta_regret", "fixed_rf_regret", "historical_regret", "random_regret",
               "meta_top1_hit", "meta_top3_hit", "prediction_mae", "ranking_spearman"]
    rng = np.random.default_rng(2026)
    bootstrap_indices = rng.integers(0, len(identities),
                                     size=(BOOTSTRAP_REPLICATES, len(identities)))
    rows = []
    for variant, report in reports.items():
        report = report.set_index("dataset").loc[identities]
        for metric in metrics:
            values = report[metric].to_numpy(dtype=float)
            means = values[bootstrap_indices].mean(axis=1)
            rows.append({
                "experiment": variant,
                "metric": metric,
                "mean": float(values.mean()),
                "dataset_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "bootstrap_95ci_low": float(np.quantile(means, 0.025)),
                "bootstrap_95ci_high": float(np.quantile(means, 0.975)),
                "n_identities": len(values),
            })
    summary = pd.DataFrame(rows)

    # Paired identity-level regret differences use identical bootstrap draws.
    paired_rows = []
    indexed = {key: value.set_index("dataset").loc[identities] for key, value in reports.items()}
    for variant in ("B", "C"):
        for metric in ("meta_regret", "meta_top1_hit", "meta_top3_hit"):
            delta = (indexed[variant][metric] - indexed["A"][metric]).to_numpy(dtype=float)
            means = delta[bootstrap_indices].mean(axis=1)
            paired_rows.append({
                "contrast": f"{variant}-A",
                "metric": metric,
                "mean_difference": float(delta.mean()),
                "bootstrap_95ci_low": float(np.quantile(means, 0.025)),
                "bootstrap_95ci_high": float(np.quantile(means, 0.975)),
                "n_identities": len(delta),
            })
    return summary, pd.DataFrame(paired_rows)


def run_feature_experiment(experiment_id: str = EXPERIMENT_ID):
    data_path = ROOT / "data/meta_dataset.csv"
    provenance_path = ROOT / "results/dataset_provenance.csv"
    baseline_report_path = ROOT / "results/evaluation_report.csv"
    baseline_model_path = ROOT / "models/meta_model.joblib"
    freeze_path = ROOT / "results/experiment_freeze.json"
    results_dir = ROOT / "results/feature_experiments" / experiment_id
    models_dir = ROOT / "models/feature_experiments" / experiment_id
    if results_dir.exists() or models_dir.exists():
        raise FileExistsError("Experiment output directory already exists; choose a new experiment_id")

    freeze = json.loads(freeze_path.read_text())
    input_sha = _sha256(data_path)
    provenance_sha = _sha256(provenance_path)
    if input_sha != freeze["input_sha256"]:
        raise ValueError("Benchmark input does not match the frozen baseline hash")
    expected_baseline = freeze["output_sha256"]
    if _sha256(baseline_model_path) != expected_baseline["models/meta_model.joblib"]:
        raise ValueError("Frozen baseline model hash does not match experiment freeze")
    if _sha256(baseline_report_path) != expected_baseline["results/evaluation_report.csv"]:
        raise ValueError("Frozen baseline report hash does not match experiment freeze")

    frozen = pd.read_csv(data_path)
    eligible = filter_trainable_records(frozen, provenance_path)
    identities = sorted(eligible.dataset.unique().tolist())
    if identities != freeze["eligible_dataset_identities"]:
        raise ValueError("Eligible dataset identities differ from the frozen baseline")
    if sorted(eligible.algorithm.unique().tolist()) != freeze["algorithms"]:
        raise ValueError("Algorithm set differs from the frozen baseline")
    pair_counts = eligible.groupby("dataset").algorithm.agg(list)
    if any(len(names) != 5 or set(names) != set(ALGORITHM_NAMES) for names in pair_counts):
        raise ValueError("Each eligible identity must have exactly one row per candidate algorithm")

    profiles = _profiles_for_eligible_identities(identities)
    frozen_profiles = eligible.groupby("dataset")[META_FEATURES].first().loc[identities]
    if not np.allclose(profiles[META_FEATURES], frozen_profiles, rtol=0, atol=1e-12):
        raise ValueError("Recomputed legacy meta-features differ from the frozen benchmark profiles")
    score_rows = eligible[["dataset", "algorithm", "macro_f1"]]
    data = score_rows.merge(profiles.reset_index(names="dataset"), on="dataset",
                            how="left", validate="many_to_one")
    if data.isna().any().any():
        raise ValueError("Feature experiment input contains missing values")

    # Precompute the same random-choice baseline per identity for all variants.
    rng = np.random.default_rng(42)
    random_baselines = {}
    for identity in identities:
        values = (eligible[eligible.dataset == identity].set_index("algorithm")
                  .loc[ALGORITHM_NAMES, "macro_f1"].to_numpy(dtype=float))
        draws = rng.choice(values, size=RANDOM_TRIALS, replace=True)
        random_baselines[identity] = (float(draws.mean()), float(draws.std()))

    reports, predictions = {}, []
    for variant, spec in FEATURE_SETS.items():
        report, candidate_predictions = _evaluate_feature_set(
            spec["name"], spec["features"], data, identities, random_baselines
        )
        reports[variant] = report
        predictions.append(candidate_predictions)

    # Confirm config A reproduces the frozen original-nine evaluation decisions.
    frozen_report = pd.read_csv(baseline_report_path).set_index("dataset").loc[identities]
    report_a = reports["A"].set_index("dataset").loc[identities]
    for current_column, frozen_column in (
        ("selected_algorithm", "selected_algorithm"),
        ("selected_f1", "meta_f1"),
        ("best_f1", "best_f1"),
        ("meta_regret", "meta_regret"),
        ("meta_top1_hit", "meta_top1_hit"),
        ("meta_top3_hit", "meta_top3_hit"),
    ):
        if current_column == "selected_algorithm":
            same = report_a[current_column].equals(frozen_report[frozen_column])
        else:
            same = np.allclose(report_a[current_column].to_numpy(dtype=float),
                               frozen_report[frozen_column].to_numpy(dtype=float),
                               rtol=0, atol=1e-12)
        if not same:
            raise ValueError(f"Experiment A does not reproduce frozen baseline field {frozen_column}")

    summary, paired = _summarize(reports, identities)
    # All checks pass; create new, versioned output trees without replacing baseline files.
    results_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    (results_dir / "feature_sets.json").write_text(json.dumps({
        key: {"name": value["name"], "features": value["features"]}
        for key, value in FEATURE_SETS.items()
    }, indent=2) + "\n")

    baseline_artifact = joblib.load(baseline_model_path)
    baseline_artifact = dict(baseline_artifact)
    baseline_artifact.update({"experiment_id": experiment_id,
                              "feature_set_version": "A_original9_v1"})
    joblib.dump(baseline_artifact, models_dir / "meta_model_A_original9_v1.joblib")
    for variant in ("B", "C"):
        spec = FEATURE_SETS[variant]
        full_model = build_meta_model()
        full_model.fit(data[spec["features"] + ["algorithm"]], data["macro_f1"])
        joblib.dump({
            "model": full_model,
            "algorithms": list(ALGORITHM_NAMES),
            "meta_features": list(spec["features"]),
            "experiment_id": experiment_id,
            "feature_set_version": f"{variant}_{spec['name']}_v1",
            "input_sha256": input_sha,
            "provenance_sha256": provenance_sha,
        }, models_dir / f"meta_model_{variant}_{spec['name']}_v1.joblib")

    pd.concat(reports.values(), ignore_index=True).to_csv(
        results_dir / "evaluation_by_identity.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        results_dir / "candidate_predictions.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    paired.to_csv(results_dir / "paired_differences.csv", index=False)
    profiles.reset_index(names="dataset").to_csv(results_dir / "profiles.csv", index=False)
    manifest = {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_input_sha256": input_sha,
        "provenance_registry_sha256": provenance_sha,
        "frozen_baseline_model_sha256": expected_baseline["models/meta_model.joblib"],
        "frozen_baseline_report_sha256": expected_baseline["results/evaluation_report.csv"],
        "eligible_identities": identities,
        "excluded_identities": freeze["excluded_dataset_identities"],
        "algorithms": list(ALGORITHM_NAMES),
        "records": int(len(data)),
        "folding": "leave one complete dataset identity out",
        "full_dataset_profiles": True,
        "random_trials": RANDOM_TRIALS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_intervals": "descriptive; ten identities include six synthetic datasets",
        "baseline_reproduced": True,
        "feature_sets": {key: {"name": value["name"], "features": value["features"]}
                         for key, value in FEATURE_SETS.items()},
    }
    (results_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.pivot(index="metric", columns="experiment", values="mean").to_string())
    print("\nPaired differences from A:")
    print(paired.to_string(index=False))
    print(f"\nVersioned outputs: {results_dir}")
    print(f"Versioned models: {models_dir}")
    return summary, paired


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    run_feature_experiment(args.experiment_id)
