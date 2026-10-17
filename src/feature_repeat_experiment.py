"""Repeated-seed feature comparison plus a constant-column control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import ALGORITHM_NAMES, ROOT
from src.dataset_profiler import META_FEATURES, PROFILE_FEATURES
from src.feature_experiment import (BOOTSTRAP_REPLICATES, FEATURE_SETS,
                                    NUMERIC_EXTENSION_FEATURES,
                                    _evaluate_feature_set,
                                    _profiles_for_eligible_identities)
from src.meta_learner import build_meta_model
from src.provenance import filter_trainable_records


EXPERIMENT_ID = "feature_abc_repeat_v1_20261014"
MODEL_SEEDS = (0, 1, 2, 3, 4, 7, 11, 21, 42, 100)
CONTROL_FEATURES = [*FEATURE_SETS["B"]["features"],
                    "__constant_control_one", "__constant_control_two"]
CONFIGURATIONS = {
    **FEATURE_SETS,
    "D": {"name": "numeric15_plus_two_constants", "features": CONTROL_FEATURES},
}
METRICS = ["meta_regret", "fixed_rf_regret", "historical_regret", "random_regret",
           "meta_top1_hit", "meta_top3_hit", "prediction_mae", "ranking_spearman"]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_summary(reports: pd.DataFrame, identities: list[str], seed: int):
    per_identity = (reports.groupby(["experiment", "dataset"], as_index=False)[METRICS]
                    .mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(identities), size=(BOOTSTRAP_REPLICATES, len(identities)))
    summary_rows = []
    for config in CONFIGURATIONS:
        subset = per_identity[per_identity.experiment == config].set_index("dataset").loc[identities]
        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            means = values[indices].mean(axis=1)
            summary_rows.append({
                "experiment": config,
                "metric": metric,
                "mean_across_identities": float(values.mean()),
                "identity_sd": float(values.std(ddof=1)),
                "bootstrap_95ci_low": float(np.quantile(means, 0.025)),
                "bootstrap_95ci_high": float(np.quantile(means, 0.975)),
                "n_independent_identities": len(values),
                "model_seed_count": len(MODEL_SEEDS),
            })

    paired_rows = []
    indexed = {config: per_identity[per_identity.experiment == config]
               .set_index("dataset").loc[identities] for config in CONFIGURATIONS}
    for left, right in (("B", "A"), ("C", "A"), ("D", "B"), ("D", "C")):
        for metric in ("meta_regret", "meta_top1_hit", "meta_top3_hit"):
            delta = (indexed[left][metric] - indexed[right][metric]).to_numpy(dtype=float)
            means = delta[indices].mean(axis=1)
            paired_rows.append({
                "contrast": f"{left}-{right}",
                "metric": metric,
                "mean_difference": float(delta.mean()),
                "bootstrap_95ci_low": float(np.quantile(means, 0.025)),
                "bootstrap_95ci_high": float(np.quantile(means, 0.975)),
                "n_independent_identities": len(delta),
            })

    real_ids = [name for name in identities if not name.startswith("synthetic_")]
    real_rows = []
    for config in CONFIGURATIONS:
        subset = per_identity[(per_identity.experiment == config)
                              & per_identity.dataset.isin(real_ids)].set_index("dataset").loc[real_ids]
        for metric in ("meta_regret", "meta_top1_hit", "meta_top3_hit"):
            real_rows.append({
                "experiment": config,
                "metric": metric,
                "mean_across_real_identities": float(subset[metric].mean()),
                "n_real_identities": len(real_ids),
                "model_seed_count": len(MODEL_SEEDS),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows), pd.DataFrame(real_rows), per_identity


def run_repeated_feature_experiment(experiment_id: str = EXPERIMENT_ID):
    freeze = json.loads((ROOT / "results/experiment_freeze.json").read_text())
    data_path = ROOT / "data/meta_dataset.csv"
    provenance_path = ROOT / "results/dataset_provenance.csv"
    baseline_report_path = ROOT / "results/evaluation_report.csv"
    baseline_report_hash = freeze["output_sha256"]["results/evaluation_report.csv"]
    results_dir = ROOT / "results/feature_experiments" / experiment_id
    if results_dir.exists():
        raise FileExistsError("Experiment output directory already exists; choose a new experiment_id")
    if _hash(data_path) != freeze["input_sha256"]:
        raise ValueError("Benchmark input differs from the frozen baseline")
    if _hash(baseline_report_path) != baseline_report_hash:
        raise ValueError("Frozen evaluation report differs from its recorded hash")

    eligible = filter_trainable_records(pd.read_csv(data_path), provenance_path)
    identities = sorted(freeze["eligible_dataset_identities"])
    if sorted(eligible.dataset.unique()) != identities:
        raise ValueError("Eligible dataset identities differ from the frozen baseline")
    if sorted(eligible.algorithm.unique()) != freeze["algorithms"]:
        raise ValueError("Algorithm set differs from the frozen baseline")
    if len(eligible) != len(identities) * len(ALGORITHM_NAMES):
        raise ValueError("Expected exactly one record per identity and algorithm")

    profiles = _profiles_for_eligible_identities(identities)
    original = eligible.groupby("dataset")[META_FEATURES].first().loc[identities]
    if not np.allclose(profiles[META_FEATURES], original, rtol=0, atol=1e-12):
        raise ValueError("Recomputed original features differ from frozen benchmark profiles")
    scored = eligible[["dataset", "algorithm", "macro_f1"]].merge(
        profiles.reset_index(names="dataset"), on="dataset", validate="many_to_one")
    if not np.isfinite(scored.select_dtypes(include=np.number).to_numpy()).all():
        raise ValueError("Experiment data contains non-finite values")

    # Independent controls use precisely the values supplied by C's two
    # categorical profile fields, which are verified constant in this cohort.
    for field in ("mean_normalized_categorical_entropy", "mean_categorical_cardinality"):
        if profiles[field].nunique() != 1:
            raise ValueError(f"Constant-feature control requires constant {field}")
    constant_value_1 = float(profiles["mean_normalized_categorical_entropy"].iloc[0])
    constant_value_2 = float(profiles["mean_categorical_cardinality"].iloc[0])
    control_data = scored.copy()
    control_data["__constant_control_one"] = constant_value_1
    control_data["__constant_control_two"] = constant_value_2

    rng = np.random.default_rng(42)
    random_baselines = {}
    for identity in identities:
        values = (eligible[eligible.dataset == identity].set_index("algorithm")
                  .loc[ALGORITHM_NAMES, "macro_f1"].to_numpy(dtype=float))
        draws = rng.choice(values, size=1000, replace=True)
        random_baselines[identity] = (float(draws.mean()), float(draws.std()))

    report_frames, prediction_frames = [], []
    for seed in MODEL_SEEDS:
        for config, spec in CONFIGURATIONS.items():
            data = control_data if config == "D" else scored
            report, predictions = _evaluate_feature_set(
                spec["name"], spec["features"], data, identities,
                random_baselines, random_state=seed,
            )
            report["experiment"] = config
            report["seed"] = seed
            predictions["experiment"] = config
            predictions["seed"] = seed
            report_frames.append(report)
            prediction_frames.append(predictions)
    reports = pd.concat(report_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    prediction_check = predictions.pivot_table(
        index=["seed", "dataset", "algorithm"], columns="experiment",
        values="predicted_macro_f1", aggfunc="first")
    control_delta = (prediction_check.C - prediction_check.D).abs()
    if not np.allclose(control_delta, 0.0, rtol=0, atol=1e-12):
        raise ValueError("C and the two-constant-column control produced different predictions")

    # Verify seed 42 reproduces the frozen A per-identity recommendations.
    frozen_report = pd.read_csv(baseline_report_path).set_index("dataset").loc[identities]
    seed_42 = reports[(reports.experiment == "A") & (reports.seed == 42)]
    seed_42 = seed_42.set_index("dataset").loc[identities]
    checks = [("selected_algorithm", "selected_algorithm"),
              ("selected_f1", "meta_f1"), ("best_f1", "best_f1"),
              ("meta_regret", "meta_regret"), ("meta_top1_hit", "meta_top1_hit"),
              ("meta_top3_hit", "meta_top3_hit")]
    for current, frozen in checks:
        if current == "selected_algorithm":
            equal = seed_42[current].equals(frozen_report[frozen])
        else:
            equal = np.allclose(seed_42[current], frozen_report[frozen], rtol=0, atol=1e-12)
        if not equal:
            raise ValueError(f"A seed 42 failed to reproduce the frozen report field {frozen}")

    summary, paired, real_summary, per_identity = _bootstrap_summary(reports, identities, 404)
    per_seed = (reports.groupby(["experiment", "seed"], as_index=False)[METRICS]
                .mean())
    seed_stability = (per_seed.groupby("experiment")[METRICS]
                      .agg(["mean", "std", "min", "max"]))

    # Persist only after the controls and frozen-baseline reproduction pass.
    results_dir.mkdir(parents=True)
    feature_sets = {key: {"name": value["name"], "features": value["features"]}
                    for key, value in CONFIGURATIONS.items()}
    (results_dir / "feature_sets.json").write_text(json.dumps(feature_sets, indent=2) + "\n")
    reports.to_csv(results_dir / "evaluation_by_seed_identity.csv", index=False)
    predictions.to_csv(results_dir / "candidate_predictions.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    paired.to_csv(results_dir / "paired_differences.csv", index=False)
    real_summary.to_csv(results_dir / "real_subset_summary.csv", index=False)
    per_seed.to_csv(results_dir / "per_seed_summary.csv", index=False)
    per_identity.to_csv(results_dir / "per_identity_seed_average.csv", index=False)
    seed_stability.to_csv(results_dir / "seed_stability.csv")
    pd.DataFrame({"max_abs_prediction_difference_C_vs_D": [float(control_delta.max())],
                  "all_predictions_identical": [bool(np.allclose(control_delta, 0.0,
                                                                   rtol=0, atol=1e-12))],
                  "prediction_pairs": [int(len(control_delta))]}).to_csv(
                      results_dir / "constant_control_check.csv", index=False)

    manifest = {
        "experiment_id": experiment_id,
        "benchmark_input_sha256": _hash(data_path),
        "provenance_registry_sha256": _hash(provenance_path),
        "frozen_baseline_report_sha256": baseline_report_hash,
        "eligible_identities": identities,
        "real_identities": [name for name in identities if not name.startswith("synthetic_")],
        "synthetic_identities": [name for name in identities if name.startswith("synthetic_")],
        "algorithms": list(ALGORITHM_NAMES),
        "records_per_seed": int(len(eligible)),
        "model_random_states": list(MODEL_SEEDS),
        "splits": "same leave-one-dataset-identity-out folds for every feature set and seed",
        "random_baseline": "same precomputed 1000-draw baseline per identity for every run",
        "constant_control": "D is B plus two zero-valued numeric columns; C/D predictions asserted identical",
        "bootstrap_unit": "dataset identity after averaging over model seeds",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "baseline_seed_42_reproduced": True,
        "model_artifacts_written": False,
        "production_model_unchanged": True,
        "full_dataset_profiles": True,
        "notes": "Ten identities include six synthetic datasets from one generator family; intervals are descriptive.",
        "feature_sets": feature_sets,
    }
    (results_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.pivot(index="metric", columns="experiment",
                        values="mean_across_identities").to_string())
    print("\nReal-only summary (four identity clusters):")
    print(real_summary.pivot(index="metric", columns="experiment",
                             values="mean_across_real_identities").to_string())
    print(f"\nC/D maximum prediction difference: {control_delta.max():.12g}")
    print(f"Versioned outputs: {results_dir}")
    return summary, paired, real_summary


if __name__ == "__main__":
    run_repeated_feature_experiment()
