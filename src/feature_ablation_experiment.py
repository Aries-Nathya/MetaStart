"""Versioned identity-held-out ablations for B's six numeric extensions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import ALGORITHM_NAMES, ROOT
from src.dataset_profiler import META_FEATURES
from src.feature_experiment import (
    BOOTSTRAP_REPLICATES,
    FEATURE_SETS,
    NUMERIC_EXTENSION_FEATURES,
    RANDOM_TRIALS,
    _evaluate_feature_set,
    _profiles_for_eligible_identities,
)
from src.feature_repeat_experiment import MODEL_SEEDS
from src.provenance import filter_trainable_records


EXPERIMENT_ID = "feature_B_ablation_v1_20261014"
JOINT_ABLATION_ID = "feature_B_joint_ablation_v1_20261014"
METRICS = ("meta_regret", "meta_top1_hit", "meta_top3_hit", "prediction_mae",
           "ranking_spearman")
_B_FEATURES = list(FEATURE_SETS["B"]["features"])
_SKEW_KURTOSIS = {"median_abs_numeric_skewness",
                  "median_abs_numeric_excess_kurtosis"}
_CORRELATIONS = {"mean_abs_numeric_correlation",
                 "fraction_numeric_pairs_abs_corr_ge_0_8"}
JOINT_CONFIGURATIONS = {
    "B": _B_FEATURES,
    "without_skewness_and_kurtosis": [f for f in _B_FEATURES if f not in _SKEW_KURTOSIS],
    "without_both_correlation_features": [f for f in _B_FEATURES if f not in _CORRELATIONS],
    "without_all_four": [f for f in _B_FEATURES if f not in _SKEW_KURTOSIS | _CORRELATIONS],
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_means(reports: pd.DataFrame, identities: list[str]) -> pd.DataFrame:
    return (reports.groupby(["configuration", "dataset"], as_index=False)[list(METRICS)]
            .mean())


def run_feature_ablation(experiment_id: str = EXPERIMENT_ID,
                         feature_configurations: dict[str, list[str]] | None = None,
                         model_seeds: tuple[int, ...] = MODEL_SEEDS):
    freeze_path = ROOT / "results/experiment_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    data_path = ROOT / "data/meta_dataset.csv"
    provenance_path = ROOT / "results/dataset_provenance.csv"
    baseline_path = ROOT / "results/evaluation_report.csv"
    results_dir = ROOT / "results/feature_experiments" / experiment_id
    if results_dir.exists():
        raise FileExistsError("Experiment output directory already exists; choose a new experiment_id")
    if _hash(data_path) != freeze["input_sha256"]:
        raise ValueError("Benchmark input differs from the frozen baseline")
    if _hash(baseline_path) != freeze["output_sha256"]["results/evaluation_report.csv"]:
        raise ValueError("Frozen evaluation report differs from its recorded hash")

    eligible = filter_trainable_records(pd.read_csv(data_path), provenance_path)
    identities = sorted(freeze["eligible_dataset_identities"])
    if sorted(eligible.dataset.unique()) != identities:
        raise ValueError("Eligible dataset identities differ from frozen inputs")
    if sorted(eligible.algorithm.unique()) != freeze["algorithms"]:
        raise ValueError("Algorithm set differs from frozen inputs")
    if len(eligible) != len(identities) * len(ALGORITHM_NAMES):
        raise ValueError("Expected exactly one score row per identity and algorithm")

    profiles = _profiles_for_eligible_identities(identities)
    legacy = eligible.groupby("dataset")[META_FEATURES].first().loc[identities]
    if not np.allclose(profiles[META_FEATURES], legacy, rtol=0, atol=1e-12):
        raise ValueError("Recomputed legacy features differ from frozen benchmark profiles")
    data = eligible[["dataset", "algorithm", "macro_f1"]].merge(
        profiles.reset_index(names="dataset"), on="dataset", validate="many_to_one")

    feature_b = list(FEATURE_SETS["B"]["features"])
    configs = feature_configurations or {"B": feature_b}
    model_seeds = tuple(model_seeds)
    if not model_seeds or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("model_seeds must be a non-empty sequence of unique seeds")
    if feature_configurations is None:
        for feature in NUMERIC_EXTENSION_FEATURES:
            configs[f"without_{feature}"] = [name for name in feature_b if name != feature]
    if "B" not in configs or configs["B"] != feature_b:
        raise ValueError("Configurations must include B unchanged as the reference")
    for name, features in configs.items():
        if not features or any(feature not in feature_b for feature in features):
            raise ValueError(f"Configuration {name} contains invalid B features")
        if features != [feature for feature in feature_b if feature in set(features)]:
            raise ValueError(f"Configuration {name} must preserve B feature ordering")

    rng = np.random.default_rng(42)
    random_baselines = {}
    for identity in identities:
        values = (eligible[eligible.dataset == identity].set_index("algorithm")
                  .loc[ALGORITHM_NAMES, "macro_f1"].to_numpy(dtype=float))
        draws = rng.choice(values, size=RANDOM_TRIALS, replace=True)
        random_baselines[identity] = (float(draws.mean()), float(draws.std()))

    all_reports = []
    for seed in model_seeds:
        for config, features in configs.items():
            report, _ = _evaluate_feature_set(
                config, features, data, identities, random_baselines, random_state=seed)
            report["configuration"] = config
            report["seed"] = seed
            all_reports.append(report)
    reports = pd.concat(all_reports, ignore_index=True)

    # Average repeated model fits within each identity first. Dataset identities,
    # not algorithm rows or seeds, are the independent units in paired intervals.
    identity_means = _identity_means(reports, identities)
    indexed = {name: identity_means[identity_means.configuration == name]
               .set_index("dataset").loc[identities] for name in configs}
    bootstrap_rng = np.random.default_rng(404)
    indices = bootstrap_rng.integers(0, len(identities),
                                     size=(BOOTSTRAP_REPLICATES, len(identities)))
    paired_rows = []
    for config in configs:
        if config == "B":
            continue
        for metric in METRICS:
            delta = (indexed[config][metric] - indexed["B"][metric]).to_numpy(float)
            boot_means = delta[indices].mean(axis=1)
            paired_rows.append({
                "contrast": f"{config}-B",
                "metric": metric,
                "mean_difference_ablation_minus_B": float(delta.mean()),
                "bootstrap_95ci_low": float(np.quantile(boot_means, 0.025)),
                "bootstrap_95ci_high": float(np.quantile(boot_means, 0.975)),
                "n_independent_identities": len(identities),
                "model_seed_count": len(model_seeds),
            })

    real_ids = [identity for identity in identities if not identity.startswith("synthetic_")]
    real_rows = []
    for config in configs:
        for metric in ("meta_regret", "meta_top1_hit", "meta_top3_hit"):
            real_rows.append({
                "configuration": config,
                "metric": metric,
                "mean_across_real_identities": float(indexed[config].loc[real_ids, metric].mean()),
                "n_real_identities": len(real_ids),
            })

    seed_means = (reports.groupby(["configuration", "seed"], as_index=False)[list(METRICS)]
                  .mean())
    seed_summary = (seed_means.groupby("configuration")[list(METRICS)]
                    .agg(["mean", "std", "min", "max"]))

    results_dir.mkdir(parents=True)
    (results_dir / "feature_sets.json").write_text(json.dumps(configs, indent=2) + "\n")
    reports.to_csv(results_dir / "evaluation_by_seed_identity.csv", index=False)
    identity_means.to_csv(results_dir / "per_identity_seed_average.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(results_dir / "paired_differences.csv", index=False)
    pd.DataFrame(real_rows).to_csv(results_dir / "real_subset_summary.csv", index=False)
    seed_summary.to_csv(results_dir / "per_seed_stability.csv")
    manifest = {
        "experiment_id": experiment_id,
        "benchmark_sha256": _hash(data_path),
        "provenance_sha256": _hash(provenance_path),
        "eligible_identities": identities,
        "real_identities": real_ids,
        "synthetic_identities": [name for name in identities if name.startswith("synthetic_")],
        "algorithms": list(ALGORITHM_NAMES),
        "model_seeds": list(model_seeds),
        "split": "leave-one-dataset-identity-out, identical for all configurations and seeds",
        "paired_unit": "dataset identity after averaging its repeated seed outcomes",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "baseline_draws": RANDOM_TRIALS,
        "feature_configurations": configs,
        "production_model_modified": False,
        "frozen_evaluation_artifacts_modified": False,
        "full_dataset_profiles": True,
        "limitations": "Only ten identities, six synthetic from one generator family; real-only subset has four identities.",
    }
    (results_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Mean outcomes over identities and seeds:")
    print(identity_means.groupby("configuration")[list(METRICS)].mean().to_string())
    print("\nPaired identity-level differences (ablation minus B):")
    print(pd.DataFrame(paired_rows).to_string(index=False))
    print("\nReal-only means:")
    print(pd.DataFrame(real_rows).pivot(index="metric", columns="configuration",
                                        values="mean_across_real_identities").to_string())
    print(f"\nVersioned outputs: {results_dir}")


if __name__ == "__main__":
    run_feature_ablation(JOINT_ABLATION_ID, JOINT_CONFIGURATIONS)
