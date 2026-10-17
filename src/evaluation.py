"""Leave-one-dataset-out evaluation against common selection baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Evaluation is a batch job; select a non-interactive backend so chart
# generation also works in headless/restricted terminals.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from src.benchmark import ROOT
from src.dataset_profiler import META_FEATURES
from src.meta_learner import build_meta_model
from src.provenance import filter_trainable_records


def evaluate_meta_learning(data_path: str | Path = ROOT / "data/meta_dataset.csv",
                           results_dir: str | Path = ROOT / "results",
                           random_trials: int = 1000):
    """Evaluate selection on entirely held-out identities.

    Benchmark meta-features were profiled from each complete dataset before
    cross-validation. This estimates identity-level algorithm ranking; it is
    not an unbiased held-out-row estimate and should not be described as one.
    """
    data = filter_trainable_records(pd.read_csv(data_path))
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if data["dataset"].nunique() < 4:
        raise ValueError("At least four datasets are needed for leave-one-dataset-out evaluation.")
    rows = []
    rng = np.random.default_rng(42)
    for heldout in sorted(data["dataset"].unique()):
        train = data[data.dataset != heldout].copy()
        test = data[data.dataset == heldout].copy()
        model = build_meta_model()
        model.fit(train[META_FEATURES + ["algorithm"]], train["macro_f1"])
        test["predicted_macro_f1"] = model.predict(test[META_FEATURES + ["algorithm"]])
        test = test.sort_values("predicted_macro_f1", ascending=False)
        best = float(test.macro_f1.max())
        chosen = test.iloc[0]
        historical_name = train.groupby("algorithm").macro_f1.mean().idxmax()
        historical_score = float(test.loc[test.algorithm == historical_name, "macro_f1"].iloc[0])
        fixed_name = "Random Forest"
        fixed_score = float(test.loc[test.algorithm == fixed_name, "macro_f1"].iloc[0])
        random_scores = rng.choice(test.macro_f1.to_numpy(), size=random_trials, replace=True)
        actual_order = test.sort_values("macro_f1", ascending=False).algorithm.tolist()
        predicted_order = test.algorithm.tolist()
        top3 = set(predicted_order[:3])
        records = {
            "dataset": heldout,
            "selected_algorithm": chosen.algorithm,
            "meta_f1": float(chosen.macro_f1),
            "meta_minus_historical_f1": float(chosen.macro_f1) - historical_score,
            "meta_minus_fixed_f1": float(chosen.macro_f1) - fixed_score,
            "meta_minus_random_f1": float(chosen.macro_f1) - float(random_scores.mean()),
            "best_f1": best,
            "meta_regret": best - float(chosen.macro_f1),
            "meta_top1_hit": int(chosen.macro_f1 == best),
            "meta_top3_hit": int(actual_order[0] in top3),
            "historical_algorithm": historical_name,
            "historical_f1": historical_score,
            "historical_regret": best - historical_score,
            "fixed_algorithm": fixed_name,
            "fixed_f1": fixed_score,
            "fixed_regret": best - fixed_score,
            "random_expected_f1": float(random_scores.mean()),
            "random_regret": best - float(random_scores.mean()),
            "random_f1_sd": float(random_scores.std()),
            "prediction_mae": float(mean_absolute_error(test.macro_f1, test.predicted_macro_f1)),
            "ranking_spearman": float(spearmanr(test.macro_f1, test.predicted_macro_f1).statistic),
        }
        rows.append(records)
    report = pd.DataFrame(rows)
    report.to_csv(results_dir / "evaluation_report.csv", index=False)
    summary_cols = ["meta_f1", "fixed_f1", "historical_f1", "random_expected_f1",
                    "best_f1", "meta_regret", "fixed_regret", "historical_regret",
                    "random_regret", "meta_top1_hit", "meta_top3_hit", "prediction_mae",
                    "ranking_spearman", "meta_minus_historical_f1", "meta_minus_fixed_f1",
                    "meta_minus_random_f1"]
    # Bootstrap over held-out dataset identities. These intervals are descriptive:
    # ten eligible identities include six synthetic datasets, so they are not
    # evidence from a large independent real-world test sample.
    rng = np.random.default_rng(2026)
    summary_records = []
    for column in summary_cols:
        values = report[column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        bootstrap_means = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        summary_records.append({
            "metric": column,
            "mean": float(values.mean()),
            "dataset_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "bootstrap_95ci_low": float(np.quantile(bootstrap_means, 0.025)),
            "bootstrap_95ci_high": float(np.quantile(bootstrap_means, 0.975)),
        })
    summary = pd.DataFrame(summary_records).set_index("metric")
    summary.to_csv(results_dir / "evaluation_summary.csv")
    ax = report[["meta_regret", "fixed_regret", "historical_regret", "random_regret"]].mean().plot(
        kind="bar", color=["#2878b5", "#999999", "#f0a202", "#66a182"])
    ax.set_ylabel("Mean regret (macro-F1 points)")
    ax.set_title("Leave-one-dataset-out selection regret")
    ax.set_xticklabels(["MetaStart", "Fixed RF", "Historical best", "Random"], rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "regret_comparison.png", dpi=160)
    plt.close()
    print(report.to_string(index=False))
    print("\nMean summary:\n", summary.to_string())
    return report


if __name__ == "__main__":
    evaluate_meta_learning()
