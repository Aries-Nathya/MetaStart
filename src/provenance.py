"""Dataset provenance registry and fail-closed training/evaluation filtering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset_catalog import EXPECTED_ALGORITHMS, PROVENANCE_CATALOG


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_COLUMNS = ["dataset", "provenance_category", "provenance_status",
                      "benchmark_status", "provenance_detail"]
TRAINABLE_PROVENANCE_STATUSES = {"verified_local", "synthetic_reproducible"}


def write_dataset_provenance(data: pd.DataFrame, datasets, output_dir: str | Path = ROOT,
                             run_results: pd.DataFrame | None = None):
    """Write one provenance/status row per identity without changing benchmark schema."""
    output_dir = Path(output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    rows = []
    identities = set(data["dataset"].dropna()) | {name for name, _X, _y in datasets}
    for name in sorted(identities):
        group = data[data["dataset"] == name]
        category, provenance_status, detail = PROVENANCE_CATALOG.get(
            name, ("unknown", "unverified", "Identity is not in the approved dataset catalog"))
        source_rows = group
        if run_results is not None and name in set(run_results["dataset"]):
            source_rows = run_results[run_results["dataset"] == name]
        algorithms = source_rows["algorithm"].tolist()
        benchmark_status = "complete" if (
            len(algorithms) == len(EXPECTED_ALGORITHMS)
            and set(algorithms) == set(EXPECTED_ALGORITHMS)
        ) else "incomplete"
        rows.append({"dataset": name, "provenance_category": category,
                     "provenance_status": provenance_status,
                     "benchmark_status": benchmark_status,
                     "provenance_detail": detail})
    registry = pd.DataFrame(rows, columns=PROVENANCE_COLUMNS)
    registry.to_csv(output_dir / "results/dataset_provenance.csv", index=False)
    return registry


def filter_trainable_records(data: pd.DataFrame,
                             provenance_path: str | Path = ROOT / "results/dataset_provenance.csv"):
    """Keep only known-provenance identities with a complete five-model benchmark.

    Missing registry entries and duplicate registry identities raise errors instead
    of being treated as implicitly verified.
    """
    provenance_path = Path(provenance_path)
    if not provenance_path.exists():
        raise FileNotFoundError(f"Dataset provenance registry not found: {provenance_path}")
    registry = pd.read_csv(provenance_path)
    missing_columns = set(PROVENANCE_COLUMNS) - set(registry.columns)
    if missing_columns:
        raise ValueError(f"Dataset provenance registry is missing columns: {sorted(missing_columns)}")
    if registry["dataset"].isna().any() or registry["dataset"].duplicated().any():
        raise ValueError("Dataset provenance registry contains duplicate identities")
    required_data_columns = {"dataset", "algorithm"}
    missing_data_columns = required_data_columns - set(data.columns)
    if missing_data_columns:
        raise ValueError(f"Benchmark data is missing columns: {sorted(missing_data_columns)}")
    if data["dataset"].isna().any() or data["algorithm"].isna().any():
        raise ValueError("Benchmark data contains missing dataset identities or algorithms")
    data_identities = set(data["dataset"])
    registry_identities = set(registry["dataset"])
    unknown = data_identities - registry_identities
    if unknown:
        raise ValueError(f"Dataset identities lack provenance status: {sorted(unknown)}")
    unrepresented = registry_identities - data_identities
    if unrepresented:
        raise ValueError(f"Provenance registry contains identities absent from benchmark data: {sorted(unrepresented)}")

    registry_by_name = registry.set_index("dataset")
    eligible = set()
    for name in sorted(data_identities):
        expected = PROVENANCE_CATALOG.get(name)
        if expected is None:
            raise ValueError(f"Dataset identity is not in the approved catalog: {name}")
        row = registry_by_name.loc[name]
        category, status, _detail = expected
        if row["provenance_category"] != category or row["provenance_status"] != status:
            raise ValueError(
                f"Contradictory or unapproved provenance for {name}: "
                f"{row['provenance_category']}/{row['provenance_status']}"
            )
        if row["benchmark_status"] not in {"complete", "incomplete"}:
            raise ValueError(f"Invalid benchmark status for {name}: {row['benchmark_status']}")

        algorithms = data.loc[data["dataset"] == name, "algorithm"]
        duplicates = algorithms[algorithms.duplicated()].tolist()
        unexpected = set(algorithms) - set(EXPECTED_ALGORITHMS)
        if duplicates:
            raise ValueError(f"Duplicate algorithm records for {name}: {sorted(set(duplicates))}")
        if unexpected:
            raise ValueError(f"Unexpected algorithms for {name}: {sorted(unexpected)}")
        actual_complete = (len(algorithms) == len(EXPECTED_ALGORITHMS)
                           and set(algorithms) == set(EXPECTED_ALGORITHMS))
        actual_status = "complete" if actual_complete else "incomplete"
        if row["benchmark_status"] != actual_status:
            raise ValueError(
                f"Stale benchmark_status for {name}: registry says "
                f"{row['benchmark_status']}, data is {actual_status}"
            )
        if status in TRAINABLE_PROVENANCE_STATUSES:
            if not actual_complete:
                raise ValueError(f"Eligible dataset {name} lacks exactly one record per expected algorithm")
            eligible.add(name)

    filtered = data[data["dataset"].isin(eligible)].copy()
    if filtered.empty:
        raise ValueError("No complete, provenance-approved datasets are available for training/evaluation")
    return filtered
