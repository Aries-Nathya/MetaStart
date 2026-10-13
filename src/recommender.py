"""Rank supported algorithms for a dataset profile."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.dataset_profiler import profile_dataset
from src.meta_learner import ROOT


def recommend_algorithms(X: pd.DataFrame, y: pd.Series,
                         artifact_path: str | Path = ROOT / "models/meta_model.joblib") -> pd.DataFrame:
    artifact = joblib.load(artifact_path)
    profile = profile_dataset(X, y)
    rows = [{**profile, "algorithm": name} for name in artifact["algorithms"]]
    candidates = pd.DataFrame(rows)
    candidates["predicted_macro_f1"] = artifact["model"].predict(candidates)
    return candidates.sort_values("predicted_macro_f1", ascending=False).reset_index(drop=True)
