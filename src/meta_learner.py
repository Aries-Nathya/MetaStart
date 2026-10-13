"""Train and persist the score-prediction meta-model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.benchmark import ALGORITHM_NAMES, ROOT
from src.dataset_profiler import META_FEATURES


def build_meta_model(random_state: int = 42):
    encoder = ColumnTransformer(
        [("algorithm", OneHotEncoder(handle_unknown="ignore"), ["algorithm"])],
        remainder="passthrough",
    )
    regressor = RandomForestRegressor(n_estimators=300, max_depth=8,
                                      min_samples_leaf=2, random_state=random_state)
    return make_pipeline(encoder, regressor)


def train_meta_model(data_path: str | Path = ROOT / "data/meta_dataset.csv",
                     artifact_path: str | Path = ROOT / "models/meta_model.joblib"):
    data = pd.read_csv(data_path)
    if data.empty or data["dataset"].nunique() < 4:
        raise ValueError("Benchmark at least four distinct datasets before training.")
    required = set(META_FEATURES + ["dataset", "algorithm", "macro_f1"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Meta-dataset is missing columns: {sorted(missing)}")
    data = data.dropna(subset=META_FEATURES + ["algorithm", "macro_f1"])
    X = data[META_FEATURES + ["algorithm"]]
    model = build_meta_model()
    model.fit(X, data["macro_f1"])
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "algorithms": sorted(data["algorithm"].unique().tolist()),
                 "meta_features": META_FEATURES}, artifact_path)
    print(f"Saved meta-model to {artifact_path}")
    return model


if __name__ == "__main__":
    train_meta_model()
