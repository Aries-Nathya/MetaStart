"""Approved local dataset identities and reproducible synthetic specifications."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification


EXPECTED_ALGORITHMS = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "k-Nearest Neighbors",
)

SKLEARN_DATASETS = {
    "breast_cancer": "Bundled scikit-learn classification dataset",
    "digits": "Bundled scikit-learn classification dataset",
    "iris": "Bundled scikit-learn classification dataset",
    "wine": "Bundled scikit-learn classification dataset",
}

# These parameters are the generation contract. Keep benchmark generation and
# provenance validation tied to this single source of truth.
SYNTHETIC_DATASET_SPECS = {
    "synthetic_42": {
        "n_samples": 450, "n_features": 8, "n_classes": 2,
        "weights": None, "n_informative": 4, "n_redundant": 1,
        "n_clusters_per_class": 2, "flip_y": 0.01, "seed": 42,
    },
    "synthetic_43": {
        "n_samples": 700, "n_features": 18, "n_classes": 3,
        "weights": (0.65, 0.25, 0.10), "n_informative": 9, "n_redundant": 3,
        "n_clusters_per_class": 2, "flip_y": 0.01, "seed": 43,
    },
    "synthetic_44": {
        "n_samples": 900, "n_features": 30, "n_classes": 2,
        "weights": (0.85, 0.15), "n_informative": 15, "n_redundant": 6,
        "n_clusters_per_class": 2, "flip_y": 0.01, "seed": 44,
    },
    "synthetic_local_300x6_imbalanced": {
        "n_samples": 300, "n_features": 6, "n_classes": 2,
        "weights": (0.90, 0.10), "n_informative": 4, "n_redundant": 1,
        "n_clusters_per_class": 1, "flip_y": 0.0, "seed": 20261001,
    },
    "synthetic_local_1200x20_four_class": {
        "n_samples": 1200, "n_features": 20, "n_classes": 4,
        "weights": (0.55, 0.25, 0.15, 0.05), "n_informative": 8,
        "n_redundant": 4, "n_clusters_per_class": 1, "flip_y": 0.0,
        "seed": 20261002,
    },
    "synthetic_local_2500x60_three_class": {
        "n_samples": 2500, "n_features": 60, "n_classes": 3,
        "weights": (1 / 3, 1 / 3, 1 / 3), "n_informative": 12,
        "n_redundant": 10, "n_clusters_per_class": 1, "flip_y": 0.0,
        "seed": 20261003,
    },
}

UNVERIFIED_OPENML_DATASETS = {
    "openml_11_balance_scale": "OpenML identity; provenance not validated locally",
    "openml_54_vehicle": "OpenML identity; provenance not validated locally",
}

PROVENANCE_CATALOG = {
    **{name: ("sklearn_builtin", "verified_local", detail)
       for name, detail in SKLEARN_DATASETS.items()},
    **{name: ("synthetic", "synthetic_reproducible",
              f"Deterministic make_classification specification: {spec}")
       for name, spec in SYNTHETIC_DATASET_SPECS.items()},
    **{name: ("openml", "unverified", detail)
       for name, detail in UNVERIFIED_OPENML_DATASETS.items()},
}


def generate_synthetic_datasets():
    """Yield the six approved synthetic datasets from their catalog specs."""
    for name, spec in SYNTHETIC_DATASET_SPECS.items():
        params = {key: value for key, value in spec.items() if key != "seed"}
        if params["weights"] is not None:
            params["weights"] = list(params["weights"])
        X, y = make_classification(**params, random_state=spec["seed"])
        yield name, pd.DataFrame(X), pd.Series(y)
