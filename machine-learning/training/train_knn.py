"""Train the KNN pipeline used by the live serial prediction client."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "data_new_2.csv"
MODEL_DIR = ROOT.parent / "models"
PIPELINE_PATH = MODEL_DIR / "knn_pipeline_best.pkl"
SCALER_PATH = MODEL_DIR / "knn_scaler_best.pkl"

FEATURE_COLUMNS = ["X", "Y", "Z", "F1", "F2", "F3", "F4", "F5"]
TARGET_COLUMN = "LABEL"


def load_training_data():
    frame = pd.read_csv(DATA_PATH, sep=r"\s+")
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = frame[TARGET_COLUMN].to_numpy()
    return features, labels


def main() -> None:
    features, labels = load_training_data()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "knn__n_neighbors": list(range(1, 51)),
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan", "minkowski", "chebyshev"],
        },
        n_iter=40,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(features, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_pipeline = search.best_estimator_
    with PIPELINE_PATH.open("wb") as model_file:
        pickle.dump(best_pipeline, model_file)
    with SCALER_PATH.open("wb") as scaler_file:
        pickle.dump(best_pipeline.named_steps["scale"], scaler_file)

    print(f"Best cross-validation accuracy: {search.best_score_:.4f}")
    print(f"Saved pipeline: {PIPELINE_PATH}")


if __name__ == "__main__":
    main()
