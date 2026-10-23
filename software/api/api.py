"""Window-based gesture prediction API."""

from pathlib import Path
import pickle

from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel

from features import extract_features

MODEL_DIR = Path(__file__).resolve().parents[2] / "machine-learning" / "models"
with (MODEL_DIR / "svm_model_test_Cdata.pkl").open("rb") as model_file:
    model = pickle.load(model_file)
with (MODEL_DIR / "scaler_Cdata.pkl").open("rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = FastAPI(title="Manusa Gesture API")


class SensorData(BaseModel):
    """A time window containing eight sensor readings per sample."""

    values: list[list[float]]


def predict_window(window: np.ndarray) -> tuple[str, float | None]:
    """Extract window features and return a class label and confidence."""
    feature_vector = extract_features(window).reshape(1, -1)
    scaled_features = scaler.transform(feature_vector)
    prediction = model.predict(scaled_features)[0]

    if hasattr(model, "predict_proba") and getattr(model, "probability", True):
        confidence = float(np.max(model.predict_proba(scaled_features)[0]))
    else:
        confidence = None
    return str(prediction), confidence


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Manusa gesture API is running"}


@app.post("/predict")
def predict(data: SensorData) -> dict[str, str | float | None]:
    window = np.asarray(data.values, dtype=float)
    if window.ndim != 2 or window.shape[1] != 8:
        return {"error": "Input must be an Nx8 window of sensor readings"}
    if window.shape[0] < 5:
        return {"error": "Window too small; provide at least 5 samples"}

    try:
        label, confidence = predict_window(window)
    except Exception as exc:
        return {"error": str(exc)}
    return {"prediction": label, "confidence": confidence}
