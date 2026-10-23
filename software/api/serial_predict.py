"""Read glove values from serial and print prediction + confidence."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
N_FEATURES = 8


def load_pipeline(root: Path):
    pipe_path = root.parents[1] / "machine-learning" / "models" / "knn_pipeline_best.pkl"
    if pipe_path.exists():
        with open(pipe_path, "rb") as f:
            return pickle.load(f)

    print("Missing knn_pipeline_best.pkl. Run: python train_knn_random_search.py", file=sys.stderr)
    sys.exit(1)


def predict_line(pipe, raw: np.ndarray) -> tuple[str, float]:
    X = raw.reshape(1, -1)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        classes = pipe.named_steps["knn"].classes_
        idx = int(np.argmax(proba))
        return str(classes[idx]), float(proba[idx])
    pred = pipe.predict(X)
    return str(pred[0]), 1.0


def main() -> None:
    p = argparse.ArgumentParser(description="Serial glove prediction")
    p.add_argument("--port", default="COM6", help="Serial port (e.g. COM6, /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=921600)
    args = p.parse_args()

    try:
        import serial
    except ImportError:
        print("Install pyserial: pip install pyserial", file=sys.stderr)
        sys.exit(1)

    pipe = load_pipeline(ROOT)
    ser = serial.Serial(args.port, args.baud, timeout=0.1)

    try:
        while True:
            line = ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != N_FEATURES:
                continue
            try:
                raw = np.array([float(x) for x in parts], dtype=float)
            except ValueError:
                continue

            label, conf = predict_line(pipe, raw)
            print(f"prediction: {label} | confidence: {conf:.4f}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
