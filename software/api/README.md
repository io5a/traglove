# Prediction API

The FastAPI service exposes `GET /` for a health message and `POST /predict` for window-based gesture classification.

Run it from the repository root:

```bash
python -m uvicorn api:app --app-dir software/api --reload
```

`POST /predict` expects JSON shaped like `{"values": [[... eight readings ...], ...]}`. Each row is ordered `X`, `Y`, `Z`, `F1`, `F2`, `F3`, `F4`, `F5`. A window must contain at least five rows. The response includes `prediction` and `confidence`; confidence is `null` if the model does not provide probability estimates.

The service loads the SVM and scaler from `machine-learning/models` using paths relative to this file, so the command works from any current directory.
