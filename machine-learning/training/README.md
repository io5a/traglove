# Training

`train_knn.py` retrains the KNN pipeline used by `software/api/serial_predict.py`. It reads `machine-learning/data/data_new_2.csv` and writes the pipeline artifact into `machine-learning/models`.

Run it from the repository root after installing `requirements.txt`:

```bash
python machine-learning/training/train_knn.py
```

The FastAPI service uses a separate, pre-trained window-based SVM model. The original folder also contained several exploratory experiments; those snapshots are under the Git-ignored `local-only/legacy-training` directory and are not part of the maintained training workflow. Review them before reuse.
