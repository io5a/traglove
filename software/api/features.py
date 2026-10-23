import numpy as np

def extract_features(window):
    feats = []

    for i in range(window.shape[1]):
        col = window[:, i]

        feats.extend([
            np.mean(col),
            np.std(col),
            np.min(col),
            np.max(col)
        ])

    return np.array(feats)