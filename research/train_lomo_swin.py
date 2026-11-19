

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
import os
import json

data = np.load("datasets/metadata/swin_features/swin_trainval_features.npz")

features = data["features"]
labels = data["label"]
categories = data["category"]
splits = data["split"]

experiments = {
    "A": ["Face2Face", "FaceSwap"],
    "B": ["Face2Face", "NeuralTextures"],
    "C": ["FaceSwap", "NeuralTextures"]
}

results = {}

for experiment, seen_categories in experiments.items():

    print("\n" + "=" * 55)
    print("SWIN LOMO EXPERIMENT", experiment)
    print("Seen:", seen_categories)
    print("Unseen:", list(set(["Face2Face", "FaceSwap", "NeuralTextures"]) - set(seen_categories)))
    print("=" * 55)

    train_mask = (splits == "train") & (
        (labels == 0) | np.isin(categories, seen_categories)
    )

    val_mask = (splits == "validation") & (
        (labels == 0) | np.isin(categories, seen_categories)
    )

    X_train = features[train_mask]
    y_train = labels[train_mask]

    X_val = features[val_mask]
    y_val = labels[val_mask]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    best_model = None
    best_f1 = -1
    best_c = None

    for c in [0.001, 0.01, 0.1, 1, 10, 100]:

        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        f1 = f1_score(y_val, predictions, zero_division=0)

        print("C:", c, "| Validation F1:", round(f1, 4))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_c = c

    probabilities = best_model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "best_C": best_c,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "confusion_matrix": confusion_matrix(y_val, predictions).tolist()
    }

    results[experiment] = metrics

    print("\nBest C:", best_c)
    print("Validation metrics:", metrics)

os.makedirs("experiments", exist_ok=True)

with open("experiments/swin_lomo_validation_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSWIN LOMO TRAINING COMPLETE")
