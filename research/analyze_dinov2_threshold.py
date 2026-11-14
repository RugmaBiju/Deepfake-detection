
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "datasets" / "metadata" / "dinov2_temporal_features.npz"
MODEL_FILE = ROOT / "models" / "dinov2_logistic_regression.joblib"

data = np.load(FEATURE_FILE)

features = data["features"].mean(axis=1)
labels = data["labels"]
splits = data["splits"]

model = joblib.load(MODEL_FILE)

X_val = features[splits == "validation"]
y_val = labels[splits == "validation"]

X_test = features[splits == "test"]
y_test = labels[splits == "test"]

val_probabilities = model.predict_proba(X_val)[:, 1]
test_probabilities = model.predict_proba(X_test)[:, 1]


print("VALIDATION THRESHOLD ANALYSIS")

best_threshold = 0.5
best_f1 = -1

for threshold in np.arange(0.20, 0.81, 0.05):

    predictions = (val_probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(
        f"Threshold={threshold:.2f} | "
        f"Accuracy={accuracy:.4f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\nSELECTED VALIDATION THRESHOLD:", best_threshold)
print("Best validation F1:", best_f1)


test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nTEST RESULTS AT VALIDATION-SELECTED THRESHOLD")

print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Precision:", precision_score(y_test, test_predictions, zero_division=0))
print("Recall:", recall_score(y_test, test_predictions, zero_division=0))
print("F1:", f1_score(y_test, test_predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, test_probabilities))
