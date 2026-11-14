
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
OUTPUT_FILE = ROOT / "experiments" / "dinov2_validation_predictions.csv"

data = np.load(FEATURE_FILE)

features = data["features"].mean(axis=1)
labels = data["labels"]
splits = data["splits"]

X_val = features[splits == "validation"]
y_val = labels[splits == "validation"]

model = joblib.load(MODEL_FILE)

predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

print("DINOV2 VALIDATION RESULTS")
print("Videos:", len(y_val))
print("Accuracy:", accuracy_score(y_val, predictions))
print("Precision:", precision_score(y_val, predictions, zero_division=0))
print("Recall:", recall_score(y_val, predictions, zero_division=0))
print("F1:", f1_score(y_val, predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_val, probabilities))

results = pd.DataFrame({
    "video_name": data["video_names"][splits == "validation"],
    "actual_label": y_val,
    "predicted_label": predictions,
    "fake_probability": probabilities
})

results.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
