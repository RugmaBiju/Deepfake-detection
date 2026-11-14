
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "datasets" / "metadata" / "efficientnet_temporal_features.npz"
OUTPUT_FILE = ROOT / "experiments" / "b0_validation_predictions.csv"

data = np.load(FEATURE_FILE)

sequences = data["sequences"]
labels = data["labels"]
splits = data["splits"]

video_features = np.mean(sequences, axis=1)

train_mask = splits == "train"
val_mask = splits == "validation"

X_train = video_features[train_mask]
X_val = video_features[val_mask]

y_train = labels[train_mask]
y_val = labels[val_mask]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = LogisticRegression(
    C=0.1,
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

print("B0 VALIDATION RESULTS")
print("Videos:", len(y_val))
print("Accuracy:", accuracy_score(y_val, predictions))
print("Precision:", precision_score(y_val, predictions, zero_division=0))
print("Recall:", recall_score(y_val, predictions, zero_division=0))
print("F1:", f1_score(y_val, predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_val, probabilities))

results = pd.DataFrame({
    "video_name": data["video_names"][val_mask],
    "actual_label": y_val,
    "predicted_label": predictions,
    "fake_probability": probabilities
})

results.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
