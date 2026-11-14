
import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "datasets" / "metadata" / "dinov2_temporal_features.npz"

MODEL_FILE = ROOT / "models" / "dinov2_logistic_regression.joblib"
METRICS_FILE = ROOT / "experiments" / "dinov2_metrics.json"
PREDICTIONS_FILE = ROOT / "experiments" / "dinov2_test_predictions.csv"


data = np.load(FEATURE_FILE)

features = data["features"]
labels = data["labels"]
video_names = data["video_names"]
splits = data["splits"]

print("Feature shape:", features.shape)

video_features = features.mean(axis=1)

print("Video feature shape:", video_features.shape)


train_mask = splits == "train"
val_mask = splits == "validation"
test_mask = splits == "test"

X_train = video_features[train_mask]
y_train = labels[train_mask]

X_val = video_features[val_mask]
y_val = labels[val_mask]

X_test = video_features[test_mask]
y_test = labels[test_mask]

test_names = video_names[test_mask]

print("\nTraining videos:", len(X_train))
print("Validation videos:", len(X_val))
print("Test videos:", len(X_test))


best_model = None
best_c = None
best_val_f1 = -1

for c in [0.001, 0.01, 0.1, 1, 10, 100]:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c,
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_probabilities = model.predict_proba(X_val)[:, 1]

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_auc = roc_auc_score(y_val, val_probabilities)

    print(
        f"C={c} | "
        f"Validation F1={val_f1:.4f} | "
        f"Validation AUC={val_auc:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_c = c
        best_model = model


print("\nBest C:", best_c)
print("Best validation F1:", best_val_f1)


test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]


metrics = {
    "model": "DINOv2 + Logistic Regression",
    "best_C": best_c,
    "validation_f1": best_val_f1,
    "test_accuracy": accuracy_score(y_test, test_predictions),
    "test_precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "test_recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "test_f1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "test_roc_auc": roc_auc_score(y_test, test_probabilities)
}


print("\nDINOV2 TEST RESULTS")

for key, value in metrics.items():
    print(f"{key}: {value}")


print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, test_predictions))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["Authentic", "Manipulated"],
    zero_division=0
))


joblib.dump(best_model, MODEL_FILE)

pd.DataFrame({
    "video_name": test_names,
    "actual_label": y_test,
    "predicted_label": test_predictions,
    "fake_probability": test_probabilities
}).to_csv(PREDICTIONS_FILE, index=False)


with open(METRICS_FILE, "w") as file:
    json.dump(metrics, file, indent=4)


print("\nDINOV2 CLASSIFIER TRAINING COMPLETED")
print("Model saved:", MODEL_FILE)
print("Metrics saved:", METRICS_FILE)
print("Predictions saved:", PREDICTIONS_FILE)
