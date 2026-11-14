
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[1]

B0_FILE = ROOT / "experiments" / "test_evaluation" / "ffpp_test_video_predictions.csv"
DINO_FILE = ROOT / "experiments" / "dinov2_test_predictions.csv"

b0 = pd.read_csv(B0_FILE)
dino = pd.read_csv(DINO_FILE)

data = b0.merge(
    dino,
    on="video_name",
    suffixes=("_b0", "_dino")
)

if len(data) != 32:
    raise ValueError(f"Expected 32 test videos, found {len(data)}")

if not (
    data["actual_label_b0"] == data["actual_label_dino"]
).all():
    raise ValueError("Test labels do not match!")

y = data["actual_label_b0"]

b0_prob = data["fake_probability_b0"]
dino_prob = data["fake_probability_dino"]

b0_weight = 0.35
dino_weight = 0.65

fused_prob = (
    b0_weight * b0_prob
    + dino_weight * dino_prob
)

predictions = (fused_prob >= 0.5).astype(int)

print("B0 + DINOV2 TEST RESULTS")
print("B0 weight:", b0_weight)
print("DINOv2 weight:", dino_weight)

print("\nAccuracy:", accuracy_score(y, predictions))
print("Precision:", precision_score(y, predictions, zero_division=0))
print("Recall:", recall_score(y, predictions, zero_division=0))
print("F1:", f1_score(y, predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y, fused_prob))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y, predictions))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y,
        predictions,
        target_names=["Authentic", "Manipulated"],
        zero_division=0
    )
)

data["fused_probability"] = fused_prob
data["fused_prediction"] = predictions

OUTPUT = ROOT / "experiments" / "b0_dinov2_fusion_test_predictions.csv"

data.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
