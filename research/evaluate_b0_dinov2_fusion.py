
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]

B0_FILE = ROOT / "experiments" / "b0_validation_predictions.csv"
DINO_FILE = ROOT / "experiments" / "dinov2_validation_predictions.csv"

b0 = pd.read_csv(B0_FILE)
dino = pd.read_csv(DINO_FILE)

data = b0.merge(
    dino,
    on="video_name",
    suffixes=("_b0", "_dino")
)

if len(data) != 28:
    raise ValueError(f"Expected 28 videos, found {len(data)}")

if not (
    data["actual_label_b0"] == data["actual_label_dino"]
).all():
    raise ValueError("Validation labels do not match!")

y = data["actual_label_b0"]

b0_prob = data["fake_probability_b0"]
dino_prob = data["fake_probability_dino"]

results = []

weights = np.arange(0.0, 1.01, 0.05)

print("VALIDATION FUSION EXPERIMENT\n")

for b0_weight in weights:

    dino_weight = 1 - b0_weight

    fused_prob = (
        b0_weight * b0_prob
        + dino_weight * dino_prob
    )

    predictions = (fused_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    auc = roc_auc_score(y, fused_prob)

    results.append({
        "b0_weight": b0_weight,
        "dino_weight": dino_weight,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })

    print(
        f"B0={b0_weight:.2f} | "
        f"DINOv2={dino_weight:.2f} | "
        f"Accuracy={accuracy:.4f} | "
        f"F1={f1:.4f} | "
        f"AUC={auc:.4f}"
    )

results_df = pd.DataFrame(results)

best = results_df.loc[results_df["f1"].idxmax()]

print("\nBEST VALIDATION FUSION")
print(best.to_string())

OUTPUT = ROOT / "experiments" / "b0_dinov2_fusion_validation.csv"

results_df.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
