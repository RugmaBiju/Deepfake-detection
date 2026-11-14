
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]

B0_FILE = ROOT / "experiments" / "test_evaluation" / "ffpp_test_video_predictions.csv"
DINOV2_FILE = ROOT / "experiments" / "dinov2_test_predictions.csv"

b0 = pd.read_csv(B0_FILE)
dinov2 = pd.read_csv(DINOV2_FILE)

print("B0 columns:", b0.columns.tolist())

comparison = b0.merge(
    dinov2,
    on="video_name",
    suffixes=("_b0", "_dinov2")
)

if len(comparison) != 32:
    raise ValueError(
        f"Expected 32 matching videos, found {len(comparison)}"
    )

if not (
    comparison["actual_label_b0"]
    == comparison["actual_label_dinov2"]
).all():
    raise ValueError("Actual labels do not match!")

comparison["b0_correct"] = (
    comparison["actual_label_b0"]
    == comparison["predicted_label_b0"]
)

comparison["dinov2_correct"] = (
    comparison["actual_label_dinov2"]
    == comparison["predicted_label_dinov2"]
)

comparison["error_pattern"] = comparison.apply(
    lambda row:
        "Both correct" if row["b0_correct"] and row["dinov2_correct"]
        else "Both wrong" if not row["b0_correct"] and not row["dinov2_correct"]
        else "B0 only correct" if row["b0_correct"]
        else "DINOv2 only correct",
    axis=1
)

print("\nMODEL ACCURACY")
print(
    "B0:",
    accuracy_score(
        comparison["actual_label_b0"],
        comparison["predicted_label_b0"]
    )
)

print(
    "DINOv2:",
    accuracy_score(
        comparison["actual_label_dinov2"],
        comparison["predicted_label_dinov2"]
    )
)

print("\nERROR OVERLAP")
print(comparison["error_pattern"].value_counts())

print("\nCOMPLEMENTARY PREDICTIONS")

print(
    comparison[
        comparison["error_pattern"].isin([
            "B0 only correct",
            "DINOv2 only correct"
        ])
    ][[
        "video_name",
        "actual_label_b0",
        "predicted_label_b0",
        "fake_probability_b0",
        "predicted_label_dinov2",
        "fake_probability_dinov2",
        "error_pattern"
    ]].to_string(index=False)
)

OUTPUT = ROOT / "experiments" / "b0_dinov2_error_comparison.csv"

comparison.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
