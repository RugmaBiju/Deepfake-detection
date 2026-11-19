
import csv
import json
from pathlib import Path

import mlflow

# Project directory
ROOT = Path(__file__).resolve().parents[1]

# Existing experiment result files
TEST_FILE = ROOT / "experiments" / "genimage_fusion_test_metrics_expanded.csv"
WEIGHTS_FILE = ROOT / "experiments" / "genimage_fusion_selected_weights_expanded.json"

# Use the existing SQLite tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TrueLens Experiments")

# Read existing test metrics
with open(TEST_FILE, "r", newline="") as file:
    rows = list(csv.DictReader(file))

fusion_row = next(
    (row for row in rows if row["model"] == "B0_DINOv2_Fusion"),
    None
)

if fusion_row is None:
    raise ValueError("Fusion results not found in the CSV file.")

# Read selected weights and validation metrics
with open(WEIGHTS_FILE, "r") as file:
    weights_data = json.load(file)

validation_metrics = weights_data["validation_metrics"]

# Start a historical experiment run
with mlflow.start_run(
    run_name="GenImage Expanded - B0-DINOv2 Fusion"
):

    # Log model configuration
    mlflow.log_params({
        "model": "EfficientNet-B0 + DINOv2",
        "fusion_method": "Weighted probability fusion",
        "b0_weight": weights_data["b0_weight"],
        "dinov2_weight": weights_data["dino_weight"],
        "weight_selection_metric": weights_data["selection_metric"],
    })

    # Log validation metrics
    mlflow.log_metrics({
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1": validation_metrics["f1"],
        "validation_roc_auc": validation_metrics["roc_auc"],
    })

    # Log test metrics
    mlflow.log_metrics({
        "test_accuracy": float(fusion_row["accuracy"]),
        "test_precision": float(fusion_row["precision"]),
        "test_recall": float(fusion_row["recall"]),
        "test_f1": float(fusion_row["f1"]),
        "test_roc_auc": float(fusion_row["roc_auc"]),
    })

    # Mark this as an import of a previously completed experiment
    mlflow.set_tags({
        "run_origin": "historical_import",
        "dataset_track": "GenImage Expanded",
        "evaluation_type": "validation_and_test",
    })

    # Preserve original result files as MLflow artifacts
    mlflow.log_artifact(str(TEST_FILE), artifact_path="source_results")
    mlflow.log_artifact(str(WEIGHTS_FILE), artifact_path="source_results")

    print("Fusion experiment successfully logged!")
    print("Run ID:", mlflow.active_run().info.run_id)
