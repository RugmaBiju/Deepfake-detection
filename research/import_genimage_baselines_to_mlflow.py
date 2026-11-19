
import csv
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]

RESULT_FILE = (
    ROOT / "experiments" / "genimage_fusion_test_metrics_expanded.csv"
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TrueLens Experiments")

with open(RESULT_FILE, "r", newline="") as file:
    rows = list(csv.DictReader(file))

models_to_import = {
    "EfficientNet-B0": "GenImage Expanded - EfficientNet-B0 Baseline",
    "DINOv2": "GenImage Expanded - DINOv2 Baseline",
}

for row in rows:
    model_name = row["model"]

    if model_name not in models_to_import:
        continue

    with mlflow.start_run(run_name=models_to_import[model_name]):

        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset", "GenImage Expanded")
        mlflow.log_param("evaluation_type", "test")

        mlflow.log_metrics({
            "test_accuracy": float(row["accuracy"]),
            "test_precision": float(row["precision"]),
            "test_recall": float(row["recall"]),
            "test_f1": float(row["f1"]),
            "test_roc_auc": float(row["roc_auc"]),
        })

        mlflow.set_tags({
            "run_origin": "historical_import",
            "comparison_group": "GenImage Expanded Model Comparison",
        })

        mlflow.log_artifact(
            str(RESULT_FILE),
            artifact_path="source_results"
        )

        print("Successfully logged:", model_name)
        print("Run ID:", mlflow.active_run().info.run_id)

print("Baseline imports completed!")
