
import csv
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "TrueLens Experiments"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment = mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()
existing_runs = client.search_runs(
    [experiment.experiment_id],
    max_results=5000
)

existing_keys = {
    run.data.tags.get("historical_import_key")
    for run in existing_runs
}

imported = 0
skipped = 0


def log_historical_run(key, name, params, metrics, tags, artifacts):
    global imported, skipped

    if key in existing_keys:
        print("Already imported, skipping:", name)
        skipped += 1
        return

    with mlflow.start_run(run_name=name):
        clean_params = {
            str(k): str(v)
            for k, v in params.items()
            if v is not None
        }

        clean_metrics = {
            str(k): float(v)
            for k, v in metrics.items()
            if v is not None
        }

        mlflow.log_params(clean_params)
        mlflow.log_metrics(clean_metrics)

        mlflow.set_tags({
            **tags,
            "run_origin": "historical_import",
            "historical_import_key": key
        })

        for artifact in artifacts:
            path = ROOT / artifact

            if path.exists():
                mlflow.log_artifact(
                    str(path),
                    artifact_path="source_results"
                )

        print("Imported:", name)

    existing_keys.add(key)
    imported += 1


# --------------------------------------------------
# 1. B0, DINOv2 and Fusion - LOMO threshold results
# --------------------------------------------------

threshold_file = (
    ROOT / "experiments" / "lomo_threshold_analysis"
    / "threshold_comparison_summary.csv"
)

summary_files = {
    "B0": "experiments/lomo_generalization/lomo_b0_summary.csv",
    "DINOv2": "experiments/lomo_generalization_dinov2/lomo_dinov2_summary.csv",
    "Fusion": "experiments/lomo_generalization_fusion/lomo_fusion_summary.csv"
}

unseen_methods = {
    "A": "NeuralTextures",
    "B": "FaceSwap",
    "C": "Face2Face"
}

with open(threshold_file, "r", newline="") as file:
    rows = list(csv.DictReader(file))

for row in rows:
    model = row["model"]
    protocol = row["experiment"]
    unseen = row["unseen_manipulation"]

    metrics = {}

    for stage in ["default", "optimized"]:
        for metric in [
            "accuracy", "precision", "recall", "f1", "roc_auc"
        ]:
            metrics[f"{stage}_{metric}"] = float(
                row[f"{stage}_{metric}"]
            )

    params = {
        "model": model,
        "evaluation_protocol": "LOMO",
        "experiment": protocol,
        "unseen_manipulation": unseen,
        "selected_C": row["selected_C"],
        "selected_threshold": row["selected_threshold"]
    }

    log_historical_run(
        key=f"ffpp_lomo_threshold_{model.lower()}_{protocol}",
        name=f"FFPP LOMO - {model} - {protocol} - {unseen}",
        params=params,
        metrics=metrics,
        tags={
            "dataset_track": "FaceForensics++",
            "experiment_category": "cross_manipulation_generalization",
            "evaluation_protocol": "leave_one_manipulation_out"
        },
        artifacts=[
            str(threshold_file.relative_to(ROOT)),
            summary_files[model]
        ]
    )


# --------------------------------------------------
# 2. ConvNeXt-Tiny - LOMO validation and test results
# --------------------------------------------------

convnext_validation_file = (
    ROOT / "experiments" / "convnext_lomo_validation_metrics.json"
)

convnext_test_file = (
    ROOT / "experiments" / "convnext_lomo_test_metrics.json"
)

with open(convnext_validation_file, "r") as file:
    convnext_validation = json.load(file)

with open(convnext_test_file, "r") as file:
    convnext_test = json.load(file)

for protocol in ["A", "B", "C"]:
    validation = convnext_validation[protocol]
    test = convnext_test[protocol]
    unseen = unseen_methods[protocol]

    metrics = {}

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        metrics[f"validation_{metric}"] = float(validation[metric])
        metrics[f"test_{metric}"] = float(test[f"test_{metric}"])

    params = {
        "model": "ConvNeXt-Tiny",
        "classifier": "Logistic Regression",
        "evaluation_protocol": "LOMO",
        "experiment": protocol,
        "unseen_manipulation": unseen,
        "best_C": test["best_C"]
    }

    log_historical_run(
        key=f"ffpp_convnext_lomo_{protocol}",
        name=f"FFPP LOMO - ConvNeXt - {protocol} - {unseen}",
        params=params,
        metrics=metrics,
        tags={
            "dataset_track": "FaceForensics++",
            "experiment_category": "cross_manipulation_generalization",
            "evaluation_protocol": "leave_one_manipulation_out"
        },
        artifacts=[
            "experiments/convnext_lomo_validation_metrics.json",
            "experiments/convnext_lomo_test_metrics.json"
        ]
    )


# --------------------------------------------------
# 3. Swin-Tiny - validation-only LOMO results
# --------------------------------------------------

swin_file = ROOT / "experiments" / "swin_lomo_validation_metrics.json"

with open(swin_file, "r") as file:
    swin_results = json.load(file)

for protocol in ["A", "B", "C"]:
    result = swin_results[protocol]
    unseen = unseen_methods[protocol]

    metrics = {
        f"validation_{metric}": float(result[metric])
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    params = {
        "model": "Swin-Tiny",
        "classifier": "Logistic Regression",
        "evaluation_protocol": "LOMO",
        "experiment": protocol,
        "unseen_manipulation": unseen,
        "best_C": result["best_C"]
    }

    log_historical_run(
        key=f"ffpp_swin_lomo_validation_{protocol}",
        name=f"FFPP LOMO - Swin-Tiny - {protocol} - {unseen}",
        params=params,
        metrics=metrics,
        tags={
            "dataset_track": "FaceForensics++",
            "experiment_category": "cross_manipulation_generalization",
            "evaluation_protocol": "leave_one_manipulation_out",
            "metric_scope": "validation_only"
        },
        artifacts=[
            "experiments/swin_lomo_validation_metrics.json"
        ]
    )


print("\n========== IMPORT SUMMARY ==========")
print("New runs imported:", imported)
print("Existing runs skipped:", skipped)
print("LOMO historical import completed!")
