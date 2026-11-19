
import csv
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment = mlflow.set_experiment("TrueLens Experiments")

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

        mlflow.log_params({
            str(k): str(v)
            for k, v in params.items()
            if v is not None
        })

        mlflow.log_metrics({
            str(k): float(v)
            for k, v in metrics.items()
            if v is not None
        })

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


def read_csv(relative_path):
    with open(ROOT / relative_path, "r", newline="") as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# 1. HYBRID FEATURE COMPARISONS
# --------------------------------------------------

comparison_files = [
    (
        "experiments/hybrid/hybrid_comparison_metrics.csv",
        "hybrid_feature_comparison"
    ),
    (
        "experiments/face_hybrid/face_hybrid_comparison_metrics.csv",
        "face_hybrid_feature_comparison"
    )
]

for relative_file, group_name in comparison_files:

    rows = read_csv(relative_file)

    for row in rows:

        feature_set = row["feature_set"]
        model = row["model"]

        metrics = {
            "reported_" + metric: float(row[metric])
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc"
            ]
        }

        params = {
            "feature_set": feature_set,
            "classifier": model
        }

        key = f"{group_name}_{feature_set}_{model}".lower()

        log_historical_run(
            key=key,
            name=f"{group_name} - {feature_set} - {model}",
            params=params,
            metrics=metrics,
            tags={
                "experiment_category": "feature_fusion_comparison",
                "evaluation_type": "as_recorded"
            },
            artifacts=[relative_file]
        )


# --------------------------------------------------
# 2. FREQUENCY AND SPATIAL TRAINING HISTORY
# --------------------------------------------------

baseline_configs = [
    {
        "name": "Frequency Baseline",
        "key": "frequency_baseline_training",
        "metrics_file": "experiments/frequency_baseline_metrics.json",
        "history_file": "experiments/frequency_baseline_history.csv"
    },
    {
        "name": "Spatial Baseline",
        "key": "spatial_baseline_training",
        "metrics_file": "experiments/spatial_baseline_metrics.json",
        "history_file": "experiments/spatial_baseline_history.csv"
    }
]

for config in baseline_configs:

    with open(ROOT / config["metrics_file"], "r") as file:
        result = json.load(file)

    metrics = {
        "best_validation_video_f1":
            float(result["best_validation_video_f1"])
    }

    params = {
        "model_type": config["name"],
        "epochs": result["epochs"],
        "batch_size": result["batch_size"],
        "learning_rate": result["learning_rate"],
        "checkpoint_file": Path(result["best_model"]).name
    }

    log_historical_run(
        key=config["key"],
        name=config["name"] + " - Training Record",
        params=params,
        metrics=metrics,
        tags={
            "experiment_category": "training_history",
            "metric_scope": "validation_only"
        },
        artifacts=[
            config["metrics_file"],
            config["history_file"]
        ]
    )


print("\n========== IMPORT SUMMARY ==========")
print("New runs imported:", imported)
print("Existing runs skipped:", skipped)
print("Missing experiment imports completed!")
