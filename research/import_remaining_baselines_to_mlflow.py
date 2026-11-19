
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


def read_json(relative_path):
    with open(ROOT / relative_path, "r") as file:
        return json.load(file)


def read_csv(relative_path):
    with open(ROOT / relative_path, "r", newline="") as file:
        return list(csv.DictReader(file))


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

            if path.exists() and path.is_file():
                mlflow.log_artifact(
                    str(path),
                    artifact_path="source_results"
                )

        print("Imported:", name)

    existing_keys.add(key)
    imported += 1


# --------------------------------------------------
# 1. EFFICIENTNET-B4
# --------------------------------------------------

b4_file = "experiments/efficientnet_b4_metrics.json"
b4_test_file = "experiments/efficientnet_b4_test_metrics.json"

b4 = read_json(b4_file)
b4_test = read_json(b4_test_file)

b4_metrics = {
    "best_validation_video_f1": b4["best_validation_video_f1"]
}

for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    b4_metrics["test_" + metric] = b4_test[metric]

b4_params = {
    "model": b4["model"],
    "image_size": b4["image_size"],
    "epochs_completed": b4["epochs_completed"],
    "batch_size": b4["batch_size"],
    "backbone_learning_rate": b4["backbone_learning_rate"],
    "classifier_learning_rate": b4["classifier_learning_rate"],
    "best_model_file": Path(b4["best_model"]).name
}

log_historical_run(
    key="efficientnet_b4_recorded_experiment",
    name="EfficientNet-B4 - Recorded Experiment",
    params=b4_params,
    metrics=b4_metrics,
    tags={
        "experiment_category": "deep_learning_model",
        "metric_scope": "validation_and_test_as_recorded"
    },
    artifacts=[
        b4_file,
        b4_test_file,
        "experiments/efficientnet_b4_history.csv"
    ]
)


# --------------------------------------------------
# 2. XCEPTION EMBEDDING CLASSIFIERS
# --------------------------------------------------

xception_file = "experiments/xception_embedding_metrics.json"
xception = read_json(xception_file)

for model, result in xception["results"].items():

    metrics = {
        "recorded_" + metric: result[metric]
        for metric in [
            "accuracy", "precision", "recall", "f1", "roc_auc"
        ]
    }

    params = {
        "feature_extractor": "Xception",
        "classifier": model
    }

    log_historical_run(
        key=f"xception_embedding_{model.lower().replace(' ', '_')}",
        name=f"Xception Embeddings - {model}",
        params=params,
        metrics=metrics,
        tags={
            "experiment_category": "embedding_classification",
            "metric_scope": "as_recorded_split_not_specified"
        },
        artifacts=[xception_file]
    )


# --------------------------------------------------
# 3. FORENSIC BASELINE CLASSIFIERS
# --------------------------------------------------

forensic_file = "experiments/forensic_baseline_metrics.json"
forensic = read_json(forensic_file)

for model, result in forensic["results"].items():

    metrics = {
        "recorded_" + metric: result[metric]
        for metric in [
            "accuracy", "precision", "recall", "f1", "roc_auc"
        ]
    }

    if model == forensic["best_model"]:
        metrics["best_validation_f1"] = forensic["best_validation_f1"]

    params = {
        "model": model,
        "input_features": json.dumps(forensic["input_features"])
    }

    log_historical_run(
        key=f"forensic_baseline_{model.lower().replace(' ', '_')}",
        name=f"Forensic Baseline - {model}",
        params=params,
        metrics=metrics,
        tags={
            "experiment_category": "forensic_baseline",
            "metric_scope": "as_recorded_split_not_specified"
        },
        artifacts=[forensic_file]
    )


# --------------------------------------------------
# 4. SPATIAL, FORENSIC AND HYBRID VALIDATION
# --------------------------------------------------

hybrid_file = "experiments/hybrid_validation_metrics.json"
hybrid_data = read_json(hybrid_file)

for model in ["spatial", "forensic", "hybrid"]:

    result = hybrid_data[model]

    metrics = {
        "validation_" + metric: result[metric]
        for metric in [
            "accuracy", "precision", "recall", "f1", "roc_auc"
        ]
    }

    params = {
        "model_branch": model,
        "fusion_method": (
            hybrid_data["fusion_method"] if model == "hybrid" else "none"
        )
    }

    if model == "hybrid":
        params["spatial_weight"] = hybrid_data["spatial_weight"]
        params["forensic_weight"] = hybrid_data["forensic_weight"]

    log_historical_run(
        key=f"hybrid_validation_{model}",
        name=f"Hybrid Architecture Validation - {model.title()}",
        params=params,
        metrics=metrics,
        tags={
            "experiment_category": "spatial_forensic_hybrid",
            "evaluation_type": "validation"
        },
        artifacts=[hybrid_file]
    )


# --------------------------------------------------
# 5. HYBRID FEATURE COMPARISON
# --------------------------------------------------

comparison_files = [
    (
        "experiments/hybrid_comparison_metrics.csv",
        "hybrid_feature_comparison"
    ),
    (
        "experiments/face_hybrid_comparison_metrics.csv",
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
                "accuracy", "precision", "recall", "f1", "roc_auc"
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
                "evaluation_type": "as_reported"
            },
            artifacts=[relative_file]
        )


# --------------------------------------------------
# 6. FREQUENCY AND SPATIAL TRAINING RECORDS
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

    result = read_json(config["metrics_file"])

    metrics = {
        "best_validation_video_f1": result["best_validation_video_f1"]
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


# --------------------------------------------------
# IMPORT SUMMARY
# --------------------------------------------------

print("\n========== BASELINE IMPORT SUMMARY ==========")
print("New runs imported:", imported)
print("Existing runs skipped:", skipped)
print("Baseline and hybrid import completed!")
