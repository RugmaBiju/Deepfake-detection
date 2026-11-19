
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


def numeric_metrics(row, columns, prefix=""):
    return {
        prefix + column: float(row[column])
        for column in columns
        if row.get(column) not in (None, "")
    }


# --------------------------------------------------
# 1. FF++ FINAL VIDEO EVALUATION
# --------------------------------------------------

ffpp_file = "experiments/test_evaluation/ffpp_test_metrics.csv"

for row in read_csv(ffpp_file):

    evaluation_name = row["experiment"]

    metrics = numeric_metrics(
        row,
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
        prefix="test_"
    )

    params = {
        "evaluation_name": evaluation_name,
        "videos": row["videos"],
        "dataset": "FaceForensics++"
    }

    log_historical_run(
        key="ffpp_final_test_" + evaluation_name.lower().replace(" ", "_"),
        name="FFPP Final Test - " + evaluation_name,
        params=params,
        metrics=metrics,
        tags={
            "dataset_track": "FaceForensics++",
            "experiment_category": "video_detection",
            "evaluation_type": "test"
        },
        artifacts=[ffpp_file]
    )


# --------------------------------------------------
# 2. CROSS-MANIPULATION VIDEO COMPARISON
# --------------------------------------------------

cross_file = "experiments/cross_manipulation_generalization/cross_manipulation_summary.csv"

cross_rows = read_csv(cross_file)

for row in cross_rows:

    model = row["model"]
    category = row["category"]

    metrics = numeric_metrics(
        row,
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
        prefix="reported_"
    )

    params = {
        "model": model,
        "manipulation_category": category,
        "videos": row["videos"],
        "evaluation_track": "cross_manipulation"
    }

    key = f"cross_video_{model.lower()}_{category.lower()}"

    log_historical_run(
        key=key,
        name=f"Cross-Manipulation Video - {model} - {category}",
        params=params,
        metrics=metrics,
        tags={
            "dataset_track": "FaceForensics++",
            "experiment_category": "cross_manipulation_video",
            "metric_scope": "as_reported_in_summary"
        },
        artifacts=[
            cross_file,
            "experiments/cross_manipulation_generalization/diagnostics/confusion_matrix_summary.csv"
        ]
    )


# --------------------------------------------------
# 3. CONVNEXT VIDEO LOMO
# --------------------------------------------------

convnext_validation_file = (
    "experiments/convnext_lomo_validation_metrics.json"
)

convnext_video_file = (
    "experiments/convnext_lomo_video_test_metrics.json"
)

with open(ROOT / convnext_validation_file, "r") as file:
    convnext_validation = json.load(file)

with open(ROOT / convnext_video_file, "r") as file:
    convnext_video = json.load(file)

unseen_methods = {
    "A": "NeuralTextures",
    "B": "FaceSwap",
    "C": "Face2Face"
}

for protocol in ["A", "B", "C"]:

    validation = convnext_validation[protocol]
    video = convnext_video[protocol]

    metrics = {
        "video_validation_f1": float(video["video_validation_f1"]),
        "test_accuracy": float(video["test_accuracy"]),
        "test_precision": float(video["test_precision"]),
        "test_recall": float(video["test_recall"]),
        "test_f1": float(video["test_f1"]),
        "test_roc_auc": float(video["test_roc_auc"])
    }

    params = {
        "model": "ConvNeXt-Tiny",
        "classifier": "Logistic Regression",
        "experiment": protocol,
        "unseen_manipulation": unseen_methods[protocol],
        "best_C": video["best_C"],
        "test_videos": video["test_videos"]
    }

    log_historical_run(
        key=f"convnext_video_lomo_{protocol}",
        name=f"ConvNeXt Video LOMO - {protocol} - {unseen_methods[protocol]}",
        params=params,
        metrics=metrics,
        tags={
            "dataset_track": "FaceForensics++",
            "experiment_category": "video_detection",
            "evaluation_protocol": "LOMO",
            "evaluation_type": "video_level_test"
        },
        artifacts=[
            convnext_validation_file,
            convnext_video_file
        ]
    )


# --------------------------------------------------
# 4. ORIGINAL VISUAL BASELINES
# --------------------------------------------------

visual_file = "experiments/test_evaluation/original_visual_test_metrics.csv"

for row in read_csv(visual_file):

    model = row["model"]

    metrics = numeric_metrics(
        row,
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
        prefix="test_"
    )

    params = {
        "model": model,
        "evaluation_type": "original_visual_test"
    }

    log_historical_run(
        key=f"original_visual_test_{model.lower()}",
        name=f"Original Visual Test - {model}",
        params=params,
        metrics=metrics,
        tags={
            "experiment_category": "visual_baseline",
            "evaluation_type": "test",
            "metric_scope": "as_recorded"
        },
        artifacts=[visual_file]
    )


# --------------------------------------------------
# 5. TEMPORAL GRU
# --------------------------------------------------

temporal_file = "experiments/temporal_gru_metrics.json"

with open(ROOT / temporal_file, "r") as file:
    temporal = json.load(file)

temporal_metrics = {
    "recorded_accuracy": float(temporal["accuracy"]),
    "recorded_precision": float(temporal["precision"]),
    "recorded_recall": float(temporal["recall"]),
    "recorded_f1": float(temporal["f1"]),
    "recorded_roc_auc": float(temporal["roc_auc"]),
    "best_validation_loss": float(temporal["best_validation_loss"])
}

temporal_params = {
    "model": "Temporal GRU",
    "best_epoch": temporal["best_epoch"]
}

log_historical_run(
    key="temporal_gru_recorded_evaluation",
    name="Temporal GRU - Recorded Evaluation",
    params=temporal_params,
    metrics=temporal_metrics,
    tags={
        "experiment_category": "temporal_video_detection",
        "metric_scope": "evaluation_split_not_specified_in_source"
    },
    artifacts=[
        temporal_file,
        "experiments/temporal_model_comparison.csv"
    ]
)


# --------------------------------------------------
# IMPORT SUMMARY
# --------------------------------------------------

print("\n========== VIDEO AND TEMPORAL IMPORT SUMMARY ==========")
print("New runs imported:", imported)
print("Existing runs skipped:", skipped)
print("Video and temporal import completed!")
