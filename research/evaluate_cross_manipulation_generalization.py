
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
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


ROOT = Path(__file__).resolve().parents[1]

METADATA_ROOT = ROOT / "datasets" / "metadata"
FEATURE_ROOT = METADATA_ROOT / "cross_manipulation_features"

B0_TRAIN_FILE = METADATA_ROOT / "efficientnet_temporal_features.npz"
DINO_CLASSIFIER_FILE = ROOT / "models" / "dinov2_logistic_regression.joblib"

OUTPUT_ROOT = ROOT / "experiments" / "cross_manipulation_generalization"

CATEGORIES = ["face2face", "faceswap", "neuraltextures"]

B0_WEIGHT = 0.35
DINO_WEIGHT = 0.65
THRESHOLD = 0.5


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def get_array(data, possible_names):
    for name in possible_names:
        if name in data.files:
            return data[name]

    raise KeyError(
        f"Could not find any of these arrays: {possible_names}. "
        f"Available arrays: {data.files}"
    )


def calculate_metrics(labels, probabilities):

    predictions = (probabilities >= THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities)
    }

    return metrics, predictions


print("LOADING ORIGINAL B0 TRAINING FEATURES")

train_data = np.load(B0_TRAIN_FILE, allow_pickle=True)

train_sequences = get_array(
    train_data,
    ["sequences", "features"]
)

train_labels = get_array(
    train_data,
    ["labels", "label"]
)

train_splits = get_array(
    train_data,
    ["splits", "split"]
)

print("Original feature shape:", train_sequences.shape)

train_features = train_sequences.mean(axis=1)

train_mask = train_splits.astype(str) == "train"

X_train = train_features[train_mask]
y_train = train_labels[train_mask]

print("Training videos:", len(X_train))

print("\nRECONSTRUCTING ORIGINAL B0 CLASSIFIER")

b0_scaler = StandardScaler()
X_train_scaled = b0_scaler.fit_transform(X_train)

b0_classifier = LogisticRegression(
    C=0.1,
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

b0_classifier.fit(X_train_scaled, y_train)

print("B0 classifier ready.")


print("\nLOADING EXISTING DINOv2 CLASSIFIER")

dino_classifier = joblib.load(DINO_CLASSIFIER_FILE)

print("DINOv2 classifier ready.")


summary = []


for category in CATEGORIES:

    print("\n" + "=" * 60)
    print("EVALUATING:", category.upper())
    print("=" * 60)

    b0_file = FEATURE_ROOT / f"{category}_b0_features.npz"
    dino_file = FEATURE_ROOT / f"{category}_dinov2_features.npz"

    b0_data = np.load(b0_file, allow_pickle=True)
    dino_data = np.load(dino_file, allow_pickle=True)

    b0_sequences = get_array(
        b0_data,
        ["sequences", "features"]
    )

    dino_sequences = get_array(
        dino_data,
        ["features", "sequences"]
    )

    labels = get_array(
        b0_data,
        ["labels", "label"]
    ).astype(int)

    video_names = get_array(
        b0_data,
        ["video_names", "videos"]
    ).astype(str)

    dino_labels = get_array(
        dino_data,
        ["labels", "label"]
    ).astype(int)

    dino_video_names = get_array(
        dino_data,
        ["video_names", "videos"]
    ).astype(str)

    if not np.array_equal(video_names, dino_video_names):
        raise ValueError(
            f"{category}: B0 and DINOv2 video order does not match."
        )

    if not np.array_equal(labels, dino_labels):
        raise ValueError(
            f"{category}: B0 and DINOv2 labels do not match."
        )

    if b0_sequences.shape[2] != 1280:
        raise ValueError("Unexpected EfficientNet-B0 feature dimension.")

    if dino_sequences.shape[2] != 384:
        raise ValueError("Unexpected DINOv2 feature dimension.")

    X_b0 = b0_sequences.mean(axis=1)
    X_dino = dino_sequences.mean(axis=1)

    X_b0_scaled = b0_scaler.transform(X_b0)

    b0_probabilities = b0_classifier.predict_proba(
        X_b0_scaled
    )[:, 1]

    dino_probabilities = dino_classifier.predict_proba(
        X_dino
    )[:, 1]

    fusion_probabilities = (
        B0_WEIGHT * b0_probabilities
        + DINO_WEIGHT * dino_probabilities
    )

    model_results = {
        "EfficientNet-B0": b0_probabilities,
        "DINOv2": dino_probabilities,
        "B0_DINOv2_Fusion": fusion_probabilities
    }

    prediction_data = {
        "video_name": video_names,
        "actual_label": labels
    }

    for model_name, probabilities in model_results.items():

        metrics, predictions = calculate_metrics(
            labels,
            probabilities
        )

        prediction_data[f"{model_name}_probability"] = probabilities
        prediction_data[f"{model_name}_prediction"] = predictions

        print(f"\n{model_name}")
        print("-" * 40)

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(labels, predictions))

        print("\nClassification Report:")
        print(
            classification_report(
                labels,
                predictions,
                target_names=["Real", "Fake"],
                zero_division=0
            )
        )

        summary.append({
            "category": category,
            "model": model_name,
            "videos": len(labels),
            **metrics
        })

    prediction_file = (
        OUTPUT_ROOT /
        f"{category}_predictions.csv"
    )

    pd.DataFrame(prediction_data).to_csv(
        prediction_file,
        index=False
    )

    print("\nPredictions saved:", prediction_file)


summary_df = pd.DataFrame(summary)

summary_file = OUTPUT_ROOT / "cross_manipulation_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False
)

json_file = OUTPUT_ROOT / "cross_manipulation_summary.json"

with open(json_file, "w") as file:
    json.dump(
        summary,
        file,
        indent=4
    )


print("\n" + "=" * 60)
print("CROSS-MANIPULATION GENERALIZATION RESULTS")
print("=" * 60)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\nSummary saved:", summary_file)
print("JSON saved:", json_file)

print("\nCROSS-MANIPULATION EVALUATION COMPLETED")
