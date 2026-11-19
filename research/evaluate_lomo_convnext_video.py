
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

trainval = np.load(
    "datasets/metadata/convnext_features/convnext_trainval_features_clean.npz"
)

X = trainval["features"]
y = trainval["label"]
categories = np.char.lower(trainval["category"].astype(str))
splits = np.char.lower(trainval["split"].astype(str))
video_names = trainval["video_name"].astype(str)

test = np.load(
    "datasets/metadata/convnext_features/convnext_test_features.npz"
)

X_test_all = test["features"]
y_test_all = test["labels"]
categories_test = np.char.lower(test["categories"].astype(str))
video_names_test = test["video_names"].astype(str)

experiments = {
    "A": {
        "seen": ["face2face", "faceswap"],
        "unseen": "neuraltextures"
    },
    "B": {
        "seen": ["face2face", "neuraltextures"],
        "unseen": "faceswap"
    },
    "C": {
        "seen": ["faceswap", "neuraltextures"],
        "unseen": "face2face"
    }
}

C_values = [0.001, 0.01, 0.1, 1, 10, 100]


def aggregate_videos(probabilities, labels, categories, video_names):
    df = pd.DataFrame({
        "probability": probabilities,
        "label": labels,
        "category": categories,
        "video_name": video_names
    })

    return df.groupby(
        ["category", "video_name"],
        as_index=False
    ).agg({
        "probability": "mean",
        "label": "first"
    })


results = {}

for name, config in experiments.items():

    print("\n" + "=" * 60)
    print("CONVNEXT VIDEO-LEVEL LOMO EXPERIMENT", name)
    print("Unseen:", config["unseen"])
    print("=" * 60)

    train_mask = (
        (splits == "train") &
        (
            (categories == "real") |
            np.isin(categories, config["seen"])
        )
    )

    val_mask = (
        (splits == "validation") &
        (
            (categories == "real") |
            np.isin(categories, config["seen"])
        )
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]
    val_categories = categories[val_mask]
    val_video_names = video_names[val_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_c = None
    best_f1 = -1

    for c in C_values:

        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        val_probabilities = model.predict_proba(
            X_val_scaled
        )[:, 1]

        val_videos = aggregate_videos(
            val_probabilities,
            y_val,
            val_categories,
            val_video_names
        )

        val_predictions = (
            val_videos["probability"].values >= 0.5
        ).astype(int)

        score = f1_score(
            val_videos["label"],
            val_predictions,
            zero_division=0
        )

        print(f"C: {c} | Video Validation F1: {score:.4f}")

        if score > best_f1:
            best_f1 = score
            best_c = c

    final_model = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    )

    final_model.fit(X_train_scaled, y_train)

    test_mask = (
        (categories_test == "real") |
        (categories_test == config["unseen"])
    )

    X_test = X_test_all[test_mask]
    y_test = y_test_all[test_mask]
    test_categories = categories_test[test_mask]
    test_video_names_selected = video_names_test[test_mask]

    X_test_scaled = scaler.transform(X_test)

    test_probabilities = final_model.predict_proba(
        X_test_scaled
    )[:, 1]

    test_videos = aggregate_videos(
        test_probabilities,
        y_test,
        test_categories,
        test_video_names_selected
    )

    predictions = (
        test_videos["probability"].values >= 0.5
    ).astype(int)

    labels = test_videos["label"].values
    probabilities = test_videos["probability"].values

    metrics = {
        "best_C": best_c,
        "video_validation_f1": float(best_f1),
        "test_videos": int(len(test_videos)),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "test_precision": float(precision_score(labels, predictions, zero_division=0)),
        "test_recall": float(recall_score(labels, predictions, zero_division=0)),
        "test_f1": float(f1_score(labels, predictions, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist()
    }

    results[name] = metrics

    print("\nSelected C:", best_c)
    print("Number of test videos:", len(test_videos))
    print("Video-level test metrics:", metrics)

print("\n" + "=" * 60)
print("CONVNEXT VIDEO-LEVEL TEST SUMMARY")
print("=" * 60)

for name, metrics in results.items():
    print(
        name,
        "| Accuracy:", round(metrics["test_accuracy"], 4),
        "| F1:", round(metrics["test_f1"], 4),
        "| AUC:", round(metrics["test_roc_auc"], 4)
    )

with open(
    "experiments/convnext_lomo_video_test_metrics.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\nSaved: experiments/convnext_lomo_video_test_metrics.json")
