
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "datasets" / "metadata" / "cross_manipulation_features"
OUTPUT_DIR = ROOT / "experiments" / "lomo_generalization_dinov2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]

EXPERIMENTS = {
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


def load_trainval(category):
    path = FEATURE_DIR / f"{category}_dinov2_features_trainval.npz"

    data = np.load(path, allow_pickle=True)

    features = data["features"]
    labels = data["labels"]
    video_names = data["video_names"]
    splits = data["splits"]

    video_features = features.mean(axis=1)

    return {
        "features": video_features,
        "labels": labels,
        "video_names": video_names,
        "splits": splits
    }


def load_test(category):
    path = FEATURE_DIR / f"{category}_dinov2_features.npz"

    data = np.load(path, allow_pickle=True)

    features = data["features"]
    labels = data["labels"]
    video_names = data["video_names"]

    video_features = features.mean(axis=1)

    return video_features, labels, video_names


def combine_seen_categories(categories, split):
    all_features = []
    all_labels = []
    all_names = []

    seen_real_names = set()

    for category in categories:
        data = load_trainval(category)

        for i in range(len(data["labels"])):
            current_split = str(data["splits"][i]).lower()

            if current_split in ["val", "valid"]:
                current_split = "validation"

            if current_split != split:
                continue

            label = int(data["labels"][i])
            video_name = str(data["video_names"][i])

            if label == 0 and video_name in seen_real_names:
                continue

            if label == 0:
                seen_real_names.add(video_name)

            all_features.append(data["features"][i])
            all_labels.append(label)
            all_names.append(video_name)

    return (
        np.array(all_features),
        np.array(all_labels),
        np.array(all_names)
    )


def calculate_metrics(y_true, y_pred, probabilities):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities)
    }


def run_experiment(experiment_name, seen_categories, unseen_category):

    print("\n" + "=" * 60)
    print(f"EXPERIMENT {experiment_name}")
    print(f"Seen categories: {seen_categories}")
    print(f"Unseen category: {unseen_category}")
    print("=" * 60)

    X_train, y_train, train_names = combine_seen_categories(
        seen_categories, "train"
    )

    X_val, y_val, val_names = combine_seen_categories(
        seen_categories, "validation"
    )

    print("\nTRAIN DATA")
    print("Shape:", X_train.shape)
    print("Real:", np.sum(y_train == 0))
    print("Fake:", np.sum(y_train == 1))

    print("\nVALIDATION DATA")
    print("Shape:", X_val.shape)
    print("Real:", np.sum(y_val == 0))
    print("Fake:", np.sum(y_val == 1))

    best_model = None
    best_C = None
    best_metrics = None
    best_score = None

    print("\nMODEL SELECTION")

    for C in C_VALUES:

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=3000,
                random_state=42
            ))
        ])

        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)
        val_probabilities = model.predict_proba(X_val)[:, 1]

        metrics = calculate_metrics(
            y_val,
            val_predictions,
            val_probabilities
        )

        print(
            f"C={C} | "
            f"Validation F1={metrics['f1']:.4f} | "
            f"Validation AUC={metrics['roc_auc']:.4f}"
        )

        score = (metrics["f1"], metrics["roc_auc"], -C)

        if best_score is None or score > best_score:
            best_score = score
            best_C = C
            best_metrics = metrics
            best_model = model

    print("\nSELECTED MODEL")
    print("Best C:", best_C)
    print("Validation metrics:", best_metrics)

    X_combined = np.concatenate([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=best_C,
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ))
    ])

    final_model.fit(X_combined, y_combined)

    X_test, y_test, test_names = load_test(unseen_category)

    print("\nUNSEEN TEST DATA")
    print("Shape:", X_test.shape)
    print("Real:", np.sum(y_test == 0))
    print("Fake:", np.sum(y_test == 1))

    test_predictions = final_model.predict(X_test)
    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities
    )

    print("\nFINAL TEST RESULTS")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    predictions_df = pd.DataFrame({
        "video_name": test_names,
        "actual_label": y_test,
        "predicted_label": test_predictions,
        "fake_probability": test_probabilities
    })

    predictions_df["result"] = np.where(
        predictions_df["actual_label"] == predictions_df["predicted_label"],
        "Correct",
        "Incorrect"
    )

    predictions_df.to_csv(
        OUTPUT_DIR / f"experiment_{experiment_name}_test_predictions.csv",
        index=False
    )

    validation_df = pd.DataFrame([{
        "experiment": experiment_name,
        "seen_categories": "+".join(seen_categories),
        "unseen_category": unseen_category,
        "selected_C": best_C,
        **best_metrics
    }])

    validation_df.to_csv(
        OUTPUT_DIR / f"experiment_{experiment_name}_validation.csv",
        index=False
    )

    return {
        "experiment": experiment_name,
        "seen_categories": "+".join(seen_categories),
        "unseen_category": unseen_category,
        "selected_C": best_C,
        **test_metrics
    }


results = []

for experiment_name, config in EXPERIMENTS.items():

    result = run_experiment(
        experiment_name,
        config["seen"],
        config["unseen"]
    )

    results.append(result)


summary_df = pd.DataFrame(results)

summary_df.to_csv(
    OUTPUT_DIR / "lomo_dinov2_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("DINOv2 LOMO SUMMARY")
print("=" * 60)

print(summary_df.to_string(index=False))

print("\nMean Test Accuracy:", summary_df["accuracy"].mean())
print("Mean Test F1:", summary_df["f1"].mean())
print("Mean Test AUC:", summary_df["roc_auc"].mean())

print("\nResults saved to:", OUTPUT_DIR)
