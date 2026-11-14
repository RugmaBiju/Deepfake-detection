
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

FEATURE_DIR = (
    ROOT / "datasets" / "metadata" /
    "cross_manipulation_features"
)

OUTPUT_DIR = (
    ROOT / "experiments" /
    "lomo_threshold_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]

MODELS = ["B0", "DINOv2", "Fusion"]

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


def load_features(category, model_type, split_type):

    suffix = "_trainval.npz" if split_type == "trainval" else ".npz"

    if model_type == "B0":

        path = FEATURE_DIR / f"{category}_b0_features{suffix}"
        data = np.load(path, allow_pickle=True)

        features = data["sequences"]

    elif model_type == "DINOv2":

        path = FEATURE_DIR / f"{category}_dinov2_features{suffix}"
        data = np.load(path, allow_pickle=True)

        features = data["features"]

    else:

        b0_path = FEATURE_DIR / f"{category}_b0_features{suffix}"
        dino_path = FEATURE_DIR / f"{category}_dinov2_features{suffix}"

        b0_data = np.load(b0_path, allow_pickle=True)
        dino_data = np.load(dino_path, allow_pickle=True)

        b0_features = b0_data["sequences"]
        dino_features = dino_data["features"]

        b0_names = b0_data["video_names"].astype(str)
        dino_names = dino_data["video_names"].astype(str)

        dino_name_to_index = {
            name: i for i, name in enumerate(dino_names)
        }

        if len(dino_name_to_index) != len(dino_names):
            raise ValueError(f"Duplicate DINOv2 video names: {category}")

        if set(b0_names) != set(dino_names):
            raise ValueError(f"B0 and DINOv2 video mismatch: {category}")

        order = np.array([
            dino_name_to_index[name] for name in b0_names
        ])

        dino_features = dino_features[order]

        if not np.array_equal(
            b0_data["labels"],
            dino_data["labels"][order]
        ):
            raise ValueError(f"Label mismatch: {category}")

        if split_type == "trainval":

            if not np.array_equal(
                b0_data["splits"].astype(str),
                dino_data["splits"][order].astype(str)
            ):
                raise ValueError(f"Split mismatch: {category}")

            if not np.array_equal(
                b0_data["group_ids"].astype(str),
                dino_data["group_ids"][order].astype(str)
            ):
                raise ValueError(f"Identity group mismatch: {category}")

        features = np.concatenate(
            [
                b0_features.mean(axis=1),
                dino_features.mean(axis=1)
            ],
            axis=1
        )

        return {
            "features": features,
            "labels": b0_data["labels"],
            "video_names": b0_names,
            "splits": (
                b0_data["splits"]
                if split_type == "trainval"
                else None
            )
        }

    video_features = features.mean(axis=1)

    return {
        "features": video_features,
        "labels": data["labels"],
        "video_names": data["video_names"].astype(str),
        "splits": (
            data["splits"]
            if split_type == "trainval"
            else None
        )
    }


def combine_seen_categories(categories, model_type, split_name):

    all_features = []
    all_labels = []
    all_names = []

    included_real_videos = set()

    for category in categories:

        data = load_features(
            category,
            model_type,
            "trainval"
        )

        for i in range(len(data["labels"])):

            current_split = str(data["splits"][i]).lower()

            if current_split in ["val", "valid"]:
                current_split = "validation"

            if current_split != split_name:
                continue

            label = int(data["labels"][i])
            video_name = str(data["video_names"][i])

            if label == 0 and video_name in included_real_videos:
                continue

            if label == 0:
                included_real_videos.add(video_name)

            all_features.append(data["features"][i])
            all_labels.append(label)
            all_names.append(video_name)

    return (
        np.asarray(all_features, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_names, dtype=str)
    )


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities)
    }


def select_threshold(y_true, probabilities):

    thresholds = np.unique(
        np.concatenate([
            np.arange(0.05, 0.951, 0.01),
            [0.5]
        ])
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        results.append({
            "threshold": threshold,
            **metrics
        })

    results_df = pd.DataFrame(results)

    # Maximize validation F1.
    # If tied, prefer the threshold closest to 0.5.
    results_df["distance_from_05"] = abs(
        results_df["threshold"] - 0.5
    )

    results_df = results_df.sort_values(
        ["f1", "distance_from_05"],
        ascending=[False, True]
    )

    best_row = results_df.iloc[0]

    return (
        float(best_row["threshold"]),
        results_df
    )


def run_experiment(experiment_name, config, model_type):

    seen = config["seen"]
    unseen = config["unseen"]

    print("\n" + "=" * 65)
    print(f"MODEL: {model_type}")
    print(f"EXPERIMENT: {experiment_name}")
    print("Seen:", seen)
    print("Unseen:", unseen)
    print("=" * 65)

    X_train, y_train, train_names = combine_seen_categories(
        seen, model_type, "train"
    )

    X_val, y_val, val_names = combine_seen_categories(
        seen, model_type, "validation"
    )

    print("\nTraining shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    best_C = None
    best_score = None
    best_val_probabilities = None

    validation_records = []

    for c_value in C_VALUES:

        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42
                )
            )
        ])

        model.fit(X_train, y_train)

        val_probabilities = model.predict_proba(X_val)[:, 1]

        metrics = calculate_metrics(
            y_val,
            val_probabilities,
            0.5
        )

        validation_records.append({
            "C": c_value,
            **metrics
        })

        score = (
            metrics["f1"],
            metrics["roc_auc"],
            -c_value
        )

        if best_score is None or score > best_score:

            best_score = score
            best_C = c_value
            best_val_probabilities = val_probabilities.copy()

    validation_df = pd.DataFrame(validation_records)

    validation_df.to_csv(
        OUTPUT_DIR /
        f"{model_type}_experiment_{experiment_name}_C_selection.csv",
        index=False
    )

    print("\nSelected C:", best_C)

    # Select threshold using validation probabilities only.
    best_threshold, threshold_df = select_threshold(
        y_val,
        best_val_probabilities
    )

    threshold_df.to_csv(
        OUTPUT_DIR /
        f"{model_type}_experiment_{experiment_name}_thresholds.csv",
        index=False
    )

    print("Selected threshold:", round(best_threshold, 3))

    # Refit using training + validation data.
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=best_C,
                class_weight="balanced",
                max_iter=3000,
                random_state=42
            )
        )
    ])

    final_model.fit(X_combined, y_combined)

    # Load untouched unseen test features.
    test_data = load_features(
        unseen,
        model_type,
        "test"
    )

    X_test = test_data["features"]
    y_test = test_data["labels"]
    test_names = test_data["video_names"]

    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    # Evaluate both thresholds on exactly the same test probabilities.
    metrics_default = calculate_metrics(
        y_test,
        test_probabilities,
        0.5
    )

    metrics_optimized = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold
    )

    print("\nTEST RESULTS")

    print("\nThreshold 0.50:")
    for key, value in metrics_default.items():
        print(f"{key}: {value:.4f}")

    print("\nValidation-selected threshold:", best_threshold)
    for key, value in metrics_optimized.items():
        print(f"{key}: {value:.4f}")

    test_predictions = pd.DataFrame({
        "video_name": test_names,
        "actual_label": y_test,
        "fake_probability": test_probabilities,
        "prediction_05": (
            test_probabilities >= 0.5
        ).astype(int),
        "prediction_optimized": (
            test_probabilities >= best_threshold
        ).astype(int)
    })

    test_predictions.to_csv(
        OUTPUT_DIR /
        f"{model_type}_experiment_{experiment_name}_test_predictions.csv",
        index=False
    )

    return {
        "model": model_type,
        "experiment": experiment_name,
        "unseen_manipulation": unseen,
        "selected_C": best_C,
        "selected_threshold": best_threshold,

        **{
            f"default_{key}": value
            for key, value in metrics_default.items()
        },

        **{
            f"optimized_{key}": value
            for key, value in metrics_optimized.items()
        }
    }


# Run all nine model-experiment combinations.

all_results = []

for model_type in MODELS:

    for experiment_name, config in EXPERIMENTS.items():

        result = run_experiment(
            experiment_name,
            config,
            model_type
        )

        all_results.append(result)


summary_df = pd.DataFrame(all_results)

summary_df.to_csv(
    OUTPUT_DIR / "threshold_comparison_summary.csv",
    index=False
)


print("\n" + "=" * 65)
print("FINAL THRESHOLD COMPARISON")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nMEAN RESULTS")

for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:

    default_mean = summary_df[
        f"default_{metric}"
    ].mean()

    optimized_mean = summary_df[
        f"optimized_{metric}"
    ].mean()

    print(
        f"{metric.upper()}: "
        f"Default={default_mean:.4f} | "
        f"Optimized={optimized_mean:.4f}"
    )


print("\nAll results saved in:", OUTPUT_DIR)
print("\nTHRESHOLD EXPERIMENT COMPLETED")
