
import numpy as np
import pandas as pd

from pathlib import Path

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

FEATURE_ROOT = (
    ROOT / "datasets" / "metadata" /
    "cross_manipulation_features"
)

OUTPUT_ROOT = (
    ROOT / "experiments" /
    "lomo_generalization"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


CATEGORIES = [
    "face2face",
    "faceswap",
    "neuraltextures"
]

C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]


EXPERIMENTS = [
    {
        "name": "A",
        "seen": ["face2face", "faceswap"],
        "unseen": "neuraltextures"
    },
    {
        "name": "B",
        "seen": ["face2face", "neuraltextures"],
        "unseen": "faceswap"
    },
    {
        "name": "C",
        "seen": ["faceswap", "neuraltextures"],
        "unseen": "face2face"
    }
]


def load_trainval(category):

    path = (
        FEATURE_ROOT /
        f"{category}_b0_features_trainval.npz"
    )

    data = np.load(path)

    features = data["sequences"].mean(axis=1)

    return {
        "features": features,
        "labels": data["labels"],
        "video_names": data["video_names"],
        "group_ids": data["group_ids"],
        "splits": data["splits"]
    }


def load_test(category):

    path = (
        FEATURE_ROOT /
        f"{category}_b0_features.npz"
    )

    data = np.load(path)

    return {
        "features": data["sequences"].mean(axis=1),
        "labels": data["labels"],
        "video_names": data["video_names"]
    }


def combine_seen_categories(categories, datasets, split_name):

    features = []
    labels = []
    video_names = []

    included_real_videos = set()

    for category in categories:

        data = datasets[category]

        indices = np.where(
            data["splits"] == split_name
        )[0]

        for index in indices:

            video_name = str(data["video_names"][index])
            label = int(data["labels"][index])

            if label == 0:

                if video_name in included_real_videos:
                    continue

                included_real_videos.add(video_name)

            features.append(data["features"][index])
            labels.append(label)
            video_names.append(video_name)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(video_names, dtype=str)
    )


def calculate_metrics(y_true, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

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
        "roc_auc": roc_auc_score(
            y_true, probabilities
        )
    }


print("\nLOMO EXPERIMENTS: EFFICIENTNET-B0")
print("=" * 60)

datasets = {}

for category in CATEGORIES:

    datasets[category] = load_trainval(category)

    print(
        category.upper(),
        "features:",
        datasets[category]["features"].shape
    )


summary_records = []


for experiment in EXPERIMENTS:

    name = experiment["name"]
    seen = experiment["seen"]
    unseen = experiment["unseen"]

    print("\n" + "=" * 60)
    print("EXPERIMENT:", name)
    print("Training methods:", seen)
    print("Unseen test method:", unseen.upper())
    print("=" * 60)

    X_train, y_train, train_names = combine_seen_categories(
        seen, datasets, "train"
    )

    X_val, y_val, val_names = combine_seen_categories(
        seen, datasets, "validation"
    )

    print("\nTRAINING DATA")
    print("Feature shape:", X_train.shape)
    print("Real:", np.sum(y_train == 0))
    print("Fake:", np.sum(y_train == 1))

    print("\nVALIDATION DATA")
    print("Feature shape:", X_val.shape)
    print("Real:", np.sum(y_val == 0))
    print("Fake:", np.sum(y_val == 1))

    validation_results = []

    best_model = None
    best_c = None
    best_metrics = None

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
            y_val, val_probabilities
        )

        validation_results.append({
            "C": c_value,
            **metrics
        })

        print(
            f"C={c_value:<7} "
            f"F1={metrics['f1']:.4f} "
            f"AUC={metrics['roc_auc']:.4f}"
        )

        current_score = (
            metrics["f1"],
            metrics["roc_auc"],
            -c_value
        )

        if best_metrics is None:

            best_model = model
            best_c = c_value
            best_metrics = metrics
            best_score = current_score

        elif current_score > best_score:

            best_model = model
            best_c = c_value
            best_metrics = metrics
            best_score = current_score


    validation_df = pd.DataFrame(validation_results)

    validation_file = (
        OUTPUT_ROOT /
        f"experiment_{name}_validation.csv"
    )

    validation_df.to_csv(validation_file, index=False)

    print("\nSELECTED C:", best_c)
    print("Best validation metrics:", best_metrics)


    X_combined = np.concatenate(
        [X_train, X_val], axis=0
    )

    y_combined = np.concatenate(
        [y_train, y_val], axis=0
    )

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=best_c,
                class_weight="balanced",
                max_iter=3000,
                random_state=42
            )
        )
    ])

    final_model.fit(X_combined, y_combined)


    print("\nLOADING UNSEEN TEST DATA")

    test_data = load_test(unseen)

    X_test = test_data["features"]
    y_test = test_data["labels"]
    test_names = test_data["video_names"]

    print("Test feature shape:", X_test.shape)
    print("Test real:", np.sum(y_test == 0))
    print("Test fake:", np.sum(y_test == 1))


    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics(
        y_test, test_probabilities
    )

    print("\nUNSEEN MANIPULATION TEST RESULTS")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    predictions_df = pd.DataFrame({
        "video_name": test_names,
        "actual_label": y_test,
        "predicted_label": test_predictions,
        "fake_probability": test_probabilities
    })

    predictions_file = (
        OUTPUT_ROOT /
        f"experiment_{name}_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False
    )


    summary_records.append({
        "experiment": name,
        "training_methods": "+".join(seen),
        "unseen_method": unseen,
        "selected_C": best_c,
        **test_metrics
    })


summary_df = pd.DataFrame(summary_records)

summary_file = OUTPUT_ROOT / "lomo_b0_summary.csv"

summary_df.to_csv(summary_file, index=False)


print("\n" + "=" * 60)
print("FINAL LOMO RESULTS")
print("=" * 60)

print(summary_df.to_string(index=False))

print("\nSummary saved:", summary_file)

print("\nALL LOMO EXPERIMENTS COMPLETED")
