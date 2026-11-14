import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT / "datasets" / "metadata" /
    "ffpp_forensic_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" /
    "forensic_baseline_best.joblib"
)

METRICS_FILE = (
    PROJECT_ROOT / "experiments" /
    "forensic_baseline_metrics.json"
)

FEATURE_COLUMNS = [
    "high_frequency_energy",
    "noise_residual_variance",
    "laplacian_variance"
]


# Load frame-level forensic features
data = pd.read_csv(FEATURE_FILE)

# Aggregate frame features into video-level features
video_data = data.groupby(
    ["video_name", "group_id", "split", "label"]
)[FEATURE_COLUMNS].agg(["mean", "std"])

video_data.columns = [
    f"{feature}_{statistic}"
    for feature, statistic in video_data.columns
]

video_data = video_data.reset_index()

video_data = video_data.fillna(0)

# Separate training and validation data
train_data = video_data[
    video_data["split"] == "train"
]

validation_data = video_data[
    video_data["split"] == "validation"
]

model_features = [
    column for column in video_data.columns
    if column not in [
        "video_name", "group_id", "split", "label"
    ]
]

X_train = train_data[model_features]
y_train = train_data["label"]

X_validation = validation_data[model_features]
y_validation = validation_data["label"]

print("Training videos:", len(X_train))
print("Validation videos:", len(X_validation))
print("Forensic inputs:", len(model_features))


# Define models
models_to_train = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
}


results = {}
best_f1 = -1
best_model_name = None
best_model = None


# Train and evaluate
for name, model in models_to_train.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    try:
        auc = roc_auc_score(
            y_validation, probabilities
        )
    except ValueError:
        auc = None

    metrics = {
        "accuracy": float(
            accuracy_score(y_validation, predictions)
        ),
        "precision": float(
            precision_score(
                y_validation, predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_validation, predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_validation, predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(auc) if auc is not None else None
    }

    results[name] = metrics

    print(name, "results:")
    print(metrics)

    if metrics["f1"] > best_f1:

        best_f1 = metrics["f1"]
        best_model_name = name
        best_model = model


# Save best model
joblib.dump(best_model, MODEL_FILE)

# Save experiment results
with open(METRICS_FILE, "w") as file:

    json.dump({
        "best_model": best_model_name,
        "best_validation_f1": best_f1,
        "results": results,
        "input_features": model_features
    }, file, indent=4)


print("\nFORENSIC BASELINE COMPLETED")
print("Best model:", best_model_name)
print("Best validation F1:", best_f1)
print("Saved model:", MODEL_FILE)
print("Saved metrics:", METRICS_FILE)