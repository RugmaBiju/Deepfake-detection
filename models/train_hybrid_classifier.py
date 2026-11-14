
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

DATA_PATH = "datasets/metadata/ffpp_hybrid_dataset.csv"
MODEL_DIR = Path("models/hybrid")
RESULT_DIR = Path("experiments/hybrid")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

forensic_columns = [
    "residual_variance_mean",
    "residual_variance_std",
    "residual_mean_absolute_mean",
    "residual_mean_absolute_std",
    "residual_energy_mean",
    "residual_energy_std",
    "low_frequency_ratio_mean",
    "low_frequency_ratio_std",
    "mid_frequency_ratio_mean",
    "mid_frequency_ratio_std",
    "high_frequency_ratio_mean",
    "high_frequency_ratio_std",
    "spectral_centroid_mean",
    "spectral_centroid_std"
]

visual_columns = [
    column for column in df.columns
    if column.startswith("visual_")
]

feature_sets = {
    "visual_only": visual_columns,
    "forensic_only": forensic_columns,
    "hybrid": visual_columns + forensic_columns
}

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

y_train = train_df["label"].to_numpy()
y_val = val_df["label"].to_numpy()

results = []

print("Training videos:", len(train_df))
print("Validation videos:", len(val_df))
print("Visual features:", len(visual_columns))
print("Forensic features:", len(forensic_columns))

for feature_name, columns in feature_sets.items():

    X_train = train_df[columns].to_numpy(dtype=np.float32)
    X_val = val_df[columns].to_numpy(dtype=np.float32)

    print(f"\n{'=' * 45}")
    print("FEATURE SET:", feature_name)
    print("Number of features:", len(columns))

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.1,
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4
        )
    }

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )
        auc = roc_auc_score(y_val, probabilities)

        matrix = confusion_matrix(y_val, predictions)

        print("\nModel:", model_name)
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))
        print("ROC-AUC:", round(auc, 4))
        print("Confusion matrix:\n", matrix)

        results.append({
            "feature_set": feature_name,
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc
        })

        model_path = MODEL_DIR / f"{feature_name}_{model_name}.joblib"
        joblib.dump(model, model_path)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="roc_auc",
    ascending=False
)

results_df.to_csv(
    RESULT_DIR / "hybrid_comparison_metrics.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nMetrics saved to:", RESULT_DIR)
print("Models saved to:", MODEL_DIR)
