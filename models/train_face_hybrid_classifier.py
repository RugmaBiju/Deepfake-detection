
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
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

from xgboost import XGBClassifier


data_path = Path("datasets/metadata/ffpp_face_hybrid_dataset.csv")

model_dir = Path("models/face_hybrid")
results_dir = Path("experiments/face_hybrid")

model_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(data_path)

train = data[data["split"] == "train"]
validation = data[data["split"] == "validation"]

visual_columns = [
    column for column in data.columns
    if column.startswith("visual_")
]

forensic_columns = [
    column for column in data.columns
    if column.endswith("_mean") or column.endswith("_std")
]

feature_sets = {
    "visual_only": visual_columns,
    "forensic_only": forensic_columns,
    "face_hybrid": visual_columns + forensic_columns
}

X_train_labels = train["label"].to_numpy()
y_validation = validation["label"].to_numpy()

print("Training videos:", len(train))
print("Validation videos:", len(validation))
print("Visual features:", len(visual_columns))
print("Face forensic features:", len(forensic_columns))


results = []


for feature_name, columns in feature_sets.items():

    X_train = train[columns].to_numpy()
    X_validation = validation[columns].to_numpy()

    print("\n" + "=" * 55)
    print("FEATURE SET:", feature_name)
    print("=" * 55)

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

        model.fit(X_train, X_train_labels)

        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)[:, 1]

        accuracy = accuracy_score(y_validation, predictions)
        precision = precision_score(
            y_validation, predictions, zero_division=0
        )
        recall = recall_score(
            y_validation, predictions, zero_division=0
        )
        f1 = f1_score(
            y_validation, predictions, zero_division=0
        )
        auc = roc_auc_score(y_validation, probabilities)

        matrix = confusion_matrix(y_validation, predictions)

        print("\n" + model_name)
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))
        print("ROC-AUC:", round(auc, 4))
        print("Confusion matrix:\n", matrix)

        model_path = model_dir / f"{feature_name}_{model_name}.joblib"
        joblib.dump(model, model_path)

        results.append({
            "feature_set": feature_name,
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc
        })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1",
    ascending=False
)

results_path = results_dir / "face_hybrid_comparison_metrics.csv"

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

print("\nResults saved to:", results_path)
print("Models saved to:", model_dir)
