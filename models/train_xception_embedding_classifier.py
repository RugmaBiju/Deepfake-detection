
import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

DATA_PATH = "datasets/metadata/xception_video_embeddings.csv"

data = pd.read_csv(DATA_PATH)

feature_columns = [
    column for column in data.columns
    if column.startswith("feature_")
]

train_data = data[data["split"] == "train"]
val_data = data[data["split"] == "validation"]

X_train = train_data[feature_columns]
y_train = train_data["label"]

X_val = val_data[feature_columns]
y_val = val_data["label"]

print("Training videos:", len(X_train))
print("Validation videos:", len(X_val))
print("Features per video:", len(feature_columns))

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=0.1,
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=0.1,
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = {}
best_f1 = -1
best_model_name = None
best_model = None
best_predictions = None
best_probabilities = None

os.makedirs("models", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

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

    results[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": confusion_matrix(
            y_val, predictions
        ).tolist()
    }

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model
        best_predictions = predictions
        best_probabilities = probabilities

joblib.dump(
    best_model,
    "models/xception_embedding_classifier.joblib"
)

prediction_data = pd.DataFrame({
    "video_name": val_data["video_name"].values,
    "actual_label": y_val.values,
    "predicted_label": best_predictions,
    "fake_probability": best_probabilities
})

prediction_data.to_csv(
    "experiments/xception_embedding_predictions.csv",
    index=False
)

with open(
    "experiments/xception_embedding_metrics.json",
    "w"
) as file:
    json.dump({
        "best_model": best_model_name,
        "best_f1": float(best_f1),
        "results": results
    }, file, indent=4)

print("\nXCEPTION CLASSIFIER TRAINING COMPLETED")
print("Best classifier:", best_model_name)
print("Best validation F1:", round(best_f1, 4))
print("Saved model: models/xception_embedding_classifier.joblib")
