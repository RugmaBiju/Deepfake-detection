
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

data = np.load(
    "datasets/metadata/efficientnet_temporal_features_all_splits.npz"
)

visual_features = data["sequences"].mean(axis=1)
labels = data["labels"]
splits = data["splits"]

test_mask = splits == "test"

X_test = visual_features[test_mask]
y_test = labels[test_mask]

model_dir = Path("models/hybrid")

models = {
    "LogisticRegression": joblib.load(
        model_dir / "visual_only_LogisticRegression.joblib"
    ),
    "XGBoost": joblib.load(
        model_dir / "visual_only_XGBoost.joblib"
    )
}

results = []

print("Training split: 140 videos")
print("Test videos:", len(y_test))

for name, model in models.items():

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    print("\nMODEL:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })

results_df = pd.DataFrame(results)

output = Path("experiments/test_evaluation/original_visual_test_metrics.csv")
results_df.to_csv(output, index=False)

print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))
print("\nSaved to:", output)
