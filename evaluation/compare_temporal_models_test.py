
import numpy as np
import joblib

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

sequences = data["sequences"]
labels = data["labels"]
splits = data["splits"]

test_mask = splits == "test"

X_test_sequences = sequences[test_mask]
y_test = labels[test_mask]

X_test_mean = X_test_sequences.mean(axis=1)
X_test_std = X_test_sequences.std(axis=1)

X_test_mean_std = np.concatenate(
    [X_test_mean, X_test_std],
    axis=1
)

models = {
    "Temporal Mean": (
        "models/hybrid/visual_only_LogisticRegression.joblib",
        X_test_mean
    ),
    "Temporal Mean + Std": (
        "models/temporal_stats/temporal_mean_std_LogisticRegression.joblib",
        X_test_mean_std
    )
}

print("\nHELD-OUT TEST COMPARISON")
print("Test videos:", len(y_test))

for name, (model_path, features) in models.items():

    model = joblib.load(model_path)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1:", f1_score(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
