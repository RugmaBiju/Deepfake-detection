
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import confusion_matrix

data = np.load(
    "datasets/metadata/efficientnet_temporal_features_all_splits.npz"
)

features = data["sequences"].mean(axis=1)
labels = data["labels"]
names = data["video_names"]
splits = data["splits"]

test_mask = splits == "test"

X_test = features[test_mask]
y_test = labels[test_mask]
test_names = names[test_mask]

model = joblib.load(
    "models/hybrid/visual_only_LogisticRegression.joblib"
)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "video_name": test_names,
    "actual_label": y_test,
    "predicted_label": predictions,
    "fake_probability": probabilities
})

results["result"] = np.where(
    results["actual_label"] == results["predicted_label"],
    "Correct",
    "Incorrect"
)

results["error_type"] = np.select(
    [
        (results["actual_label"] == 1) & (results["predicted_label"] == 0),
        (results["actual_label"] == 0) & (results["predicted_label"] == 1)
    ],
    [
        "False Negative - Fake missed",
        "False Positive - Real flagged"
    ],
    default="Correct"
)

results = results.sort_values(
    "fake_probability",
    ascending=False
)

output_dir = Path("experiments/test_evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "ffpp_test_video_predictions.csv"

results.to_csv(output_path, index=False)

print("\nTEST VIDEO ERROR ANALYSIS")
print(results.to_string(index=False))

print("\nERROR SUMMARY")
print(results["error_type"].value_counts())

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nSaved to:", output_path)
