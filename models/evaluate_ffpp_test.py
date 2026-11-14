
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

PROJECT = Path(".")
FEATURE_FILE = PROJECT / "datasets/metadata/efficientnet_temporal_features_all_splits.npz"
FORENSIC_FILE = PROJECT / "datasets/metadata/ffpp_face_forensic_video_features.csv"

MODEL_DIR = PROJECT / "models/face_hybrid"
OUTPUT_DIR = PROJECT / "experiments/test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all EfficientNet features
data = np.load(FEATURE_FILE)

sequences = data["sequences"]
labels = data["labels"]
video_names = data["video_names"]
splits = data["splits"]

visual_features = sequences.mean(axis=1)

visual_columns = [f"visual_{i}" for i in range(visual_features.shape[1])]

visual_df = pd.DataFrame(visual_features, columns=visual_columns)
visual_df["video_name"] = video_names
visual_df["label"] = labels
visual_df["split"] = splits

# Select held-out test videos only
test_visual = visual_df[visual_df["split"] == "test"].copy()

# Load face forensic features
forensic_df = pd.read_csv(FORENSIC_FILE)

forensic_columns = [
    column for column in forensic_df.columns
    if column.endswith("_mean") or column.endswith("_std")
]

# Match test videos with available face features
test_hybrid = test_visual.merge(
    forensic_df[["video_name"] + forensic_columns],
    on="video_name",
    how="inner",
    validate="one_to_one"
)

print("Held-out test videos:", len(test_visual))
print("Test videos with face features:", len(test_hybrid))

results = []

def evaluate_model(model, X, y, name):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    auc = roc_auc_score(y, probabilities)

    print("\n", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion matrix:\n", confusion_matrix(y, predictions))

    results.append({
        "experiment": name,
        "videos": len(y),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })


# Full 32-video test evaluation
y_test = test_visual["label"].to_numpy()
X_test_visual = test_visual[visual_columns].to_numpy()

visual_model = joblib.load(
    MODEL_DIR / "visual_only_LogisticRegression.joblib"
)

evaluate_model(
    visual_model,
    X_test_visual,
    y_test,
    "Visual-only - all test videos"
)

# Fair comparison on videos with face features
y_hybrid = test_hybrid["label"].to_numpy()

X_hybrid_visual = test_hybrid[visual_columns].to_numpy()
X_hybrid_forensic = test_hybrid[forensic_columns].to_numpy()

X_hybrid = np.column_stack([
    X_hybrid_visual,
    X_hybrid_forensic
])

visual_model = joblib.load(
    MODEL_DIR / "visual_only_LogisticRegression.joblib"
)

hybrid_model = joblib.load(
    MODEL_DIR / "face_hybrid_LogisticRegression.joblib"
)

evaluate_model(
    visual_model,
    X_hybrid_visual,
    y_hybrid,
    "Visual-only - matched face subset"
)

evaluate_model(
    hybrid_model,
    X_hybrid,
    y_hybrid,
    "Face-hybrid - matched face subset"
)

results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "ffpp_test_metrics.csv"
results_df.to_csv(results_path, index=False)

print("\nFINAL TEST COMPARISON")
print(results_df.to_string(index=False))
print("\nSaved to:", results_path)

