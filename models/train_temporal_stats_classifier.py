import numpy as np
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = np.load(
    "datasets/metadata/efficientnet_temporal_features_all_splits.npz"
)

sequences = data["sequences"]
labels = data["labels"]
splits = data["splits"]

mean_features = sequences.mean(axis=1)
std_features = sequences.std(axis=1)

temporal_features = np.concatenate(
    [mean_features, std_features],
    axis=1
)

train_mask = splits == "train"
val_mask = splits == "validation"

X_train = temporal_features[train_mask]
y_train = labels[train_mask]

X_val = temporal_features[val_mask]
y_val = labels[val_mask]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

print("\nTEMPORAL MEAN + STANDARD DEVIATION")
print("Training videos:", len(y_train))
print("Validation videos:", len(y_val))
print("Features:", X_train.shape[1])

print("\nAccuracy:", accuracy_score(y_val, predictions))
print("Precision:", precision_score(y_val, predictions))
print("Recall:", recall_score(y_val, predictions))
print("F1:", f1_score(y_val, predictions))
print("ROC-AUC:", roc_auc_score(y_val, probabilities))

output_dir = Path("models/temporal_stats")
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    output_dir / "temporal_mean_std_LogisticRegression.joblib"
)

print("\nModel saved.")