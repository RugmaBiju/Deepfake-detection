
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

spatial = pd.read_csv("experiments/hybrid_validation_predictions.csv")
gru = pd.read_csv("experiments/temporal_gru_predictions.csv")

comparison = pd.merge(
    spatial[["video_name", "label", "spatial_probability"]],
    gru[["video_name", "actual_label", "fake_probability"]],
    on="video_name"
)

comparison = comparison.rename(columns={
    "label": "actual_label_spatial",
    "spatial_probability": "spatial_probability",
    "fake_probability": "gru_probability"
})

comparison["spatial_prediction"] = (comparison["spatial_probability"] >= 0.5).astype(int)
comparison["gru_prediction"] = (comparison["gru_probability"] >= 0.5).astype(int)

y_true = comparison["actual_label_spatial"]

print("Videos compared:", len(comparison))

for name, prediction, probability in [
    ("EfficientNet", comparison["spatial_prediction"], comparison["spatial_probability"]),
    ("GRU", comparison["gru_prediction"], comparison["gru_probability"])
]:
    print("\n", name)
    print("Accuracy:", accuracy_score(y_true, prediction))
    print("Precision:", precision_score(y_true, prediction, zero_division=0))
    print("Recall:", recall_score(y_true, prediction, zero_division=0))
    print("F1:", f1_score(y_true, prediction, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_true, probability))

comparison.to_csv("experiments/temporal_model_comparison.csv", index=False)

print("\nComparison saved to experiments/temporal_model_comparison.csv")
