import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT / "datasets" / "metadata" /
    "ffpp_frame_metadata.csv"
)

FORENSIC_FILE = (
    PROJECT_ROOT / "datasets" / "metadata" /
    "ffpp_forensic_features.csv"
)

SPATIAL_MODEL_FILE = (
    PROJECT_ROOT / "models" /
    "efficientnet_b0_spatial_best.pth"
)

FORENSIC_MODEL_FILE = (
    PROJECT_ROOT / "models" /
    "forensic_baseline_best.joblib"
)

FORENSIC_METRICS_FILE = (
    PROJECT_ROOT / "experiments" /
    "forensic_baseline_metrics.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "experiments" /
    "hybrid_validation_predictions.csv"
)

METRICS_FILE = (
    PROJECT_ROOT / "experiments" /
    "hybrid_validation_metrics.json"
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16

SPATIAL_WEIGHT = 0.80
FORENSIC_WEIGHT = 0.20


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation dataset
class ValidationDataset(Dataset):

    def __init__(self, metadata):

        self.records = []

        validation_metadata = metadata[
            metadata["split"] == "validation"
        ]

        for _, video in validation_metadata.iterrows():

            frame_folder = Path(video["output_folder"])

            for frame_path in sorted(
                frame_folder.glob("*.png")
            ):

                self.records.append({
                    "image_path": frame_path,
                    "video_name": video["video_name"]
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image = Image.open(
            record["image_path"]
        ).convert("RGB")

        image = transform(image)

        return image, record["video_name"]


# Load metadata
metadata = pd.read_csv(METADATA_FILE)

dataset = ValidationDataset(metadata)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)

print("Validation frames:", len(dataset))


# Load spatial model
print("\nLoading spatial model...")

spatial_model = models.efficientnet_b0(
    weights=None
)

spatial_model.classifier[1] = nn.Linear(
    spatial_model.classifier[1].in_features,
    2
)

checkpoint = torch.load(
    SPATIAL_MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

spatial_model.load_state_dict(
    checkpoint["model_state_dict"]
)

spatial_model = spatial_model.to(DEVICE)
spatial_model.eval()


# Generate spatial predictions
spatial_records = []

with torch.no_grad():

    for images, video_names in tqdm(
        loader,
        desc="Spatial predictions"
    ):

        images = images.to(DEVICE)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda"
        ):

            outputs = spatial_model(images)

        probabilities = torch.softmax(
            outputs.float(), dim=1
        )[:, 1]

        for name, probability in zip(
            video_names,
            probabilities.cpu().numpy()
        ):

            spatial_records.append({
                "video_name": name,
                "spatial_probability": float(probability)
            })


spatial_predictions = pd.DataFrame(
    spatial_records
)

spatial_predictions = spatial_predictions.groupby(
    "video_name"
)["spatial_probability"].mean().reset_index()


# Aggregate forensic features at video level
print("\nLoading forensic model...")

forensic_data = pd.read_csv(FORENSIC_FILE)

forensic_data = forensic_data[
    forensic_data["split"] == "validation"
]

feature_columns = [
    "high_frequency_energy",
    "noise_residual_variance",
    "laplacian_variance"
]

forensic_video_data = forensic_data.groupby(
    "video_name"
)[feature_columns].agg(["mean", "std"])

forensic_video_data.columns = [
    f"{feature}_{statistic}"
    for feature, statistic in forensic_video_data.columns
]

forensic_video_data = forensic_video_data.reset_index()

forensic_video_data = forensic_video_data.fillna(0)

with open(FORENSIC_METRICS_FILE, "r") as file:
    forensic_info = json.load(file)

model_features = forensic_info["input_features"]

forensic_model = joblib.load(
    FORENSIC_MODEL_FILE
)

forensic_probabilities = forensic_model.predict_proba(
    forensic_video_data[model_features]
)[:, 1]

forensic_video_data["forensic_probability"] = (
    forensic_probabilities
)

forensic_predictions = forensic_video_data[
    ["video_name", "forensic_probability"]
]


# Combine predictions
results = spatial_predictions.merge(
    forensic_predictions,
    on="video_name"
)

validation_labels = metadata[
    metadata["split"] == "validation"
][["video_name", "label"]].drop_duplicates()

results = results.merge(
    validation_labels,
    on="video_name"
)


# Weighted late fusion
results["hybrid_probability"] = (
    SPATIAL_WEIGHT * results["spatial_probability"]
    +
    FORENSIC_WEIGHT * results["forensic_probability"]
)


# Evaluation
def evaluate(labels, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            (predictions == labels).mean()
        ),
        "precision": float(
            __import__("sklearn.metrics", fromlist=[
                "precision_score"
            ]).precision_score(
                labels, predictions, zero_division=0
            )
        ),
        "recall": float(
            __import__("sklearn.metrics", fromlist=[
                "recall_score"
            ]).recall_score(
                labels, predictions, zero_division=0
            )
        ),
        "f1": float(
            __import__("sklearn.metrics", fromlist=[
                "f1_score"
            ]).f1_score(
                labels, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            __import__("sklearn.metrics", fromlist=[
                "roc_auc_score"
            ]).roc_auc_score(
                labels, probabilities
            )
        )
    }


labels = results["label"].values

spatial_metrics = evaluate(
    labels,
    results["spatial_probability"].values
)

forensic_metrics = evaluate(
    labels,
    results["forensic_probability"].values
)

hybrid_metrics = evaluate(
    labels,
    results["hybrid_probability"].values
)


# Display results
print("\nSPATIAL BASELINE:")
print(spatial_metrics)

print("\nFORENSIC BASELINE:")
print(forensic_metrics)

print("\nHYBRID MODEL:")
print(hybrid_metrics)


# Save results
results.to_csv(
    OUTPUT_FILE,
    index=False
)

with open(METRICS_FILE, "w") as file:

    json.dump({
        "fusion_method": "weighted_late_fusion",
        "spatial_weight": SPATIAL_WEIGHT,
        "forensic_weight": FORENSIC_WEIGHT,
        "spatial": spatial_metrics,
        "forensic": forensic_metrics,
        "hybrid": hybrid_metrics
    }, file, indent=4)


print("\nHYBRID FUSION EVALUATION COMPLETED")
print("Predictions saved:", OUTPUT_FILE)
print("Metrics saved:", METRICS_FILE)