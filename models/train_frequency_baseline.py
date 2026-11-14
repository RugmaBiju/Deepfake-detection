import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT / "datasets" / "metadata" /
    "ffpp_frame_metadata.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" /
    "efficientnet_b0_frequency_best.pth"
)

HISTORY_FILE = (
    PROJECT_ROOT / "experiments" /
    "frequency_baseline_history.csv"
)

METRICS_FILE = (
    PROJECT_ROOT / "experiments" /
    "frequency_baseline_metrics.json"
)


BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# Frequency image transformation
class FrequencyTransform:

    def __call__(self, image):

        image = np.array(image.convert("RGB"))

        gray = cv2.cvtColor(
            image, cv2.COLOR_RGB2GRAY
        )

        gray = gray.astype(np.float32) / 255.0

        fft = np.fft.fft2(gray)

        fft_shift = np.fft.fftshift(fft)

        magnitude = np.abs(fft_shift)

        spectrum = np.log1p(magnitude)

        spectrum = (
            spectrum - spectrum.min()
        ) / (
            spectrum.max() - spectrum.min() + 1e-8
        )

        spectrum = cv2.resize(
            spectrum,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        spectrum = np.stack(
            [spectrum, spectrum, spectrum],
            axis=0
        )

        tensor = torch.tensor(
            spectrum,
            dtype=torch.float32
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        return (tensor - mean) / std


# Dataset
class FrequencyDataset(Dataset):

    def __init__(self, metadata, split):

        self.records = []

        self.transform = FrequencyTransform()

        split_metadata = metadata[
            metadata["split"] == split
        ]

        for _, video in split_metadata.iterrows():

            frame_folder = Path(video["output_folder"])

            for frame_path in sorted(
                frame_folder.glob("*.png")
            ):

                self.records.append({
                    "image_path": frame_path,
                    "label": int(video["label"]),
                    "video_name": video["video_name"]
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image = Image.open(
            record["image_path"]
        ).convert("RGB")

        image = self.transform(image)

        label = torch.tensor(
            record["label"],
            dtype=torch.long
        )

        return image, label, record["video_name"]


# Load metadata
metadata = pd.read_csv(METADATA_FILE)

train_dataset = FrequencyDataset(
    metadata, "train"
)

validation_dataset = FrequencyDataset(
    metadata, "validation"
)

print("\nTraining frames:", len(train_dataset))
print("Validation frames:", len(validation_dataset))


# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)


# Load EfficientNet-B0
print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model = model.to(DEVICE)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=DEVICE.type == "cuda"
)


# Metrics
def calculate_metrics(labels, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(
                labels, predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                labels, predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                labels, predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        )
    }


# Validation
def evaluate_model():

    model.eval()

    labels_all = []
    probabilities_all = []
    video_names_all = []

    with torch.no_grad():

        for images, labels, video_names in tqdm(
            validation_loader,
            desc="Validation"
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda"
            ):

                outputs = model(images)

            probabilities = torch.softmax(
                outputs.float(), dim=1
            )[:, 1]

            labels_all.extend(
                labels.cpu().numpy()
            )

            probabilities_all.extend(
                probabilities.cpu().numpy()
            )

            video_names_all.extend(video_names)

    frame_metrics = calculate_metrics(
        np.array(labels_all),
        np.array(probabilities_all)
    )

    video_results = pd.DataFrame({
        "video_name": video_names_all,
        "label": labels_all,
        "probability": probabilities_all
    })

    video_results = video_results.groupby(
        "video_name"
    ).agg({
        "label": "first",
        "probability": "mean"
    }).reset_index()

    video_metrics = calculate_metrics(
        video_results["label"].values,
        video_results["probability"].values
    )

    return frame_metrics, video_metrics


# Training
best_f1 = -1
history = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels, _ in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda"
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = running_loss / total
    train_accuracy = correct / total

    frame_metrics, video_metrics = evaluate_model()

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")

    print("\nValidation Frame Metrics:")
    print(frame_metrics)

    print("\nValidation Video Metrics:")
    print(video_metrics)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_frame_accuracy": frame_metrics["accuracy"],
        "validation_video_accuracy": video_metrics["accuracy"],
        "validation_video_f1": video_metrics["f1"],
        "validation_video_auc": video_metrics["roc_auc"]
    })

    if video_metrics["f1"] > best_f1:

        best_f1 = video_metrics["f1"]

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "validation_video_metrics": video_metrics,
            "class_mapping": {
                "authentic": 0,
                "manipulated": 1
            }
        }, MODEL_FILE)

        print("\nBest frequency model saved!")


# Save experiment history
pd.DataFrame(history).to_csv(
    HISTORY_FILE,
    index=False
)

with open(METRICS_FILE, "w") as file:

    json.dump({
        "best_validation_video_f1": best_f1,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "best_model": str(MODEL_FILE)
    }, file, indent=4)


print("\nFREQUENCY BASELINE TRAINING COMPLETED")
print("Best validation video F1:", best_f1)
print("Model saved:", MODEL_FILE)
print("History saved:", HISTORY_FILE)
print("Metrics saved:", METRICS_FILE)