
import os
import json
import torch
import timm
import pandas as pd
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.amp import autocast, GradScaler

ROOT = os.getcwd()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 8
LR = 0.0001

FRAME_DIR = os.path.join(ROOT, "datasets", "processed", "ffpp_frames")

metadata = pd.read_csv(
    os.path.join(ROOT, "datasets", "metadata", "ffpp_frame_metadata.csv")
)

transform_train = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class VideoFrameDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        folder = os.path.join(FRAME_DIR, row["output_folder"])
        files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))

        images = []
        for file in files:
            image = Image.open(os.path.join(folder, file)).convert("RGB")
            images.append(self.transform(image))

        return torch.stack(images), int(row["label"]), row["video_name"]


train_data = metadata[metadata["split"] == "train"]
val_data = metadata[metadata["split"] == "validation"]

train_dataset = VideoFrameDataset(train_data, transform_train)
val_dataset = VideoFrameDataset(val_data, transform_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = timm.create_model(
    "legacy_xception.tf_in1k",
    pretrained=True,
    num_classes=2
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")

best_f1 = 0

os.makedirs("models", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

history = []

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):

        batch_size, frames, channels, height, width = images.shape

        images = images.view(
            batch_size * frames, channels, height, width
        ).to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast("cuda", enabled=DEVICE.type == "cuda"):
            outputs = model(images)
            frame_labels = labels.repeat_interleave(frames)
            loss = criterion(outputs, frame_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    model.eval()

    predictions = []
    probabilities = []
    actual = []
    video_names = []

    with torch.no_grad():

        for images, labels, names in val_loader:

            batch_size, frames, channels, height, width = images.shape

            images = images.view(
                batch_size * frames, channels, height, width
            ).to(DEVICE)

            with autocast("cuda", enabled=DEVICE.type == "cuda"):
                outputs = model(images)

            probs = torch.softmax(outputs.float(), dim=1)[:, 1]
            probs = probs.view(batch_size, frames).mean(dim=1)

            predictions.extend((probs >= 0.5).cpu().int().tolist())
            probabilities.extend(probs.cpu().tolist())
            actual.extend(labels.tolist())
            video_names.extend(names)

    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions, zero_division=0)
    recall = recall_score(actual, predictions, zero_division=0)
    f1 = f1_score(actual, predictions, zero_division=0)
    auc = roc_auc_score(actual, probabilities)

    avg_loss = total_loss / len(train_loader)

    print(f"\nEpoch: {epoch + 1}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Video Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    history.append({
        "epoch": epoch + 1,
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })

    if f1 > best_f1:
        best_f1 = f1

        torch.save(model.state_dict(), "models/xception_best.pth")

        with open("experiments/xception_best_metrics.json", "w") as f:
            json.dump({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc
            }, f, indent=4)

    pd.DataFrame(history).to_csv(
        "experiments/xception_history.csv",
        index=False
    )

print("\nXCEPTION TRAINING COMPLETED")
print("Best validation F1:", best_f1)
print("Saved: models/xception_best.pth")

print("\nChecking Xception data loading...")

images, labels, names = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Labels shape:", labels.shape)
print("Video names:", names[:3])
print("GPU:", torch.cuda.get_device_name(0))
