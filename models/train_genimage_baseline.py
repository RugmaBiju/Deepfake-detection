
import os
import copy
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

CSV_PATH = "datasets/metadata/genimage_metadata.csv"
MODEL_PATH = "models/efficientnet_genimage_best.pth"
METRICS_PATH = "experiments/genimage_baseline_metrics.csv"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "validation"].reset_index(drop=True)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class GenImageDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return image, label

train_dataset = GenImageDataset(train_df, train_transform)
val_dataset = GenImageDataset(val_df, val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=True
)

weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE
)

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

best_f1 = 0
best_state = None
history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    model.eval()
    val_loss = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = torch.argmax(outputs, dim=1)

            val_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc = roc_auc_score(all_labels, all_probabilities)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", round(avg_train_loss, 4))
    print("Validation Loss:", round(avg_val_loss, 4))
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "validation_loss": avg_val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })

    if f1 > best_f1:
        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())

if best_state is not None:
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "best_validation_f1": best_f1,
        "image_size": IMAGE_SIZE
    }, MODEL_PATH)

os.makedirs("experiments", exist_ok=True)
pd.DataFrame(history).to_csv(METRICS_PATH, index=False)

print("\nTRAINING COMPLETED")
print("Best validation F1:", best_f1)
print("Model saved to:", MODEL_PATH)
print("Metrics saved to:", METRICS_PATH)
