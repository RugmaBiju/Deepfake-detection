
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT / "datasets" / "metadata" /
    "efficientnet_temporal_features.npz"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" /
    "efficientnet_gru_temporal_best.pth"
)

METRICS_FILE = (
    PROJECT_ROOT / "experiments" /
    "temporal_gru_metrics.json"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT / "experiments" /
    "temporal_gru_predictions.csv"
)

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0005
PATIENCE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("Device:", DEVICE)


print("\nLoading temporal features...")

data = np.load(DATA_FILE)

sequences = data["sequences"].astype(np.float32)
labels = data["labels"].astype(np.float32)
video_names = data["video_names"]
splits = data["splits"]

train_mask = splits == "train"
val_mask = splits == "validation"

X_train = sequences[train_mask]
X_val = sequences[val_mask]

y_train = labels[train_mask]
y_val = labels[val_mask]

train_names = video_names[train_mask]
val_names = video_names[val_mask]

print("Training videos:", len(X_train))
print("Validation videos:", len(X_val))
print("Sequence shape:", X_train.shape)


# Normalize using training data only

feature_mean = X_train.reshape(
    -1, X_train.shape[-1]
).mean(axis=0)

feature_std = X_train.reshape(
    -1, X_train.shape[-1]
).std(axis=0)

feature_std[feature_std < 1e-6] = 1.0

X_train = (X_train - feature_mean) / feature_std
X_val = (X_val - feature_mean) / feature_std


# Convert to tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# GRU model

class TemporalGRU(nn.Module):

    def __init__(self):

        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        x = self.projection(x)

        output, hidden = self.gru(x)

        final_state = hidden[-1]

        return self.classifier(
            final_state
        ).squeeze(1)


model = TemporalGRU().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.001
)


# Training and validation

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

print("\nStarting GRU training...")

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )

    model.eval()

    with torch.no_grad():

        val_logits = model(X_val.to(DEVICE))

        val_loss = criterion(
            val_logits,
            y_val.to(DEVICE)
        ).item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Train Loss: {average_train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss
        }, MODEL_FILE)

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# Load best model

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

with torch.no_grad():

    probabilities = torch.sigmoid(
        model(X_val.to(DEVICE))
    ).cpu().numpy()


predictions = (
    probabilities >= 0.5
).astype(int)


metrics = {
    "accuracy": float(
        accuracy_score(y_val.numpy(), predictions)
    ),
    "precision": float(
        precision_score(
            y_val.numpy(), predictions,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_val.numpy(), predictions,
            zero_division=0
        )
    ),
    "f1": float(
        f1_score(
            y_val.numpy(), predictions,
            zero_division=0
        )
    ),
    "roc_auc": float(
        roc_auc_score(y_val.numpy(), probabilities)
    ),
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss
}


with open(METRICS_FILE, "w") as file:

    json.dump(metrics, file, indent=4)


prediction_data = pd.DataFrame({
    "video_name": val_names,
    "actual_label": y_val.numpy().astype(int),
    "predicted_label": predictions,
    "fake_probability": probabilities
})

prediction_data.to_csv(
    PREDICTIONS_FILE,
    index=False
)


print("\nTEMPORAL GRU TRAINING COMPLETED")

for metric, value in metrics.items():

    print(f"{metric}: {value}")

print("\nSaved model:", MODEL_FILE)
print("Saved metrics:", METRICS_FILE)
print("Saved predictions:", PREDICTIONS_FILE)
