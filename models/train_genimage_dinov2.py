
import os
import copy
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 42
BATCH_SIZE = 8
IMAGE_SIZE = 224

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv("datasets/metadata/genimage_metadata.csv")

train_parts = []
val_parts = []

for category, count in [("Real", 3000), ("DDPM", 1500), ("DDIM", 1500)]:
    part = df[(df["split"] == "train") & (df["category"] == category)]
    train_parts.append(part.sample(n=count, random_state=SEED))

for category, count in [("Real", 750), ("DDPM", 375), ("DDIM", 375)]:
    part = df[(df["split"] == "validation") & (df["category"] == category)]
    val_parts.append(part.sample(n=count, random_state=SEED))

train_df = pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df = pd.concat(val_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))


class GenImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = transform(image)
        label = int(row["label"])
        return image, label

train_loader = DataLoader(
    GenImageDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    GenImageDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0
)

model = model.to(device)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

data_config = timm.data.resolve_model_data_config(model)

transform = timm.data.create_transform(
    **data_config,
    is_training=False
)

print("DINOv2 model:", MODEL_NAME)
print("Feature dimensions:", model.num_features)
print("Input size:", data_config["input_size"])

def extract_features(loader):
    features = []
    labels = []

    with torch.inference_mode():
        for images, batch_labels in tqdm(loader):
            images = images.to(device)

            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    embeddings = model(images)
            else:
                embeddings = model(images)

            features.append(embeddings.cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.concatenate(features), np.array(labels)

print("\nExtracting training features...")
X_train, y_train = extract_features(train_loader)

print("\nExtracting validation features...")
X_val, y_val = extract_features(val_loader)

print("Training feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)

classifier = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED
    )
)

print("\nTraining Logistic Regression...")
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_val)
probabilities = classifier.predict_proba(X_val)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_val, predictions),
    "precision": precision_score(y_val, predictions, zero_division=0),
    "recall": recall_score(y_val, predictions, zero_division=0),
    "f1": f1_score(y_val, predictions, zero_division=0),
    "roc_auc": roc_auc_score(y_val, probabilities)
}

print("\nFROZEN DINOV2 BASELINE RESULTS")
for name, value in metrics.items():
    print(name.upper() + ":", round(value, 4))

os.makedirs("models", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

joblib.dump(classifier, "models/genimage_dinov2_classifier.joblib")

pd.DataFrame([metrics]).to_csv(
    "experiments/genimage_dinov2_metrics.csv", index=False
)

print("\nMODEL SAVED: models/genimage_dinov2_classifier.joblib")
print("METRICS SAVED: experiments/genimage_dinov2_metrics.csv")