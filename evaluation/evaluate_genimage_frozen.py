
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BATCH_SIZE = 32
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv("datasets/metadata/genimage_metadata.csv")
test_df = df[df["split"] == "test"].reset_index(drop=True)

print("\nTest images:", len(test_df))
print("\nTest categories:")
print(test_df["category"].value_counts())

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class GenImageTestDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return transform(image), int(row["label"])

test_loader = DataLoader(
    GenImageTestDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
model.classifier = nn.Identity()
model = model.to(device)
model.eval()

features = []
labels = []

print("\nExtracting test features...")

with torch.no_grad():
    for images, batch_labels in tqdm(test_loader):
        images = images.to(device)

        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):
            embeddings = model(images)

        features.append(embeddings.cpu().numpy())
        labels.extend(batch_labels.numpy())

X_test = np.concatenate(features)
y_test = np.array(labels)

print("\nTest feature shape:", X_test.shape)

classifier = joblib.load(
    "models/genimage_frozen_classifier.joblib"
)

predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions, zero_division=0),
    "f1": f1_score(y_test, predictions, zero_division=0),
    "roc_auc": roc_auc_score(y_test, probabilities)
}

print("\nFINAL GENIMAGE TEST RESULTS")

for name, value in metrics.items():
    print(name.upper() + ":", round(value, 4))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    target_names=["Real", "AI-generated"],
    zero_division=0
))

os.makedirs("experiments", exist_ok=True)

pd.DataFrame([metrics]).to_csv(
    "experiments/genimage_frozen_test_metrics.csv",
    index=False
)

print("\nTest metrics saved to:")
print("experiments/genimage_frozen_test_metrics.csv")
