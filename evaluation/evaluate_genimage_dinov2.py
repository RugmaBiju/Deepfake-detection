
import os
import numpy as np
import pandas as pd
import torch
import joblib
from PIL import Image
from tqdm import tqdm
import timm
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

BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv("datasets/metadata/genimage_metadata.csv")
test_df = df[df["split"] == "test"].reset_index(drop=True)

print("\nTest images:", len(test_df))
print("\nTest categories:")
print(test_df["category"].value_counts())


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

features = []
labels = []

print("\nExtracting test features...")

with torch.inference_mode():
    for images, batch_labels in tqdm(test_loader):
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

X_test = np.concatenate(features)
y_test = np.array(labels)

print("\nTest feature shape:", X_test.shape)

classifier = joblib.load(
     "models/genimage_dinov2_classifier.joblib"
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
    "experiments/genimage_dinov2_test_metrics.csv",
    index=False
)

print("\nTest metrics saved to:")
print("experiments/genimage_dinov2_test_metrics.csv")

data_config = timm.data.resolve_model_data_config(model)

transform = timm.data.create_transform(
    **data_config,
    is_training=False
)
