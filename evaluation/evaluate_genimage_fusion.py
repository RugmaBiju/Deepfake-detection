
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

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

SEED = 42
B0_BATCH_SIZE = 32
DINO_BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


df = pd.read_csv("datasets/metadata/genimage_metadata.csv")

val_parts = []

for category, count in [
    ("Real", 750),
    ("DDPM", 375),
    ("DDIM", 375)
]:
    part = df[
        (df["split"] == "validation") &
        (df["category"] == category)
    ]

    val_parts.append(
        part.sample(n=count, random_state=SEED)
    )

val_df = pd.concat(val_parts).sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

test_df = df[
    df["split"] == "test"
].reset_index(drop=True)

print("\nValidation images:", len(val_df))
print("Test images:", len(test_df))


class GenImageDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        image = self.transform(image)

        return image, int(row["label"])


def extract_features(model, loader, description):

    features = []
    labels = []

    with torch.inference_mode():

        for images, batch_labels in tqdm(
            loader,
            desc=description
        ):

            images = images.to(device)

            if device.type == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    embeddings = model(images)

            else:
                embeddings = model(images)

            features.append(
                embeddings.float().cpu().numpy()
            )

            labels.extend(batch_labels.numpy())

    return (
        np.concatenate(features),
        np.array(labels)
    )


def calculate_metrics(y_true, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        )
    }


# ==========================================
# EFFICIENTNET-B0 FEATURE EXTRACTION
# ==========================================

print("\n" + "=" * 55)
print("EXTRACTING EFFICIENTNET-B0 FEATURES")
print("=" * 55)

b0_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

b0_val_loader = DataLoader(
    GenImageDataset(val_df, b0_transform),
    batch_size=B0_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

b0_test_loader = DataLoader(
    GenImageDataset(test_df, b0_transform),
    batch_size=B0_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

b0_weights = models.EfficientNet_B0_Weights.DEFAULT

b0_model = models.efficientnet_b0(
    weights=b0_weights
)

b0_model.classifier = nn.Identity()
b0_model = b0_model.to(device)
b0_model.eval()

X_b0_val, y_val = extract_features(
    b0_model,
    b0_val_loader,
    "B0 validation"
)

X_b0_test, y_test = extract_features(
    b0_model,
    b0_test_loader,
    "B0 test"
)

print("B0 validation shape:", X_b0_val.shape)
print("B0 test shape:", X_b0_test.shape)

del b0_model
if device.type == "cuda":
    torch.cuda.empty_cache()


# ==========================================
# DINOV2 FEATURE EXTRACTION
# ==========================================

print("\n" + "=" * 55)
print("EXTRACTING DINOV2 FEATURES")
print("=" * 55)

DINO_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"

dino_model = timm.create_model(
    DINO_MODEL_NAME,
    pretrained=True,
    num_classes=0
)

dino_model = dino_model.to(device)
dino_model.eval()

for parameter in dino_model.parameters():
    parameter.requires_grad = False

dino_config = timm.data.resolve_model_data_config(
    dino_model
)

dino_transform = timm.data.create_transform(
    **dino_config,
    is_training=False
)

dino_val_loader = DataLoader(
    GenImageDataset(val_df, dino_transform),
    batch_size=DINO_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

dino_test_loader = DataLoader(
    GenImageDataset(test_df, dino_transform),
    batch_size=DINO_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

X_dino_val, y_dino_val = extract_features(
    dino_model,
    dino_val_loader,
    "DINOv2 validation"
)

X_dino_test, y_dino_test = extract_features(
    dino_model,
    dino_test_loader,
    "DINOv2 test"
)

print("DINOv2 validation shape:", X_dino_val.shape)
print("DINOv2 test shape:", X_dino_test.shape)

del dino_model
if device.type == "cuda":
    torch.cuda.empty_cache()


# ==========================================
# LOAD TRAINED CLASSIFIERS
# ==========================================

print("\nLoading trained classifiers...")

b0_classifier = joblib.load(
    "models/genimage_frozen_classifier.joblib"
)

dino_classifier = joblib.load(
    "models/genimage_dinov2_classifier.joblib"
)

if not np.array_equal(y_val, y_dino_val):
    raise ValueError("Validation labels do not match")

if not np.array_equal(y_test, y_dino_test):
    raise ValueError("Test labels do not match")


# ==========================================
# VALIDATION PROBABILITIES
# ==========================================

print("\nCalculating validation probabilities...")

b0_val_prob = b0_classifier.predict_proba(
    X_b0_val
)[:, 1]

dino_val_prob = dino_classifier.predict_proba(
    X_dino_val
)[:, 1]


# ==========================================
# FUSION WEIGHT SELECTION
# ==========================================

print("\n" + "=" * 55)
print("VALIDATION FUSION WEIGHT SEARCH")
print("=" * 55)

weight_results = []

for b0_weight in np.arange(0.0, 1.01, 0.05):

    dino_weight = 1.0 - b0_weight

    fused_prob = (
        b0_weight * b0_val_prob +
        dino_weight * dino_val_prob
    )

    metrics = calculate_metrics(
        y_val,
        fused_prob
    )

    result = {
        "b0_weight": round(float(b0_weight), 2),
        "dino_weight": round(float(dino_weight), 2),
        **metrics
    }

    weight_results.append(result)

    print(
        "B0:", round(float(b0_weight), 2),
        "| DINO:", round(float(dino_weight), 2),
        "| Accuracy:", round(metrics["accuracy"], 4),
        "| F1:", round(metrics["f1"], 4),
        "| AUC:", round(metrics["roc_auc"], 4)
    )


weight_df = pd.DataFrame(weight_results)

best_result = max(
    weight_results,
    key=lambda result: (
        result["roc_auc"],
        result["f1"],
        result["accuracy"]
    )
)

best_b0_weight = best_result["b0_weight"]
best_dino_weight = best_result["dino_weight"]

print("\nBEST VALIDATION WEIGHTS")
print("B0 weight:", best_b0_weight)
print("DINOv2 weight:", best_dino_weight)
print("Validation metrics:", best_result)


# ==========================================
# TEST EVALUATION
# ==========================================

print("\n" + "=" * 55)
print("FINAL TEST EVALUATION")
print("=" * 55)

b0_test_prob = b0_classifier.predict_proba(
    X_b0_test
)[:, 1]

dino_test_prob = dino_classifier.predict_proba(
    X_dino_test
)[:, 1]

fused_test_prob = (
    best_b0_weight * b0_test_prob +
    best_dino_weight * dino_test_prob
)

test_results = []

for model_name, probabilities in [
    ("EfficientNet-B0", b0_test_prob),
    ("DINOv2", dino_test_prob),
    ("B0_DINOv2_Fusion", fused_test_prob)
]:

    metrics = calculate_metrics(
        y_test,
        probabilities
    )

    print("\nMODEL:", model_name)

    for name, value in metrics.items():
        print(name.upper() + ":", round(value, 4))

    test_results.append({
        "model": model_name,
        "b0_weight": (
            best_b0_weight
            if model_name == "B0_DINOv2_Fusion"
            else None
        ),
        "dino_weight": (
            best_dino_weight
            if model_name == "B0_DINOv2_Fusion"
            else None
        ),
        **metrics
    })


fused_predictions = (
    fused_test_prob >= 0.5
).astype(int)

print("\nFUSION CONFUSION MATRIX")
print(confusion_matrix(y_test, fused_predictions))

print("\nFUSION CLASSIFICATION REPORT")
print(classification_report(
    y_test,
    fused_predictions,
    target_names=["Real", "AI-generated"],
    zero_division=0
))


# ==========================================
# SAVE RESULTS
# ==========================================

os.makedirs("experiments", exist_ok=True)

weight_df.to_csv(
    "experiments/genimage_fusion_validation_search.csv",
    index=False
)

pd.DataFrame(test_results).to_csv(
    "experiments/genimage_fusion_test_metrics.csv",
    index=False
)

with open(
    "experiments/genimage_fusion_selected_weights.json",
    "w"
) as file:

    json.dump(
        {
            "b0_weight": best_b0_weight,
            "dino_weight": best_dino_weight,
            "selection_metric": "validation ROC-AUC",
            "validation_metrics": best_result
        },
        file,
        indent=4
    )


print("\nFUSION EXPERIMENT COMPLETED")

print(
    "Validation search:",
    "experiments/genimage_fusion_validation_search.csv"
)

print(
    "Test metrics:",
    "experiments/genimage_fusion_test_metrics.csv"
)

print(
    "Selected weights:",
    "experiments/genimage_fusion_selected_weights.json"
)
