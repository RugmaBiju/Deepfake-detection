
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import timm
import streamlit as st

from PIL import Image
from torchvision import models, transforms


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B0_CLASSIFIER_PATH = os.path.join(
    BASE_DIR, "models", "genimage_frozen_classifier_expanded.joblib"
)

DINO_CLASSIFIER_PATH = os.path.join(
    BASE_DIR, "models", "genimage_dinov2_classifier_expanded.joblib"
)


@st.cache_resource
def load_models():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b0_model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    b0_model.classifier = nn.Identity()
    b0_model.to(device)
    b0_model.eval()

    dino_model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0
    )
    dino_model.to(device)
    dino_model.eval()

    b0_classifier = joblib.load(B0_CLASSIFIER_PATH)
    dino_classifier = joblib.load(DINO_CLASSIFIER_PATH)

    b0_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    dino_config = timm.data.resolve_model_data_config(dino_model)
    dino_transform = timm.data.create_transform(
        **dino_config,
        is_training=False
    )

    return (
        device,
        b0_model,
        dino_model,
        b0_classifier,
        dino_classifier,
        b0_transform,
        dino_transform
    )


def predict_image(image):

    (
        device,
        b0_model,
        dino_model,
        b0_classifier,
        dino_classifier,
        b0_transform,
        dino_transform
    ) = load_models()

    image = image.convert("RGB")

    b0_input = b0_transform(image).unsqueeze(0).to(device)
    dino_input = dino_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():

        b0_features = b0_model(b0_input)
        dino_features = dino_model(dino_input)

    b0_features = b0_features.float().cpu().numpy()
    dino_features = dino_features.float().cpu().numpy()

    b0_probability = float(
        b0_classifier.predict_proba(b0_features)[0][1]
    )

    dino_probability = float(
        dino_classifier.predict_proba(dino_features)[0][1]
    )

    fusion_probability = (
        0.20 * b0_probability +
        0.80 * dino_probability
    )

    prediction = (
        "AI-Generated"
        if fusion_probability >= 0.5
        else "Real"
    )

    return {
        "prediction": prediction,
        "b0_probability": b0_probability,
        "dino_probability": dino_probability,
        "fusion_probability": fusion_probability,
        "device": str(device)
    }
