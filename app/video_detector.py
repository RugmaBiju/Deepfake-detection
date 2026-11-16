
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from timm.data import resolve_model_data_config, create_transform


ROOT = Path(__file__).resolve().parents[1]

FRAMES_PER_VIDEO = 10
IMAGE_SIZE = 224

B0_CHECKPOINT = ROOT / "models" / "efficientnet_b0_spatial_best.pth"
B0_CLASSIFIER = ROOT / "models" / "face_hybrid" / "visual_only_LogisticRegression.joblib"
DINO_CLASSIFIER = ROOT / "models" / "dinov2_logistic_regression.joblib"

DINO_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"

B0_WEIGHT = 0.35
DINO_WEIGHT = 0.65

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_video_models():

    b0 = models.efficientnet_b0(weights=None)
    b0.classifier[1] = nn.Linear(
        b0.classifier[1].in_features, 2
    )

    checkpoint = torch.load(
        B0_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    b0.load_state_dict(checkpoint["model_state_dict"])
    b0.classifier = nn.Identity()
    b0 = b0.to(DEVICE)
    b0.eval()

    dino = timm.create_model(
        DINO_MODEL_NAME,
        pretrained=True,
        num_classes=0
    )

    dino_data_config = resolve_model_data_config(dino)
    dino_transform = create_transform(
        **dino_data_config,
        is_training=False
    )

    dino = dino.to(DEVICE)
    dino.eval()

    b0_classifier = joblib.load(B0_CLASSIFIER)
    dino_classifier = joblib.load(DINO_CLASSIFIER)

    b0_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return (
        b0,
        dino,
        b0_classifier,
        dino_classifier,
        b0_transform,
        dino_transform
    )


def extract_video_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < FRAMES_PER_VIDEO:
        cap.release()
        raise ValueError(
            "Video must contain at least 10 frames."
        )

    frame_indices = [
        int(x) for x in
        np.linspace(
            0,
            total_frames - 1,
            FRAMES_PER_VIDEO
        )
    ]

    frames = []

    for frame_number in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            cap.release()
            raise ValueError(
                f"Could not read video frame {frame_number}."
            )

        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(Image.fromarray(frame))

    cap.release()

    if len(frames) != FRAMES_PER_VIDEO:
        raise ValueError("Could not extract all 10 frames.")

    return frames


def predict_video(uploaded_file):

    temp_path = None

    try:
        suffix = Path(uploaded_file.name).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        frames = extract_video_frames(temp_path)

        (
            b0,
            dino,
            b0_classifier,
            dino_classifier,
            b0_transform,
            dino_transform
        ) = load_video_models()

        b0_features = []
        dino_features = []

        with torch.no_grad():

            for frame in frames:

                b0_input = b0_transform(frame).unsqueeze(0)
                b0_input = b0_input.to(DEVICE)

                b0_feature = b0.features(b0_input)
                b0_feature = b0.avgpool(b0_feature)
                b0_feature = torch.flatten(b0_feature, 1)

                b0_features.append(
                    b0_feature.cpu().numpy()[0]
                )

                dino_input = dino_transform(frame).unsqueeze(0)
                dino_input = dino_input.to(DEVICE)

                dino_feature = dino(dino_input)

                dino_features.append(
                    dino_feature.cpu().numpy()[0]
                )

        b0_video_feature = np.mean(
            np.array(b0_features),
            axis=0
        ).reshape(1, -1)

        dino_video_feature = np.mean(
            np.array(dino_features),
            axis=0
        ).reshape(1, -1)

        b0_probability = float(
            b0_classifier.predict_proba(
                b0_video_feature
            )[0, 1]
        )

        dino_probability = float(
            dino_classifier.predict_proba(
                dino_video_feature
            )[0, 1]
        )

        fusion_probability = (
            B0_WEIGHT * b0_probability
            + DINO_WEIGHT * dino_probability
        )

        prediction = (
            "Manipulated"
            if fusion_probability >= 0.5
            else "Authentic"
        )

        return {
            "prediction": prediction,
            "b0_probability": b0_probability,
            "dino_probability": dino_probability,
            "fusion_probability": fusion_probability,
            "frames_analyzed": len(frames),
            "device": str(DEVICE)
        }

    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()
