
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = (
    PROJECT_ROOT / "datasets" / "metadata" /
    "ffpp_frame_metadata.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" /
    "efficientnet_b0_spatial_best.pth"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "datasets" / "metadata" /
    "efficientnet_temporal_features.npz"
)

IMAGE_SIZE = 224
FRAME_BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.classifier = nn.Identity()

model = model.to(DEVICE)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False


metadata = pd.read_csv(METADATA_FILE)

metadata = metadata[
    metadata["split"].isin(["train", "validation"])
].copy()

sequences = []
labels = []
video_names = []
splits = []

print("\nExtracting temporal features...")

with torch.no_grad():

    for _, video in tqdm(
        metadata.iterrows(),
        total=len(metadata)
    ):

        frame_folder = Path(video["output_folder"])

        frame_files = sorted(
            frame_folder.glob("*.png"),
            key=lambda path: int(
                path.stem.split("_")[1]
            )
        )

        if len(frame_files) != 10:
            print(
                "Skipping:",
                video["video_name"],
                "Frames:",
                len(frame_files)
            )
            continue

        video_features = []

        for start in range(
            0, len(frame_files), FRAME_BATCH_SIZE
        ):

            batch_files = frame_files[
                start:start + FRAME_BATCH_SIZE
            ]

            images = []

            for frame_path in batch_files:

                image = Image.open(
                    frame_path
                ).convert("RGB")

                images.append(transform(image))

            images = torch.stack(images).to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda"
            ):

                features = model.features(images)
                features = model.avgpool(features)
                features = torch.flatten(features, 1)

            video_features.extend(
                features.float().cpu().numpy()
            )

        sequences.append(
            np.array(video_features, dtype=np.float32)
        )

        labels.append(int(video["label"]))
        video_names.append(video["video_name"])
        splits.append(video["split"])


sequences = np.array(sequences, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)
video_names = np.array(video_names)
splits = np.array(splits)

np.savez_compressed(
    OUTPUT_FILE,
    sequences=sequences,
    labels=labels,
    video_names=video_names,
    splits=splits
)

print("\nTEMPORAL FEATURE EXTRACTION COMPLETED")
print("Videos:", len(sequences))
print("Sequence shape:", sequences.shape)
print("Training videos:", np.sum(splits == "train"))
print("Validation videos:", np.sum(splits == "validation"))
print("Test videos: 0")
print("Saved to:", OUTPUT_FILE)
