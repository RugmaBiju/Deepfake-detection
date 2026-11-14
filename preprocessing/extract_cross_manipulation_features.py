
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from timm.data import resolve_model_data_config, create_transform


ROOT = Path(__file__).resolve().parents[1]

METADATA_ROOT = ROOT / "datasets" / "metadata"

OUTPUT_ROOT = (
    METADATA_ROOT / "cross_manipulation_features"
)

B0_CHECKPOINT = ROOT / "models" / "efficientnet_b0_spatial_best.pth"

CATEGORIES = ["face2face", "faceswap", "neuraltextures"]

BATCH_SIZE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


b0_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


print("\nLoading EfficientNet-B0...")

b0_model = models.efficientnet_b0(weights=None)

b0_model.classifier[1] = nn.Linear(
    b0_model.classifier[1].in_features,
    2
)

checkpoint = torch.load(
    B0_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

b0_model.load_state_dict(checkpoint["model_state_dict"])

b0_model.classifier = nn.Identity()
b0_model = b0_model.to(DEVICE)
b0_model.eval()

for parameter in b0_model.parameters():
    parameter.requires_grad = False


print("\nLoading pretrained DINOv2...")

dino_model = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m",
    pretrained=True,
    num_classes=0
)

dino_model = dino_model.to(DEVICE)
dino_model.eval()

for parameter in dino_model.parameters():
    parameter.requires_grad = False

dino_config = resolve_model_data_config(dino_model)

dino_transform = create_transform(
    **dino_config,
    is_training=False
)

print("DINOv2 feature dimensions:", dino_model.num_features)


for category in CATEGORIES:

    print(f"\n{'=' * 50}")
    print("PROCESSING:", category.upper())
    print(f"{'=' * 50}")

    metadata_file = (
        METADATA_ROOT /
        f"cross_manipulation_{category}_trainval_metadata.csv"
    )

    metadata = pd.read_csv(metadata_file)

    b0_sequences = []
    dino_sequences = []

    labels = []
    video_names = []
    group_ids = []
    splits = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):

        frame_folder = Path(row["output_folder"])

        frame_files = sorted(
            frame_folder.glob("*.png"),
            key=lambda path: int(
                re.findall(r"\d+", path.stem)[-1]
            )
        )

        if len(frame_files) != 10:
            raise ValueError(
                f"{row['video_name']} has {len(frame_files)} frames"
            )

        b0_video_features = []
        dino_video_features = []

        for start in range(0, len(frame_files), BATCH_SIZE):

            batch_files = frame_files[start:start + BATCH_SIZE]

            b0_images = []
            dino_images = []

            for frame_path in batch_files:

                image = Image.open(frame_path).convert("RGB")

                b0_images.append(b0_transform(image))
                dino_images.append(dino_transform(image))

            b0_batch = torch.stack(b0_images).to(DEVICE)
            dino_batch = torch.stack(dino_images).to(DEVICE)

            with torch.inference_mode():

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=DEVICE.type == "cuda"
                ):

                    b0_features = b0_model.features(b0_batch)
                    b0_features = b0_model.avgpool(b0_features)
                    b0_features = torch.flatten(b0_features, 1)

                    dino_features = dino_model(dino_batch)

            b0_video_features.extend(
                b0_features.float().cpu().numpy()
            )

            dino_video_features.extend(
                dino_features.float().cpu().numpy()
            )

        b0_sequences.append(
            np.asarray(b0_video_features, dtype=np.float32)
        )

        dino_sequences.append(
            np.asarray(dino_video_features, dtype=np.float32)
        )

        labels.append(int(row["label"]))
        video_names.append(row["video_name"])
        group_ids.append(str(row["group_id"]))
        splits.append(row["split"])

    b0_sequences = np.asarray(b0_sequences, dtype=np.float32)
    dino_sequences = np.asarray(dino_sequences, dtype=np.float32)

    labels = np.asarray(labels, dtype=np.int64)
    video_names = np.asarray(video_names, dtype=str)
    group_ids = np.asarray(group_ids, dtype=str)
    splits = np.asarray(splits, dtype=str)

    b0_output = (
        OUTPUT_ROOT /
        f"{category}_b0_features_trainval.npz"
    )

    dino_output = (
        OUTPUT_ROOT /
        f"{category}_dinov2_features_trainval.npz"
    )

    np.savez_compressed(
        b0_output,
        sequences=b0_sequences,
        labels=labels,
        video_names=video_names,
        group_ids=group_ids,
        splits=splits
    )

    np.savez_compressed(
        dino_output,
        features=dino_sequences,
        labels=labels,
        video_names=video_names,
        group_ids=group_ids,
        splits=splits
    )

    print("\nFEATURE EXTRACTION RESULTS")
    print("Category:", category)
    print("Videos:", len(video_names))
    print("B0 feature shape:", b0_sequences.shape)
    print("DINOv2 feature shape:", dino_sequences.shape)
    print("Training videos:", np.sum(splits == "train"))
    print("Validation videos:", np.sum(splits == "validation"))
    print("B0 saved:", b0_output)
    print("DINOv2 saved:", dino_output)


print("\nALL TRAINING AND VALIDATION FEATURES EXTRACTED")
