
import os
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.amp import autocast

ROOT = os.getcwd()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRAME_DIR = os.path.join(
    ROOT, "datasets", "processed", "ffpp_frames"
)

METADATA_PATH = os.path.join(
    ROOT, "datasets", "metadata", "ffpp_frame_metadata.csv"
)

OUTPUT_PATH = os.path.join(
    ROOT, "datasets", "metadata", "xception_video_embeddings.csv"
)

FRAME_BATCH_SIZE = 4

metadata = pd.read_csv(METADATA_PATH)

# Use training and validation videos only.
# The test set remains untouched.
metadata = metadata[
    metadata["split"].isin(["train", "validation"])
].reset_index(drop=True)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

print("Loading pretrained Xception...")

model = timm.create_model(
    "legacy_xception.tf_in1k",
    pretrained=True,
    num_classes=0,
    global_pool="avg"
)

model = model.to(DEVICE)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

results = []

for _, row in tqdm(metadata.iterrows(), total=len(metadata)):

    folder = os.path.join(FRAME_DIR, row["output_folder"])

    frame_files = sorted([
        file for file in os.listdir(folder)
        if file.lower().endswith(".png")
    ])

    frame_features = []

    for start in range(0, len(frame_files), FRAME_BATCH_SIZE):

        batch_files = frame_files[
            start:start + FRAME_BATCH_SIZE
        ]

        images = []

        for file in batch_files:
            image_path = os.path.join(folder, file)

            image = Image.open(image_path).convert("RGB")
            images.append(transform(image))

        images = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            with autocast(
                "cuda",
                enabled=DEVICE.type == "cuda"
            ):
                features = model(images)

        frame_features.extend(
            features.float().cpu().numpy()
        )

        del images, features

    video_embedding = np.mean(
        np.array(frame_features),
        axis=0
    )

    result = {
        "video_name": row["video_name"],
        "split": row["split"],
        "label": int(row["label"])
    }

    for index, value in enumerate(video_embedding):
        result[f"feature_{index}"] = float(value)

    results.append(result)

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

pd.DataFrame(results).to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nXCEPTION FEATURE EXTRACTION COMPLETED")
print("Videos processed:", len(results))
print("Feature dimensions:", len(results[0]) - 3)
print("Saved to:", OUTPUT_PATH)
