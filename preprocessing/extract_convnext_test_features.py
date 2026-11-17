
import os
import glob
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

metadata_files = [
    "datasets/metadata/cross_manipulation_face2face_metadata.csv",
    "datasets/metadata/cross_manipulation_faceswap_metadata.csv",
    "datasets/metadata/cross_manipulation_neuraltextures_metadata.csv"
]

all_videos = []
seen_real = set()

for file in metadata_files:
    df = pd.read_csv(file)

    for _, row in df.iterrows():
        category = str(row["category"]).lower()
        video_name = str(row["video_name"])

        if category == "real":
            if video_name in seen_real:
                continue
            seen_real.add(video_name)

        all_videos.append({
            "video_name": video_name,
            "label": int(row["label"]),
            "category": category,
            "output_folder": str(row["output_folder"])
        })

print("\nUnique test videos:", len(all_videos))

for category in sorted(set(v["category"] for v in all_videos)):
    count = sum(v["category"] == category for v in all_videos)
    print(category, ":", count)

model = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    num_classes=0
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

features = []
labels = []
categories = []
video_names = []
frame_names = []

batch_size = 32

for video in tqdm(all_videos, desc="Extracting test features"):

    frame_paths = sorted(
        glob.glob(os.path.join(video["output_folder"], "frame_*.png"))
    )

    if len(frame_paths) != 10:
        raise ValueError(
            f"Expected 10 frames for {video['video_name']}, "
            f"found {len(frame_paths)}"
        )

    for start in range(0, len(frame_paths), batch_size):

        batch_paths = frame_paths[start:start + batch_size]

        images = []

        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            images.append(transform(image))

        batch = torch.stack(images).to(device)

        with torch.no_grad():
            batch_features = model(batch)

        features.extend(batch_features.cpu().numpy())

        for path in batch_paths:
            labels.append(video["label"])
            categories.append(video["category"])
            video_names.append(video["video_name"])
            frame_names.append(os.path.basename(path))

features = np.array(features, dtype=np.float32)

os.makedirs(
    "datasets/metadata/convnext_features",
    exist_ok=True
)

output_path = (
    "datasets/metadata/convnext_features/"
    "convnext_test_features.npz"
)

np.savez_compressed(
    output_path,
    features=features,
    labels=np.array(labels),
    categories=np.array(categories),
    video_names=np.array(video_names),
    frame_names=np.array(frame_names)
)

print("\nTEST FEATURE EXTRACTION COMPLETE")
print("Feature shape:", features.shape)
print("Total frames:", len(features))
print("Unique videos:", len(all_videos))
print("Saved:", output_path)
