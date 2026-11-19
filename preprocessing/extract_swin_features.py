
import os
import torch
import timm
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model = timm.create_model(
    "swin_tiny_patch4_window7_224.ms_in1k",
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

metadata_files = [
    "datasets/metadata/cross_manipulation_face2face_trainval_metadata.csv",
    "datasets/metadata/cross_manipulation_faceswap_trainval_metadata.csv",
    "datasets/metadata/cross_manipulation_neuraltextures_trainval_metadata.csv"
]

all_features = []

for metadata_file in metadata_files:

    metadata = pd.read_csv(metadata_file)

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):

        folder = row["output_folder"]

        for image_name in sorted(os.listdir(folder)):

            image_path = os.path.join(folder, image_name)

            try:
                image = Image.open(image_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = model(image)

                features = features.squeeze(0).cpu().numpy()

                all_features.append({
                    "video_name": row["video_name"],
                    "group_id": row["group_id"],
                    "split": row["split"],
                    "label": row["label"],
                    "category": row["category"],
                    "image_name": image_name,
                    "features": features
                })

            except Exception as e:
                print("Error:", image_path, e)

os.makedirs("datasets/metadata/swin_features", exist_ok=True)

np.savez_compressed(
    "datasets/metadata/swin_features/swin_trainval_features.npz",
    video_name=np.array([x["video_name"] for x in all_features]),
    group_id=np.array([x["group_id"] for x in all_features]),
    split=np.array([x["split"] for x in all_features]),
    label=np.array([x["label"] for x in all_features]),
    category=np.array([x["category"] for x in all_features]),
    image_name=np.array([x["image_name"] for x in all_features]),
    features=np.stack([x["features"] for x in all_features])
)

print("\nFEATURE EXTRACTION COMPLETE")
print("Total frames:", len(all_features))
print("Feature shape:", np.stack([x["features"] for x in all_features]).shape)
print("Saved: datasets/metadata/swin_features/swin_trainval_features.npz")
