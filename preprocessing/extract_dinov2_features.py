
import re
import numpy as np
import pandas as pd
import torch
import timm

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from timm.data import resolve_model_data_config, create_transform
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = ROOT / "datasets" / "metadata" / "ffpp_frame_metadata.csv"
OUTPUT_FILE = ROOT / "datasets" / "metadata" / "dinov2_temporal_features.npz"

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
BATCH_SIZE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


print("\nLoading pretrained DINOv2...")

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0
)

model = model.to(DEVICE)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

print("Model loaded.")
print("Feature dimensions:", model.num_features)


data_config = resolve_model_data_config(model)

transform = create_transform(
    **data_config,
    is_training=False
)

print("Input size:", data_config["input_size"])


metadata = pd.read_csv(METADATA_FILE)

print("\nDataset videos:", len(metadata))
print(metadata["split"].value_counts())


class VideoFrameDataset(Dataset):

    def __init__(self, dataframe):
        self.samples = []

        for _, row in dataframe.iterrows():

            folder = Path(row["output_folder"])

            image_paths = [
                path for path in folder.iterdir()
                if path.suffix.lower() in [
                    ".jpg", ".jpeg", ".png", ".bmp"
                ]
            ]

            image_paths = sorted(
                image_paths,
                key=lambda path: int(
                    re.findall(r"\d+", path.stem)[-1]
                )
            )

            if len(image_paths) != 10:
                raise ValueError(
                    f"{row['video_name']} has {len(image_paths)} frames"
                )

            for frame_index, image_path in enumerate(image_paths):

                self.samples.append((
                    image_path,
                    row["video_name"],
                    frame_index,
                    int(row["label"])
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, video_name, frame_index, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        return image, video_name, frame_index, label


dataset = VideoFrameDataset(metadata)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTotal frames:", len(dataset))


features_by_video = {}

with torch.inference_mode():

    for images, video_names, frame_indices, labels in tqdm(loader):

        images = images.to(DEVICE)

        if DEVICE.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                features = model(images)
        else:
            features = model(images)

        features = features.float().cpu().numpy()

        for video_name, frame_index, feature in zip(
            video_names,
            frame_indices.tolist(),
            features
        ):

            if video_name not in features_by_video:
                features_by_video[video_name] = [None] * 10

            features_by_video[video_name][frame_index] = feature


all_features = []
all_labels = []
all_video_names = []
all_splits = []

for _, row in metadata.iterrows():

    video_name = row["video_name"]
    video_features = features_by_video[video_name]

    if any(feature is None for feature in video_features):
        raise ValueError(
            f"Missing extracted features for {video_name}"
        )

    all_features.append(np.stack(video_features))
    all_labels.append(int(row["label"]))
    all_video_names.append(video_name)
    all_splits.append(row["split"])


all_features = np.stack(all_features).astype(np.float32)

all_labels = np.array(all_labels, dtype=np.int64)
all_video_names = np.array(all_video_names, dtype=str)
all_splits = np.array(all_splits, dtype=str)


np.savez_compressed(
    OUTPUT_FILE,
    features=all_features,
    labels=all_labels,
    video_names=all_video_names,
    splits=all_splits
)


print("\nDINOV2 FEATURE EXTRACTION COMPLETED")
print("Feature shape:", all_features.shape)
print("Videos:", len(all_video_names))
print("Features per frame:", all_features.shape[2])
print("Output:", OUTPUT_FILE)
