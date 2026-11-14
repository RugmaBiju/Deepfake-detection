
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

metadata_path = Path("datasets/metadata/ffpp_frame_metadata.csv")

frame_output_path = Path("datasets/metadata/ffpp_hybrid_frame_features.csv")
video_output_path = Path("datasets/metadata/ffpp_hybrid_video_features.csv")

metadata = pd.read_csv(metadata_path)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

def extract_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0

    # Noise residual features
    smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
    residual = gray - smoothed

    residual_variance = float(np.var(residual))
    residual_mean_absolute = float(np.mean(np.abs(residual)))
    residual_energy = float(np.mean(residual ** 2))

    # Frequency-domain features
    centered = gray - np.mean(gray)

    fft = np.fft.fftshift(np.fft.fft2(centered))
    magnitude = np.abs(fft)
    power = magnitude ** 2

    height, width = gray.shape

    y, x = np.ogrid[:height, :width]
    center_y = height // 2
    center_x = width // 2

    radius = np.sqrt(
        ((y - center_y) / height) ** 2 +
        ((x - center_x) / width) ** 2
    )

    low_mask = radius < 0.10
    mid_mask = (radius >= 0.10) & (radius < 0.30)
    high_mask = radius >= 0.30

    total_energy = float(np.sum(power)) + 1e-10

    low_frequency_ratio = float(np.sum(power[low_mask]) / total_energy)
    mid_frequency_ratio = float(np.sum(power[mid_mask]) / total_energy)
    high_frequency_ratio = float(np.sum(power[high_mask]) / total_energy)

    spectral_centroid = float(
        np.sum(radius * power) / total_energy
    )

    return {
        "residual_variance": residual_variance,
        "residual_mean_absolute": residual_mean_absolute,
        "residual_energy": residual_energy,
        "low_frequency_ratio": low_frequency_ratio,
        "mid_frequency_ratio": mid_frequency_ratio,
        "high_frequency_ratio": high_frequency_ratio,
        "spectral_centroid": spectral_centroid
    }


frame_results = []

for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    folder = Path(row["output_folder"])

    image_paths = sorted(
        path for path in folder.rglob("*")
        if path.suffix.lower() in image_extensions
    )

    for frame_index, image_path in enumerate(image_paths):
        features = extract_features(image_path)

        if features is None:
            continue

        result = {
            "video_name": row["video_name"],
            "group_id": row["group_id"],
            "split": row["split"],
            "label": row["label"],
            "category": row["category"],
            "frame_index": frame_index,
            "image_name": image_path.name
        }

        result.update(features)
        frame_results.append(result)


frame_df = pd.DataFrame(frame_results)

frame_output_path.parent.mkdir(parents=True, exist_ok=True)
frame_df.to_csv(frame_output_path, index=False)

feature_columns = [
    "residual_variance",
    "residual_mean_absolute",
    "residual_energy",
    "low_frequency_ratio",
    "mid_frequency_ratio",
    "high_frequency_ratio",
    "spectral_centroid"
]

video_df = frame_df.groupby(
    ["video_name", "group_id", "split", "label", "category"]
)[feature_columns].agg(["mean", "std"])

video_df.columns = [
    f"{feature}_{stat}"
    for feature, stat in video_df.columns
]

video_df = video_df.reset_index()
video_df = video_df.fillna(0)

video_df.to_csv(video_output_path, index=False)

print("\nFORENSIC FEATURE EXTRACTION COMPLETED")
print("Frames processed:", len(frame_df))
print("Videos processed:", len(video_df))
print("Features per frame:", len(feature_columns))
print("Features per video:", len(video_df.columns) - 5)
print("Frame features saved to:", frame_output_path)
print("Video features saved to:", video_output_path)
