
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

metadata_path = Path("datasets/metadata/ffpp_frame_metadata.csv")

frame_output = Path("datasets/metadata/ffpp_face_forensic_frame_features.csv")
video_output = Path("datasets/metadata/ffpp_face_forensic_video_features.csv")

metadata = pd.read_csv(metadata_path)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load Haar face detector")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_face_features(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # Select the largest detected face
    x, y, width, height = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Add a small margin around the face
    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(gray_image.shape[1], x + width + margin_x)
    y2 = min(gray_image.shape[0], y + height + margin_y)

    face = gray_image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    # Resize to a consistent size
    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float32) / 255.0

    # Noise residual features
    smoothed = cv2.GaussianBlur(face, (5, 5), 1.0)
    residual = face - smoothed

    residual_variance = float(np.var(residual))
    residual_mean_absolute = float(np.mean(np.abs(residual)))
    residual_energy = float(np.mean(residual ** 2))

    # FFT frequency features
    centered = face - np.mean(face)

    fft = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(fft) ** 2

    height, width = face.shape

    yy, xx = np.ogrid[:height, :width]

    radius = np.sqrt(
        ((yy - height // 2) / height) ** 2 +
        ((xx - width // 2) / width) ** 2
    )

    low_mask = radius < 0.10
    mid_mask = (radius >= 0.10) & (radius < 0.30)
    high_mask = radius >= 0.30

    total_energy = float(np.sum(power)) + 1e-10

    low_ratio = float(np.sum(power[low_mask]) / total_energy)
    mid_ratio = float(np.sum(power[mid_mask]) / total_energy)
    high_ratio = float(np.sum(power[high_mask]) / total_energy)

    spectral_centroid = float(
        np.sum(radius * power) / total_energy
    )

    return {
        "face_area_ratio": float((x2 - x1) * (y2 - y1) / (image.shape[0] * image.shape[1])),
        "residual_variance": residual_variance,
        "residual_mean_absolute": residual_mean_absolute,
        "residual_energy": residual_energy,
        "low_frequency_ratio": low_ratio,
        "mid_frequency_ratio": mid_ratio,
        "high_frequency_ratio": high_ratio,
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

        features = extract_face_features(image_path)

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

if frame_df.empty:
    raise RuntimeError("No faces detected in the dataset")

frame_df.to_csv(frame_output, index=False)

feature_columns = [
    "face_area_ratio",
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

video_df = video_df.reset_index().fillna(0)

video_df.to_csv(video_output, index=False)

print("\nFACE-FOCUSED FORENSIC EXTRACTION COMPLETED")
print("Frames with detected faces:", len(frame_df))
print("Videos with detected faces:", len(video_df))
print("Face detection rate:", round(len(frame_df) / 2000 * 100, 2), "%")
print("Features per frame:", len(feature_columns))
print("Features per video:", len(video_columns := video_df.columns) - 5)
print("Frame features saved to:", frame_output)
print("Video features saved to:", video_output)
