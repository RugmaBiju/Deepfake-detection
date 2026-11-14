
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = PROJECT_ROOT / "datasets" / "metadata" / "ffpp_frame_metadata.csv"

OUTPUT_FILE = PROJECT_ROOT / "datasets" / "metadata" / "ffpp_forensic_features.csv"

metadata = pd.read_csv(METADATA_FILE)

records = []

for _, video in tqdm(metadata.iterrows(), total=len(metadata)):

    frame_folder = Path(video["output_folder"])

    frame_files = sorted(frame_folder.glob("*.png"))

    for frame_path in frame_files:

        image = cv2.imread(str(frame_path))

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_float = gray.astype(np.float32)

        # High-frequency energy
        fft = np.fft.fft2(gray_float)
        fft_shift = np.fft.fftshift(fft)

        power_spectrum = np.abs(fft_shift) ** 2

        height, width = gray.shape

        y, x = np.ogrid[:height, :width]

        center_y = height // 2
        center_x = width // 2

        distance = np.sqrt(
            (y - center_y) ** 2 +
            (x - center_x) ** 2
        )

        radius = min(height, width) / 2

        high_frequency_mask = distance > (0.25 * radius)

        total_energy = np.sum(power_spectrum) + 1e-8

        high_frequency_energy = (
            np.sum(power_spectrum[high_frequency_mask])
            / total_energy
        )

        # Noise residual variance
        smoothed = cv2.GaussianBlur(gray_float, (3, 3), 0)

        residual = gray_float - smoothed

        noise_residual_variance = np.var(residual)

        # Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        laplacian_variance = laplacian.var()

        records.append({
            "video_name": video["video_name"],
            "frame_name": frame_path.name,
            "group_id": video["group_id"],
            "split": video["split"],
            "label": video["label"],
            "high_frequency_energy": high_frequency_energy,
            "noise_residual_variance": noise_residual_variance,
            "laplacian_variance": laplacian_variance
        })

features = pd.DataFrame(records)

features.to_csv(OUTPUT_FILE, index=False)

print("\nFORENSIC FEATURE EXTRACTION COMPLETED")
print("Total frames:", len(features))
print("Features extracted:", len(features.columns) - 5)
print("Saved to:", OUTPUT_FILE)

print("\nFeature summary:")
print(features.groupby(["split", "label"])[
    [
        "high_frequency_energy",
        "noise_residual_variance",
        "laplacian_variance"
    ]
].mean())
