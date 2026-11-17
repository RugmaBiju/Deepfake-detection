"""
TruthLens feature extraction.

Combines:
1. Reference-free PRNU-style residual consistency
2. Poisson-Gaussian noise statistics
3. High-frequency residual statistics

Run:

python -m src.features.feature_builder \
    --labels data/labels.csv \
    --out data/features_new.csv \
    --limit 10
"""

import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.face_detector import FaceCropper
from src.noise_analysis.denoiser import residual_grayscale
from src.noise_analysis.prnu import block_consistency_score
from src.noise_analysis.poisson_gaussian import fit_poisson_gaussian
from src.noise_analysis.high_freq import hf_residual_statistics


def build_features_for_image(image_bgr: np.ndarray) -> dict:
    """
    Run all forensic feature branches on one cropped face.
    """

    # ---------------------------------------
    # 1. PRNU-style residual features
    # ---------------------------------------

    gray_residual = residual_grayscale(image_bgr)

    features = {}

    features.update(
        block_consistency_score(gray_residual)
    )

    # ---------------------------------------
    # 2. Poisson-Gaussian features
    # ---------------------------------------

    features.update(
        fit_poisson_gaussian(image_bgr)
    )

    # ---------------------------------------
    # 3. High-frequency residual features
    # ---------------------------------------

    features.update(
        hf_residual_statistics(image_bgr)
    )

    return features


def process_dataset(
    labels_csv: str,
    out_csv: str,
    limit: int | None = None,
):

    df = pd.read_csv(labels_csv)

    if limit is not None:
        df = df.sample(
            n=min(limit, len(df)),
            random_state=42,
        ).reset_index(drop=True)

    cropper = FaceCropper(
        output_size=256,
        margin=0.35,
    )

    rows = []
    skipped = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting features",
    ):

        image_path = row["filepath"]

        # ---------------------------------------
        # Read image
        # ---------------------------------------

        img = cv2.imread(image_path)

        if img is None:
            skipped += 1
            continue

        # ---------------------------------------
        # Detect + crop face
        # ---------------------------------------

        face = cropper.crop(img)

        if face is None:
            skipped += 1
            continue

        # ---------------------------------------
        # Extract features
        # ---------------------------------------

        try:

            feats = build_features_for_image(face)

        except Exception as e:

            print(
                f"\nFeature extraction failed for: "
                f"{image_path}"
            )

            print(f"Error: {e}")

            skipped += 1
            continue

        # ---------------------------------------
        # Add metadata
        # ---------------------------------------

        feats["filepath"] = image_path
        feats["label"] = row["label"]
        feats["generator"] = row["generator"]

        rows.append(feats)

    cropper.close()

    # ---------------------------------------
    # Save
    # ---------------------------------------

    out_df = pd.DataFrame(rows)

    out_df.to_csv(
        out_csv,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Input images : {len(df)}")
    print(f"Output rows  : {len(out_df)}")
    print(f"Skipped      : {skipped}")
    print(f"Output file  : {out_csv}")

    print("\nExtracted features:")

    feature_columns = [
        c
        for c in out_df.columns
        if c not in [
            "filepath",
            "label",
            "generator",
        ]
    ]

    for feature in feature_columns:
        print(f"  {feature}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        type=str,
        default="data/labels.csv",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="data/features.csv",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process",
    )

    args = parser.parse_args()

    process_dataset(
        args.labels,
        args.out,
        args.limit,
    )