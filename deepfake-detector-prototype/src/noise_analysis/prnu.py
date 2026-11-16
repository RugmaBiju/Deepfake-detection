"""
Reference-free PRNU-style camera-noise consistency features.

This does NOT estimate a true sensor-specific PRNU fingerprint because
the dataset does not provide camera/source groupings.

Instead, it measures whether the extracted noise residual has spatially
consistent structure across image blocks.
"""

import numpy as np

from src.noise_analysis.denoiser import residual_grayscale


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized correlation between two arrays."""
    a = a.astype(np.float64) - np.mean(a)
    b = b.astype(np.float64) - np.mean(b)

    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12

    return float(np.sum(a * b) / denom)


def block_consistency_score(
    residual_gray: np.ndarray,
    block_size: int = 32
) -> dict:
    """
    Reference-free spatial consistency analysis.

    The residual is divided into blocks. Adjacent blocks are compared
    after normalization. Camera-acquired noise should have more stable
    statistical structure across neighboring regions than highly
    synthetic residuals.

    Returns:
        prnu_block_mean
        prnu_block_std
        prnu_block_min
    """

    h, w = residual_gray.shape

    scores = []

    # Compare horizontally adjacent blocks
    for y in range(0, h - block_size + 1, block_size):

        for x in range(0, w - 2 * block_size + 1, block_size):

            block_a = residual_gray[
                y:y + block_size,
                x:x + block_size
            ]

            block_b = residual_gray[
                y:y + block_size,
                x + block_size:x + 2 * block_size
            ]

            scores.append(
                normalized_cross_correlation(block_a, block_b)
            )

    # Compare vertically adjacent blocks
    for y in range(0, h - 2 * block_size + 1, block_size):

        for x in range(0, w - block_size + 1, block_size):

            block_a = residual_gray[
                y:y + block_size,
                x:x + block_size
            ]

            block_b = residual_gray[
                y + block_size:y + 2 * block_size,
                x:x + block_size
            ]

            scores.append(
                normalized_cross_correlation(block_a, block_b)
            )

    if not scores:
        return {
            "prnu_block_mean": 0.0,
            "prnu_block_std": 0.0,
            "prnu_block_min": 0.0,
        }

    scores = np.asarray(scores, dtype=np.float64)

    return {
        "prnu_block_mean": float(np.mean(scores)),
        "prnu_block_std": float(np.std(scores)),
        "prnu_block_min": float(np.min(scores)),
    }


def prnu_features_single_image(
    image_bgr_uint8: np.ndarray
) -> dict:
    """Extract reference-free PRNU-style features."""
    residual = residual_grayscale(image_bgr_uint8)

    return block_consistency_score(residual)