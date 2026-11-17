"""
High-frequency forensic analysis.

We estimate high-frequency image content by subtracting
a Gaussian-smoothed version of the image from the original.

Formula:

I_low(x,y) = G_sigma * I(x,y)

H(x,y) = I(x,y) - I_low(x,y)

HF_energy_ratio =
    sum(H(x,y)^2) /
    [sum((I(x,y)-mean(I))^2) + epsilon]
"""

import cv2
import numpy as np


def high_frequency_residual(image_bgr_uint8, sigma=1.5):
    """
    Compute the high-frequency residual of an image.

    Parameters
    ----------
    image_bgr_uint8 : np.ndarray
        Input BGR uint8 image.

    sigma : float
        Gaussian smoothing strength.

    Returns
    -------
    residual : np.ndarray
        High-frequency residual as float32.
    """

    gray = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2GRAY)

    gray = gray.astype(np.float32) / 255.0

    # Gaussian smoothing removes high-frequency components
    low_frequency = cv2.GaussianBlur(
        gray,
        ksize=(0, 0),
        sigmaX=sigma
    )

    # High-frequency residual
    residual = gray - low_frequency

    return residual


def hf_energy_ratio(image_bgr_uint8, sigma=1.5):
    """
    Measure relative high-frequency energy.
    """

    gray = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0

    residual = high_frequency_residual(
        image_bgr_uint8,
        sigma=sigma
    )

    numerator = np.sum(residual ** 2)

    centered = gray - np.mean(gray)

    denominator = np.sum(centered ** 2) + 1e-12

    return float(numerator / denominator)


def hf_residual_statistics(image_bgr_uint8, sigma=1.5):
    """
    Extract high-frequency forensic features.
    """

    residual = high_frequency_residual(
        image_bgr_uint8,
        sigma=sigma
    )

    energy = float(np.mean(residual ** 2))

    std = float(np.std(residual))

    mean_abs = float(np.mean(np.abs(residual)))

    max_abs = float(np.max(np.abs(residual)))

    ratio = hf_energy_ratio(
        image_bgr_uint8,
        sigma=sigma
    )

    return {
        "hf_energy_ratio": ratio,
        "hf_residual_energy": energy,
        "hf_residual_std": std,
        "hf_residual_mean_abs": mean_abs,
        "hf_residual_max_abs": max_abs,
    }