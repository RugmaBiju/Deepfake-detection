"""
Fits the signal-dependent Poisson-Gaussian noise model:
    Var(patch) ≈ a * mean(patch) + b

Real camera sensors follow this heteroscedastic relationship (a = photon
shot noise term, b = read noise / fixed pattern term, a > 0 expected).
Diffusion-generated images often violate this: variance can be too flat
(over-smoothed) or fail to scale with local intensity the way physical
photon noise does.

We estimate (a, b) via local patch statistics computed on smooth /
low-detail patches only (edges would inflate variance from real texture,
not noise).
"""
import numpy as np
import cv2


def _patch_grid_stats(gray: np.ndarray, patch_size: int = 8):
    h, w = gray.shape
    means, variances = [], []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            means.append(patch.mean())
            variances.append(patch.var())
    return np.array(means), np.array(variances)


def _flatness_mask(gray: np.ndarray, patch_size: int = 8, edge_thresh: float = 10.0):
    """Identify patches with low gradient energy (flat regions), where
    variance is attributable to sensor/generation noise rather than texture."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    h, w = gray.shape
    flat = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch_grad = grad_mag[y : y + patch_size, x : x + patch_size]
            flat.append(patch_grad.mean() < edge_thresh)
    return np.array(flat)


def fit_poisson_gaussian(image_bgr_uint8: np.ndarray, patch_size: int = 8) -> dict:
    gray = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32)

    means, variances = _patch_grid_stats(gray, patch_size)
    flat_mask = _flatness_mask(gray, patch_size)

    if flat_mask.sum() < 10:
        # Not enough flat patches to fit reliably (e.g. very busy image) —
        # fall back to using all patches, weighted lower in downstream use.
        flat_mask = np.ones_like(flat_mask, dtype=bool)

    x = means[flat_mask]
    y = variances[flat_mask]

    # Linear regression: y = a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), residuals_sum, rank, sv = np.linalg.lstsq(A, y, rcond=None)

    y_pred = a * x + b
    fit_residual = float(np.sqrt(np.mean((y - y_pred) ** 2)))  # RMSE of the fit

    return {
        "pg_a": float(a),
        "pg_b": float(b),
        "pg_fit_rmse": fit_residual,
        "pg_num_flat_patches": int(flat_mask.sum()),
    }
