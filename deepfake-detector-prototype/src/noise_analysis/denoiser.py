"""
Extracts the noise residual W = I - Denoise(I) using wavelet shrinkage
(the standard approach in PRNU literature, e.g. Lukas/Fridrich/Goljan).

This residual is shared input for the PRNU, Poisson-Gaussian, and
high-frequency branches, so we compute it once per image/channel.
"""
import numpy as np
import pywt


def wavelet_denoise_channel(channel: np.ndarray, wavelet: str = "db8", level: int = 4) -> np.ndarray:
    """
    Denoise a single-channel float image via wavelet coefficient shrinkage.
    Uses a 4th-level decomposition with soft-thresholding on detail
    coefficients (VisuShrink-style, sigma estimated from finest-level HH band).
    """
    coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)
    cA = coeffs[0]
    detail_coeffs = coeffs[1:]

    # Estimate noise sigma from the finest-scale diagonal (HH) subband
    # via robust median absolute deviation estimator.
    finest_hh = detail_coeffs[-1][2]
    sigma = np.median(np.abs(finest_hh)) / 0.6745
    sigma = max(sigma, 1e-8)

    n = channel.size
    universal_thresh = sigma * np.sqrt(2 * np.log(n))

    def soft_threshold(x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

    new_details = []
    for cH, cV, cD in detail_coeffs:
        new_details.append(
            (
                soft_threshold(cH, universal_thresh),
                soft_threshold(cV, universal_thresh),
                soft_threshold(cD, universal_thresh),
            )
        )

    denoised = pywt.waverec2([cA] + new_details, wavelet=wavelet)

    # waverec2 can return an array 1px larger than input depending on padding;
    # crop back to original shape.
    denoised = denoised[: channel.shape[0], : channel.shape[1]]
    return denoised


def compute_residual(image_bgr_uint8: np.ndarray) -> np.ndarray:
    """
    image_bgr_uint8: HxWx3 uint8 image (as read by cv2.imread).
    Returns: HxWx3 float32 residual W = I - Denoise(I), per channel.
    """
    img = image_bgr_uint8.astype(np.float32)
    residual = np.zeros_like(img)

    for c in range(img.shape[2]):
        denoised = wavelet_denoise_channel(img[:, :, c])
        residual[:, :, c] = img[:, :, c] - denoised

    return residual


def residual_grayscale(image_bgr_uint8: np.ndarray) -> np.ndarray:
    """Convenience: single-channel residual, averaged across color channels."""
    residual = compute_residual(image_bgr_uint8)
    return residual.mean(axis=2)
