"""Linear spatial filtering written with NumPy only.

The filtering is a correlation ("sum of products", INF605 week 5, slides 5-12):
each output pixel is the weighted sum of its neighbourhood, weights taken from
the kernel without flipping it. Derivative kernels used by the energy map live
here as module constants.
"""

from __future__ import annotations

import numpy as np

# np.pad modes for each supported border. "reflect" mirrors without repeating
# the edge pixel, which is cv2.BORDER_REFLECT_101 (the OpenCV default).
_PAD_MODES = {
    "reflect": "reflect",  # cv2.BORDER_REFLECT_101
    "edge": "edge",  # cv2.BORDER_REPLICATE
    "zero": "constant",  # cv2.BORDER_CONSTANT with value 0
}


def correlate2d(img: np.ndarray, kernel: np.ndarray, border: str = "reflect") -> np.ndarray:
    """Correlate a 2-D image with a kernel.

    Args:
        img: (H, W) array of any numeric dtype. Not modified.
        kernel: (kh, kw) array; kh and kw must be odd. The anchor is the centre.
        border: "reflect" (= cv2.BORDER_REFLECT_101), "edge" (= BORDER_REPLICATE)
            or "zero" (= BORDER_CONSTANT, value 0).

    Returns:
        (H, W) float64 array, ``out[r, c] = sum_ij kernel[i, j] * pad[r + i, c + j]``.

    The loop runs over kernel elements only; each step adds one shifted slice of
    the padded image, so the cost is kh * kw vectorised array operations.
    """
    img = np.asarray(img)
    kernel = np.asarray(kernel, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"img must be 2-D (H, W), got shape {img.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2-D, got shape {kernel.shape}")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"kernel sides must be odd, got {kernel.shape}")
    if border not in _PAD_MODES:
        raise ValueError(f"unknown border {border!r}; expected one of {tuple(_PAD_MODES)}")

    src = img.astype(np.float64)
    h, w = src.shape
    pr, pc = kh // 2, kw // 2
    padded = np.pad(src, ((pr, pr), (pc, pc)), mode=_PAD_MODES[border])

    out = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight != 0.0:
                out += weight * padded[i : i + h, j : j + w]
    return out


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Return a square Gaussian kernel with an odd side length that sums to 1."""
    raise NotImplementedError
