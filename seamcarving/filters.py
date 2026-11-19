"""Linear filtering written from scratch. Owner: Ahsen."""

import numpy as np


def correlate2d(img: np.ndarray, kernel: np.ndarray, border: str = "reflect") -> np.ndarray:
    """Correlate a single-channel image with a kernel.

    Args:
        img: (H, W) array.
        kernel: 2-D array; both sides are odd.
        border: "reflect" (= cv2.BORDER_REFLECT_101) | "edge" (= cv2.BORDER_REPLICATE) | "zero".

    Returns:
        (H, W) float64 array.
    """
    raise NotImplementedError


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Return a square Gaussian kernel with an odd side length that sums to 1."""
    raise NotImplementedError
