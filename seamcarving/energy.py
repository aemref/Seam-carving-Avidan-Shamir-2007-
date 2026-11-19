"""Energy map of an image (Avidan & Shamir 2007, Eq. 1).

    e1(I) = |dI/dx| + |dI/dy|

This is the gradient magnitude approximation M ~ |gx| + |gy| of INF605 week 5,
slide 56; ``norm="l2"`` gives the square-root version. Low energy marks pixels
that blend into their neighbourhood, which is where seams should pass.
"""

from __future__ import annotations

import numpy as np

from seamcarving.filters import (
    LAPLACIAN,
    ROBERTS_ANTI,
    ROBERTS_DIAG,
    SCHARR_COLS,
    SCHARR_ROWS,
    SOBEL_COLS,
    SOBEL_ROWS,
    correlate2d,
    gaussian_kernel,
)
from seamcarving.types import EnergyFn

METHODS = ("forward", "roberts", "sobel", "scharr", "laplacian")
COLORS = ("gray", "sum")
NORMS = ("l1", "l2")

# BGR weights for grey conversion (ITU-R BT.601, same as cv2.COLOR_BGR2GRAY).
_GRAY_WEIGHTS_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float64)

# (kernel along rows, kernel along columns) for the correlation-based operators.
_KERNEL_PAIRS = {
    "roberts": (ROBERTS_DIAG, ROBERTS_ANTI),
    "sobel": (SOBEL_ROWS, SOBEL_COLS),
    "scharr": (SCHARR_ROWS, SCHARR_COLS),
}

# Reflect padding avoids the fake edge that zero padding creates at the border.
_BORDER = "reflect"


def _check_options(method: str, color: str, blur_sigma: float, norm: str) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if color not in COLORS:
        raise ValueError(f"unknown color {color!r}; expected one of {COLORS}")
    if norm not in NORMS:
        raise ValueError(f"unknown norm {norm!r}; expected one of {NORMS}")
    if blur_sigma < 0:
        raise ValueError(f"blur_sigma must be >= 0, got {blur_sigma}")


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR (H, W, 3) or grey (H, W) image -> (H, W) float64 grey image."""
    f = np.asarray(img).astype(np.float64)
    if f.ndim == 2:
        return f
    if f.ndim == 3 and f.shape[2] == 3:
        return f @ _GRAY_WEIGHTS_BGR
    raise ValueError(f"expected (H, W) or (H, W, 3) image, got shape {f.shape}")


def _forward_differences(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward differences (week 5, slide 43); zero in the last row / column."""
    d_rows = np.zeros_like(f)
    d_cols = np.zeros_like(f)
    d_rows[:-1, :] = f[1:, :] - f[:-1, :]
    d_cols[:, :-1] = f[:, 1:] - f[:, :-1]
    return d_rows, d_cols


def _channel_energy(f: np.ndarray, method: str, blur_sigma: float, norm: str) -> np.ndarray:
    """Energy of one float64 (H, W) channel."""
    if blur_sigma > 0:
        f = correlate2d(f, gaussian_kernel(blur_sigma), border=_BORDER)

    if method == "laplacian":
        return np.abs(correlate2d(f, LAPLACIAN, border=_BORDER))

    if method == "forward":
        d_rows, d_cols = _forward_differences(f)
    else:
        k_rows, k_cols = _KERNEL_PAIRS[method]
        d_rows = correlate2d(f, k_rows, border=_BORDER)
        d_cols = correlate2d(f, k_cols, border=_BORDER)

    if norm == "l1":
        return np.abs(d_rows) + np.abs(d_cols)
    return np.sqrt(d_rows**2 + d_cols**2)


def compute_energy(
    img: np.ndarray,
    method: str = "sobel",
    color: str = "gray",
    blur_sigma: float = 0.0,
    norm: str = "l1",
) -> np.ndarray:
    """Energy map of an image.

    Args:
        img: (H, W) grey or (H, W, 3) BGR image, any numeric dtype. Not modified.
        method: one of METHODS. "laplacian" is |second derivative| and ignores ``norm``.
        color: "gray" converts to grey first; "sum" computes the energy of each
            channel separately and adds them. Both are equal for a grey input.
        blur_sigma: if > 0, Gaussian smoothing with this sigma before differentiating.
        norm: "l1" -> |d_rows| + |d_cols|, "l2" -> sqrt(d_rows^2 + d_cols^2).

    Returns:
        (H, W) float64 array, every value >= 0.
    """
    _check_options(method, color, blur_sigma, norm)
    # Convert before any subtraction: uint8 arithmetic wraps (200 - 250 = 206).
    f = np.asarray(img).astype(np.float64)
    if f.ndim not in (2, 3) or (f.ndim == 3 and f.shape[2] != 3):
        raise ValueError(f"expected (H, W) or (H, W, 3) image, got shape {f.shape}")

    if color == "gray" or f.ndim == 2:
        return _channel_energy(to_gray(f), method, blur_sigma, norm)

    energy = np.zeros(f.shape[:2], dtype=np.float64)
    for ch in range(f.shape[2]):
        energy += _channel_energy(f[:, :, ch], method, blur_sigma, norm)
    return energy


def make_energy_fn(**kwargs) -> EnergyFn:
    """Wrap ``compute_energy`` (with ``kwargs``) into the ``(img, carry)`` signature."""
    raise NotImplementedError
