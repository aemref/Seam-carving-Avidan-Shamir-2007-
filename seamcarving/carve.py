"""Seam removal and shrinking (paper Sec. 4.1). Owner: Mirza.

Images are indexed ``img[r, c]``. A vertical seam is an ``(H,)`` integer array
where ``seam[r]`` is the column removed in row ``r``. Horizontal shrinking is
done by transposing the inputs, shrinking the width and transposing back.
"""

import operator
from typing import NamedTuple

import numpy as np

from seamcarving.types import EnergyFn, SeamFn


class ShrinkResult(NamedTuple):
    """Result of removing k seams.

    For ``shrink_height`` the image is (H-k, W[, C]) and ``removed`` has k True per column.
    """

    image: np.ndarray  # (H, W-k[, C]), same dtype as the input
    carry: tuple  # the carry arrays, cut with the same seams
    removed: np.ndarray  # bool (H, W): removed pixels in ORIGINAL coordinates; k True per row
    order: np.ndarray  # int32 (H, W): index 0..k-1 of the seam that removed the pixel; -1 if kept


def _check_seam(seam: np.ndarray, height: int, width: int) -> np.ndarray:
    """Validate a vertical seam for an (height, width) array; return it as int64 (H,)."""
    seam = np.asarray(seam)
    if seam.shape != (height,):
        raise ValueError(f"seam must have shape ({height},), got {seam.shape}")
    if not np.issubdtype(seam.dtype, np.integer):
        raise ValueError(f"seam must have an integer dtype, got {seam.dtype}")
    if height > 0 and (seam.min() < 0 or seam.max() >= width):
        raise ValueError(
            f"seam columns must lie in [0, {width}), got [{seam.min()}, {seam.max()}]"
        )
    return seam.astype(np.int64, copy=False)


def remove_vertical_seam(arr: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Remove one vertical seam: (H, W[, C]) -> (H, W-1[, C]).

    arr:  (H, W) or (H, W, C) array of any dtype (not modified).
    seam: (H,) integer array, ``seam[r]`` in [0, W).
    Returns a new (H, W-1[, C]) array with the same dtype.
    Raises ValueError if ``arr`` has fewer than 2 dims or the seam is invalid.
    """
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(f"arr must have at least 2 dimensions, got {arr.ndim}")
    h, w = arr.shape[:2]
    seam = _check_seam(seam, h, w)
    keep = np.ones((h, w), dtype=bool)
    keep[np.arange(h), seam] = False
    return arr[keep].reshape((h, w - 1) + arr.shape[2:])


def _check_inputs(img: np.ndarray, k: int, carry: tuple) -> tuple:
    """Validate shrink inputs; return (img, k, carry) with arrays and k normalised."""
    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ValueError(f"img must have shape (H, W) or (H, W, C), got {img.shape}")
    k = operator.index(k)
    h, w = img.shape[:2]
    if not 0 <= k < w:
        raise ValueError(f"k must satisfy 0 <= k < W={w}, got {k}")
    carry = tuple(np.asarray(a) for a in carry)
    for i, a in enumerate(carry):
        if a.ndim < 2 or a.shape[:2] != (h, w):
            raise ValueError(f"carry[{i}] must start with shape ({h}, {w}), got {a.shape}")
    return img, k, carry


def shrink_width(
    img: np.ndarray,
    k: int,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
    carry: tuple = (),
) -> ShrinkResult:
    """Remove k vertical seams.

    Each step: ``energy_fn(img, carry)`` -> ``seam_fn(energy)`` -> the image and the carry arrays
    are cut with the same seam. ``energy_fn`` None means ``compute_energy(img)``; ``seam_fn`` None
    means ``find_vertical_seam``.

    img:       (H, W) or (H, W, C) image (not modified).
    k:         number of seams, 0 <= k < W. The energy is recomputed at every step.
    energy_fn: (img, carry) -> (H, W) float64.
    seam_fn:   energy (H, W) -> seam (H,) int64.
    carry:     arrays with leading shape (H, W) that are cut with the same seams.
    Returns a ShrinkResult with image (H, W-k[, C]) and removed/order (H, W).
    """
    raise NotImplementedError


def shrink_height(
    img: np.ndarray,
    k: int,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
    carry: tuple = (),
) -> ShrinkResult:
    """Remove k horizontal seams.

    Transposes the inputs, calls ``shrink_width`` and transposes the outputs back. ``energy_fn``
    is called with the transposed image and transposed carry. ``removed`` / ``order`` are in the
    original (H, W) layout; exactly k True per COLUMN.

    img:   (H, W) or (H, W, C) image (not modified); 0 <= k < H.
    carry: arrays with leading shape (H, W).
    Returns a ShrinkResult with image (H-k, W[, C]) and removed/order (H, W).
    """
    raise NotImplementedError
