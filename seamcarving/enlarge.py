"""Seam insertion and content-aware enlarging (Avidan & Shamir 2007, sections 4.3 and 4.4).

Enlarging by k columns is done the paper's way: find the first k seams that shrinking would
remove (``carve.shrink_width(...).removed`` holds them in original coordinates) and duplicate
all of them at once. Repeatedly inserting the single best seam would pick the same seam again
and again and produce a stretched band.
"""

import math

import cv2
import numpy as np

from seamcarving import carve
from seamcarving.types import EnergyFn, SeamFn


def _seam_count(chosen: np.ndarray) -> int:
    """Validate a seam map and return k, the number of True values per row.

    chosen: bool (H, W). Raises ValueError if it is not a 2-D bool array or if the rows do
    not all hold the same number of True values.
    """
    if not isinstance(chosen, np.ndarray) or chosen.dtype != np.bool_:
        raise ValueError("chosen must be a bool numpy array")
    if chosen.ndim != 2:
        raise ValueError(f"chosen must be 2-D (H, W), got shape {chosen.shape}")
    counts = chosen.sum(axis=1)
    if counts.size == 0:
        return 0
    if np.any(counts != counts[0]):
        raise ValueError("chosen must hold the same number of True values in every row")
    return int(counts[0])


def _to_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert float64 values back to the image dtype.

    Integer dtypes are rounded with np.rint (round half to even) and clipped to the dtype
    range; for uint8 this is np.clip(np.rint(x), 0, 255).astype(np.uint8). Float dtypes are
    cast without rounding or clipping.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    raise TypeError(f"unsupported image dtype {dtype}")


def inserted_mask(chosen: np.ndarray) -> np.ndarray:
    """Positions of the inserted pixels in the enlarged image.

    chosen: bool (H, W) with exactly k True values in every row.
    Returns bool (H, W + k). If the selected columns of a row are c_0 < c_1 < ..., the new
    pixel of c_j sits at output column c_j + j + 1 (right after its source pixel).
    """
    k = _seam_count(chosen)
    h, w = chosen.shape
    mask = np.zeros((h, w + k), dtype=bool)
    if k == 0:
        return mask
    rows, cols = np.nonzero(chosen)  # row-major, so cols increase within each row
    rank = np.tile(np.arange(k), h)  # j for each selected pixel
    mask[rows, cols + rank + 1] = True
    return mask


def insert_vertical_seams(img: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Duplicate the selected pixels of every row, widening the image by k columns.

    img: (H, W) or (H, W, C); chosen: bool (H, W) with exactly k True values per row.
    Returns a new (H, W + k[, C]) array with the dtype of img. Each selected pixel gets a new
    pixel right after it, equal to the float64 average of the selected pixel and its right
    neighbour in img (left neighbour in the last column). Integer images are rounded with
    np.rint and clipped; float images are not rounded or clipped. Original pixels keep their
    values and order. img and chosen are not modified.
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be (H, W) or (H, W, C), got shape {img.shape}")
    if not isinstance(chosen, np.ndarray) or chosen.shape != img.shape[:2]:
        raise ValueError(f"chosen must have shape {img.shape[:2]}")
    k = _seam_count(chosen)
    if k == 0:
        return img.copy()

    h, w = chosen.shape
    tail = img.shape[2:]

    # Every selected pixel is written twice. Each row then has W + k entries, so the flat
    # repeat reshapes back into a rectangle with originals in their original order.
    reps = 1 + chosen.astype(np.intp)
    out = np.repeat(img.reshape(h * w, *tail), reps.ravel(), axis=0).reshape(h, w + k, *tail)

    # Replace the second copy by the neighbour average.
    rows, cols = np.nonzero(chosen)
    rank = np.tile(np.arange(k), h)
    neighbour = np.where(cols < w - 1, cols + 1, cols - 1)
    neighbour = np.clip(neighbour, 0, w - 1)  # W == 1: the pixel is its own neighbour
    avg = (img[rows, cols].astype(np.float64) + img[rows, neighbour].astype(np.float64)) / 2.0
    out[rows, cols + rank + 1] = _to_dtype(avg, img.dtype)
    return out


def enlarge_width(
    img: np.ndarray,
    k: int,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
    max_step: float = 0.5,
) -> np.ndarray:
    """Widen an image by k columns with the k-seam insertion method.

    img: (H, W) or (H, W, C). Returns (H, W + k[, C]) with the same dtype.
    Works in steps: step = max(1, min(remaining, floor(max_step * current width))). Each step
    finds the first `step` seams that carve.shrink_width would remove and duplicates all of
    them at once. energy_fn and seam_fn are passed to carve.shrink_width unchanged.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if not 0 < max_step <= 1:
        raise ValueError(f"max_step must be in (0, 1], got {max_step}")
    cur = img.copy()
    remaining = int(k)
    while remaining > 0:
        width = cur.shape[1]
        step = max(1, min(remaining, math.floor(max_step * width)))
        res = carve.shrink_width(cur, step, energy_fn=energy_fn, seam_fn=seam_fn)
        cur = insert_vertical_seams(cur, res.removed)
        remaining -= step
    return cur


def enlarge_height(
    img: np.ndarray,
    k: int,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
    max_step: float = 0.5,
) -> np.ndarray:
    """Make an image k rows taller by inserting horizontal seams.

    img: (H, W) or (H, W, C). Returns (H + k, W[, C]) with the same dtype.
    Rows and columns are swapped (the channel axis stays last), enlarge_width is called and
    the result is swapped back. energy_fn and seam_fn therefore see transposed images.
    """
    out_t = enlarge_width(np.swapaxes(img, 0, 1), k, energy_fn, seam_fn, max_step)
    return np.ascontiguousarray(np.swapaxes(out_t, 0, 1))


def amplify_content(
    img: np.ndarray,
    scale: float = 1.2,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
) -> np.ndarray:
    """Content amplification (paper section 4.4): scale up, then carve back to the original size.

    img: (H, W) or (H, W, C). Returns an array with the same shape and dtype as img.
    The image is resized with cv2.INTER_CUBIC to (round(scale * H), round(scale * W)), then
    carve.shrink_width and carve.shrink_height remove the extra columns and rows.
    scale must be >= 1; scale == 1 returns a copy.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if scale == 1:
        return img.copy()
    h, w = img.shape[:2]
    new_h, new_w = round(scale * h), round(scale * w)
    big = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    if big.ndim < img.ndim:  # cv2 drops a single channel axis
        big = big[..., np.newaxis]
    if new_w > w:
        big = carve.shrink_width(big, new_w - w, energy_fn=energy_fn, seam_fn=seam_fn).image
    if new_h > h:
        big = carve.shrink_height(big, new_h - h, energy_fn=energy_fn, seam_fn=seam_fn).image
    return big
