"""Seam insertion and content-aware enlarging (Avidan & Shamir 2007, sections 4.3 and 4.4).

Enlarging by k columns is done the paper's way: find the first k seams that shrinking would
remove (``carve.shrink_width(...).removed`` holds them in original coordinates) and duplicate
all of them at once. Repeatedly inserting the single best seam would pick the same seam again
and again and produce a stretched band.
"""

import numpy as np

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
    """Widen the image by k columns, at most ``max_step`` of the current width per stage."""
    raise NotImplementedError


def enlarge_height(
    img: np.ndarray,
    k: int,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
    max_step: float = 0.5,
) -> np.ndarray:
    """Heighten the image by k rows, at most ``max_step`` of the current height per stage."""
    raise NotImplementedError


def amplify_content(
    img: np.ndarray,
    scale: float = 1.2,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
) -> np.ndarray:
    """Content amplification: scale up, then carve back. Output has the input's size."""
    raise NotImplementedError
