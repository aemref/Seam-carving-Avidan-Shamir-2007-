"""Baseline resizing methods to compare against. Owner: Mirza.

Paper Figures 2 and 5. Images are indexed ``img[r, c]`` and are (H, W) or (H, W, C)
arrays. Every function returns a new array with the input dtype; inputs are not
modified. Energy maps are (H, W) float arrays; when ``energy`` is None it is computed
with ``seamcarving.energy.compute_energy``.
"""

import operator

import numpy as np


def _check_image(img: np.ndarray) -> np.ndarray:
    """Return ``img`` as an ndarray, raising ValueError unless it is (H, W) or (H, W, C)."""
    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ValueError(f"img must have shape (H, W) or (H, W, C), got {img.shape}")
    return img


def _check_size(img: np.ndarray, new_h: int, new_w: int, allow_larger: bool) -> tuple[int, int]:
    """Validate a target size; return it as (int, int)."""
    new_h, new_w = operator.index(new_h), operator.index(new_w)
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"target size must be positive, got ({new_h}, {new_w})")
    h, w = img.shape[:2]
    if not allow_larger and (new_h > h or new_w > w):
        raise ValueError(f"target size ({new_h}, {new_w}) exceeds image size ({h}, {w})")
    return new_h, new_w


def _get_energy(img: np.ndarray, energy: np.ndarray | None) -> np.ndarray:
    """Return a (H, W) float64 energy map for ``img``, computing it if ``energy`` is None."""
    if energy is None:
        from seamcarving.energy import compute_energy

        energy = compute_energy(img)
    energy = np.asarray(energy, dtype=np.float64)
    if energy.shape != img.shape[:2]:
        raise ValueError(f"energy must have shape {img.shape[:2]}, got {energy.shape}")
    return energy


def _check_k(k: int, width: int) -> int:
    """Validate a column count ``0 <= k < width``."""
    k = operator.index(k)
    if not 0 <= k < width:
        raise ValueError(f"k must satisfy 0 <= k < W={width}, got {k}")
    return k


def scale(img: np.ndarray, new_h: int, new_w: int, interpolation: int | None = None) -> np.ndarray:
    """Resize with ``cv2.resize``. None: INTER_AREA when shrinking, INTER_CUBIC when enlarging.

    (H, W[, C]) -> (new_h, new_w[, C]). interpolation: a ``cv2.INTER_*`` flag. "Shrinking"
    means neither axis grows (new_h <= H and new_w <= W); a mixed shrink/grow resize
    uses INTER_CUBIC.
    """
    import cv2

    img = _check_image(img)
    new_h, new_w = _check_size(img, new_h, new_w, allow_larger=True)
    h, w = img.shape[:2]
    if interpolation is None:
        interpolation = cv2.INTER_AREA if new_h <= h and new_w <= w else cv2.INTER_CUBIC
    # cv2.resize takes the size as (width, height).
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def center_crop(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Crop a (new_h, new_w) window from the centre.

    (H, W[, C]) -> (new_h, new_w[, C]); odd margins put the extra pixel at the end.
    Raises ValueError if the target is non-positive or larger than the image.
    """
    img = _check_image(img)
    new_h, new_w = _check_size(img, new_h, new_w, allow_larger=False)
    r0 = (img.shape[0] - new_h) // 2
    c0 = (img.shape[1] - new_w) // 2
    return img[r0:r0 + new_h, c0:c0 + new_w].copy()


def best_crop(
    img: np.ndarray, new_h: int, new_w: int, energy: np.ndarray | None = None
) -> np.ndarray:
    """Crop the (new_h, new_w) window with the highest total energy.

    Paper Fig. 2b. (H, W[, C]) -> (new_h, new_w[, C]); energy (H, W). Window sums come
    from a zero-padded 2D cumulative sum (integral image), so no Python loop runs over
    windows. Ties go to the first window in row-major order (smallest r, then smallest c).
    Ties are exact for integer-valued energies; with float energies, rounding in the
    cumulative sum can break near-ties.
    """
    img = _check_image(img)
    new_h, new_w = _check_size(img, new_h, new_w, allow_larger=False)
    energy = _get_energy(img, energy)
    h, w = energy.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral[1:, 1:] = energy.cumsum(axis=0).cumsum(axis=1)
    n_r, n_c = h - new_h + 1, w - new_w + 1
    # sums[r, c] = energy[r:r + new_h, c:c + new_w].sum()
    sums = (
        integral[new_h:, new_w:]
        - integral[:n_r, new_w:]
        - integral[new_h:, :n_c]
        + integral[:n_r, :n_c]
    )
    r0, c0 = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return img[r0:r0 + new_h, c0:c0 + new_w].copy()


def remove_columns(img: np.ndarray, k: int, energy: np.ndarray | None = None) -> np.ndarray:
    """Remove the k columns with the lowest total energy.

    Paper Fig. 2c. img: (H, W[, C]); energy (H, W); 0 <= k < W. Returns (H, W-k[, C])
    with the remaining columns in their original order. Ties remove the leftmost column
    first (stable sort).
    """
    raise NotImplementedError
