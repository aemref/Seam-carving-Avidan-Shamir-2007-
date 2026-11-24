"""Object removal by mask-weighted seam carving (Avidan & Shamir 2007, section 4.6).

Pixels the user wants removed get a very negative energy so seams are drawn through them;
pixels the user wants protected get a very large energy so seams avoid them. The weights are
derived from the energy map itself instead of being fixed constants, so their dominance over
the ordinary energies holds for any energy scale.
"""

import operator
from typing import NamedTuple

import cv2
import numpy as np

from seamcarving import carve, energy, enlarge
from seamcarving.types import EnergyFn, SeamFn

# Lower bound for the weight scale, so the weights stay strictly positive on a flat (all-zero)
# energy map.
MIN_WEIGHT_SCALE = 1.0

DIRECTIONS = ("auto", "vertical", "horizontal")

# Per seam direction: the image axis that loses one pixel per seam, and the carve/enlarge
# functions working along it.
_SEAM_AXIS = {"vertical": 1, "horizontal": 0}
_SHRINKERS = {"vertical": carve.shrink_width, "horizontal": carve.shrink_height}
_ENLARGERS = {"vertical": enlarge.enlarge_width, "horizontal": enlarge.enlarge_height}


class RemovalResult(NamedTuple):
    """Result of removing a masked object."""

    image: np.ndarray
    n_seams: int
    direction: str  # "vertical" | "horizontal"


def apply_mask_weights(
    energy: np.ndarray,
    remove_mask: np.ndarray | None = None,
    protect_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a copy of `energy` with masked pixels replaced by removal/protection weights.

    energy: (H, W) float energy map. remove_mask, protect_mask: (H, W) bool or None (skipped).
    Removal pixels get -R and protected pixels get +P (see `_removal_weight` and
    `_protection_weight`); protection wins where the masks overlap. Returns (H, W) float64.
    Raises ValueError if the energy contains inf/nan or a mask's shape differs from the
    energy's shape.
    """
    base = _validated_energy(energy)
    weighted = base.copy()
    if remove_mask is not None:
        weighted[_validated_mask(remove_mask, base.shape)] = -_removal_weight(base)
    if protect_mask is not None:
        weighted[_validated_mask(protect_mask, base.shape)] = _protection_weight(base)
    return weighted


def _removal_weight(energy: np.ndarray) -> float:
    """Return R > 2 * H * max|e| (and R > 0) for an (H, W) energy map.

    With -R on every removal pixel, a vertical seam crossing one more removal pixel is always
    cheaper than one crossing fewer, whatever the (possibly negative) base energies are.
    """
    n_rows = energy.shape[0]
    return (2 * n_rows + 1) * _weight_scale(energy)


def _protection_weight(energy: np.ndarray) -> float:
    """Return P > H * (R + max|e|) for an (H, W) energy map, with R = `_removal_weight`.

    With +P on every protected pixel, a single protected pixel costs a vertical seam more than
    it can gain by crossing the whole removal mask.
    """
    n_rows = energy.shape[0]
    scale = _weight_scale(energy)
    return n_rows * (_removal_weight(energy) + scale) + scale


def _weight_scale(energy: np.ndarray) -> float:
    """Return max|e| of the energy map, but at least MIN_WEIGHT_SCALE."""
    return max(float(np.max(np.abs(energy), initial=0.0)), MIN_WEIGHT_SCALE)


def _validated_energy(energy: np.ndarray) -> np.ndarray:
    """Return `energy` as float64; raise ValueError if it contains inf or nan."""
    energy = np.asarray(energy, dtype=np.float64)
    if not np.all(np.isfinite(energy)):
        raise ValueError("energy contains inf or nan; masked weights need finite energies")
    return energy


def _validated_mask(mask: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    """Return `mask` as a bool array; raise ValueError if its shape is not `expected_shape`."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != expected_shape:
        raise ValueError(f"mask shape {mask.shape} does not match expected shape {expected_shape}")
    return mask


def seams_needed(mask: np.ndarray) -> tuple[int, int]:
    """Return how many seams remove every pixel of an (H, W) bool mask in each direction.

    Returns (vertical, horizontal): the mask pixel count of the most crowded row and of the
    most crowded column. An empty mask gives (0, 0).
    """
    pixels_per_row = np.count_nonzero(mask, axis=1)
    pixels_per_col = np.count_nonzero(mask, axis=0)
    return int(np.max(pixels_per_row, initial=0)), int(np.max(pixels_per_col, initial=0))


def remove_object(
    img: np.ndarray,
    remove_mask: np.ndarray,
    protect_mask: np.ndarray | None = None,
    restore_size: bool = True,
    direction: str = "auto",
    dilate: int = 0,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
) -> RemovalResult:
    """Remove the object under `remove_mask` by carving seams until no removal pixel is left.

    The removal mask is first grown by `dilate` pixels (to catch shadows and edge halos), then
    pixels it shares with `protect_mask` are dropped from it (protection wins). Seams are
    removed one at a time; each step computes energy_fn(img, carry) with
    carry = (remove_mask, protect_mask) carved along with the image, weights it with
    `apply_mask_weights` and finds the seam with seam_fn. Like any seam carving, this changes
    the whole image, not only the masked region (paper section 4.6).

    Args:
        img: (H, W) or (H, W, C) image (not modified).
        remove_mask: (H, W) bool, pixels to remove.
        protect_mask: (H, W) bool or None, pixels to keep.
        restore_size: afterwards insert as many seams as were removed (paper Fig. 12), so the
            output has the input's size. The insertion ignores `protect_mask`.
        direction: "vertical", "horizontal" or "auto": the one needing fewer seams according
            to `seams_needed`, vertical on a tie.
        dilate: grow the removal mask by this many pixels first (square max filter, so
            diagonal neighbours are included); 0 keeps it.
        energy_fn: base energy (img, carry) -> (H, W); None means `compute_energy(img)`.
        seam_fn: vertical seam finder; None means `find_vertical_seam`.

    Returns:
        RemovalResult(image, n_seams, direction): image is (H, W - n_seams[, C]) for vertical
        seams, (H - n_seams, W[, C]) for horizontal ones, or (H, W[, C]) with `restore_size`.

    Raises:
        ValueError: bad image or mask shape, unknown direction or negative `dilate`.
        RuntimeError: the mask is still not cleared after W - 1 (or H - 1) seams.
    """
    img = _validated_image(img)
    remove, protect = _prepared_masks(remove_mask, protect_mask, img.shape[:2], dilate)
    chosen = _resolved_direction(direction, remove)
    carved = _carve_until_cleared(img, (remove, protect), chosen, _mask_weighted(energy_fn), seam_fn)
    seam_axis = _SEAM_AXIS[chosen]
    n_seams = img.shape[seam_axis] - carved.shape[seam_axis]
    if restore_size and n_seams > 0:
        carved = _ENLARGERS[chosen](carved, n_seams, energy_fn=energy_fn, seam_fn=seam_fn)
    return RemovalResult(carved, n_seams, chosen)


def _validated_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ValueError(f"img must have shape (H, W) or (H, W, C), got {img.shape}")
    return img


def _prepared_masks(
    remove_mask: np.ndarray,
    protect_mask: np.ndarray | None,
    image_shape: tuple[int, ...],
    dilate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (remove, protect) bool masks: remove dilated, minus protect; protect never None."""
    remove = _dilated(_validated_mask(remove_mask, image_shape), dilate)
    if protect_mask is None:
        protect = np.zeros(image_shape, dtype=bool)
    else:
        protect = _validated_mask(protect_mask, image_shape)
    return remove & ~protect, protect


def _dilated(mask: np.ndarray, radius: int) -> np.ndarray:
    """Return `mask` grown by `radius` pixels in every direction, diagonals included.

    This is a max filter over a (2 * radius + 1) square window. Raises ValueError if radius < 0.
    """
    radius = operator.index(radius)
    if radius < 0:
        raise ValueError(f"dilate must be >= 0, got {radius}")
    if radius == 0:
        return mask
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _resolved_direction(direction: str, remove_mask: np.ndarray) -> str:
    """Return "vertical" or "horizontal"; "auto" picks the one needing fewer seams."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if direction != "auto":
        return direction
    n_vertical, n_horizontal = seams_needed(remove_mask)
    return "horizontal" if n_horizontal < n_vertical else "vertical"


def _mask_weighted(base: EnergyFn | None) -> EnergyFn:
    """Wrap `base` so the masks in carry = (remove, protect) weight its energy.

    shrink_height calls the wrapper with the transposed image and transposed carry, so the
    masks always match the image they come with.
    """

    def weighted_energy(img: np.ndarray, carry: tuple) -> np.ndarray:
        base_energy = energy.compute_energy(img) if base is None else base(img, carry)
        return apply_mask_weights(base_energy, carry[0], carry[1])

    return weighted_energy


def _carve_until_cleared(
    img: np.ndarray, masks: tuple, direction: str, energy_fn: EnergyFn, seam_fn: SeamFn | None
) -> np.ndarray:
    """Remove one seam at a time until masks[0] (the removal mask) is empty; return the image.

    Stops with RuntimeError after the last seam the image can lose (W - 1 or H - 1), so a mask
    that never clears cannot cause an endless loop.
    """
    shrink = _SHRINKERS[direction]
    max_seams = img.shape[_SEAM_AXIS[direction]] - 1
    carved, carry = img.copy(), masks
    for _ in range(max_seams):
        if not carry[0].any():
            break
        result = shrink(carved, 1, energy_fn=energy_fn, seam_fn=seam_fn, carry=carry)
        carved, carry = result.image, result.carry
    if carry[0].any():
        raise RuntimeError(
            f"{int(carry[0].sum())} removal pixels are left after {max_seams} {direction} seams"
        )
    return carved
