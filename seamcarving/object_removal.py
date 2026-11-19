"""Object removal by mask-weighted seam carving (Avidan & Shamir 2007, section 4.6).

Pixels the user wants removed get a very negative energy so seams are drawn through them;
pixels the user wants protected get a very large energy so seams avoid them. The weights are
derived from the energy map itself instead of being fixed constants, so their dominance over
the ordinary energies holds for any energy scale.
"""

from typing import NamedTuple

import numpy as np

from seamcarving.types import EnergyFn, SeamFn

# Lower bound for the weight scale, so the weights stay strictly positive on a flat (all-zero)
# energy map.
MIN_WEIGHT_SCALE = 1.0


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
        raise ValueError(f"mask shape {mask.shape} does not match energy shape {expected_shape}")
    return mask


def seams_needed(mask: np.ndarray) -> tuple[int, int]:
    """Return (number of vertical seams, number of horizontal seams) needed to clear the mask."""
    raise NotImplementedError


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
    """Remove seams until no pixel of ``remove_mask`` is left.

    Args:
        img: (H, W[, 3]) uint8 image.
        remove_mask: bool (H, W), pixels to remove.
        protect_mask: bool (H, W), pixels to keep.
        restore_size: insert seams afterwards to get back to the input size.
        direction: "auto" | "vertical" | "horizontal".
        dilate: grow the removal mask by this many pixels first.
    """
    raise NotImplementedError
