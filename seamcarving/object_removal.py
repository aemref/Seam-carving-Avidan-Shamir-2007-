"""Mask-driven object removal (paper Sec. 4.6). Owner: Berkay."""

from typing import NamedTuple

import numpy as np

from seamcarving.types import EnergyFn, SeamFn


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
    """Return a copy of the energy with removal (negative) and protection (positive) weights."""
    raise NotImplementedError


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
