"""Seam removal and shrinking (paper Sec. 4.1). Owner: Mirza."""

from typing import NamedTuple

import numpy as np

from seamcarving.types import EnergyFn, SeamFn


class ShrinkResult(NamedTuple):
    """Result of removing k seams."""

    image: np.ndarray  # (H, W-k[, C]), same dtype as the input
    carry: tuple  # the carry arrays, cut with the same seams
    removed: np.ndarray  # bool (H, W): removed pixels in ORIGINAL coordinates; k True per row
    order: np.ndarray  # int32 (H, W): index 0..k-1 of the seam that removed the pixel; -1 if kept


def remove_vertical_seam(arr: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Remove one vertical seam: (H, W[, C]) -> (H, W-1[, C])."""
    raise NotImplementedError


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
    """
    raise NotImplementedError
