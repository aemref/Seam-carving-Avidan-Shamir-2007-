"""Seam insertion and enlarging (paper Sec. 4.3-4.4). Owner: Furkan."""

import numpy as np

from seamcarving.types import EnergyFn, SeamFn


def insert_vertical_seams(img: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Duplicate the chosen pixels: (H, W[, C]) -> (H, W+k[, C]).

    Args:
        img: (H, W[, C]) image.
        chosen: bool (H, W) with exactly k True per row.

    A new pixel is inserted right after each chosen pixel: the average of the chosen pixel and
    its right neighbour (left neighbour in the last column). Original pixels keep their values
    and their order.
    """
    raise NotImplementedError


def inserted_mask(chosen: np.ndarray) -> np.ndarray:
    """Return bool (H, W+k): positions of the inserted pixels in the output."""
    raise NotImplementedError


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
