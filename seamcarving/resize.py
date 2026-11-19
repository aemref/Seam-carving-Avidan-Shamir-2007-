"""Single entry point for content-aware resizing. Owner: Emre."""

import numpy as np

from seamcarving.types import EnergyFn, SeamFn


def resize(
    img: np.ndarray,
    new_h: int,
    new_w: int,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
    protect_mask: np.ndarray | None = None,
    order: str = "width_first",
) -> np.ndarray:
    """Resize to (new_h, new_w) by removing or inserting seams.

    Args:
        img: (H, W[, 3]) uint8 image.
        protect_mask: bool (H, W); applied when shrinking only.
        order: "width_first" | "height_first".
    """
    raise NotImplementedError
