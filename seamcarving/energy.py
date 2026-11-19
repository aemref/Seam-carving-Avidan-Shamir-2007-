"""Energy maps (paper Eq. 1 and variants). Owner: Ahsen."""

import numpy as np

from seamcarving.types import EnergyFn

METHODS = ("forward", "roberts", "sobel", "scharr", "laplacian")


def compute_energy(
    img: np.ndarray,
    method: str = "sobel",
    color: str = "gray",
    blur_sigma: float = 0.0,
    norm: str = "l1",
) -> np.ndarray:
    """Compute the energy map of an image.

    Args:
        img: (H, W) or (H, W, 3) image.
        method: one of ``METHODS``.
        color: "gray" | "sum" (compute per channel, then add).
        blur_sigma: if > 0, apply a Gaussian blur first.
        norm: "l1" | "l2".

    Returns:
        (H, W) float64 array, every value >= 0.
    """
    raise NotImplementedError


def make_energy_fn(**kwargs) -> EnergyFn:
    """Wrap ``compute_energy`` (with ``kwargs``) into the ``(img, carry)`` signature."""
    raise NotImplementedError
