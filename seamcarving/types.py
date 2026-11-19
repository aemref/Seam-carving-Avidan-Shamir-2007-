"""Shared type aliases for the pluggable parts of the pipeline."""

from collections.abc import Callable

import numpy as np

# (img, carry) -> energy (H, W) float64. ``carry`` holds the side arrays (masks, ...)
# that are carved together with the image, in their current state.
EnergyFn = Callable[[np.ndarray, tuple], np.ndarray]

# energy (H, W) -> vertical seam (H,) int64.
SeamFn = Callable[[np.ndarray], np.ndarray]
