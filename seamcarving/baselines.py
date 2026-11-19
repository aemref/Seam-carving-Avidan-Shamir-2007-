"""Baseline resizing methods to compare against. Owner: Mirza."""

import numpy as np


def scale(img: np.ndarray, new_h: int, new_w: int, interpolation: int | None = None) -> np.ndarray:
    """Resize with ``cv2.resize``. None: INTER_AREA when shrinking, INTER_CUBIC when enlarging."""
    raise NotImplementedError


def center_crop(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Crop a (new_h, new_w) window from the centre."""
    raise NotImplementedError


def best_crop(
    img: np.ndarray, new_h: int, new_w: int, energy: np.ndarray | None = None
) -> np.ndarray:
    """Crop the (new_h, new_w) window with the highest total energy."""
    raise NotImplementedError


def remove_columns(img: np.ndarray, k: int, energy: np.ndarray | None = None) -> np.ndarray:
    """Remove the k columns with the lowest total energy."""
    raise NotImplementedError
