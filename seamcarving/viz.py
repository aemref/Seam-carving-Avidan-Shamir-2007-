"""Visualisation helpers shared by all seam-carving experiments.

Conventions: images are (H, W, 3) BGR uint8 or (H, W) gray uint8, energy maps are
(H, W) float64, a vertical seam is an (H,) int array with seam[r] = column in row r.
No function modifies its inputs.
"""

from pathlib import Path

import cv2
import numpy as np

UINT8_MAX = 255


def energy_to_image(energy: np.ndarray, log: bool = False) -> np.ndarray:
    """Min-max stretch an (H, W) energy map to an (H, W) uint8 image.

    With log=True the map is first compressed with log(1 + e - e.min()), so maps with
    large negative object-removal weights still show their structure. A constant map
    yields all zeros.
    """
    values = _as_2d_float(energy)
    if log:
        values = np.log1p(values - values.min())
    return _stretch_to_uint8(values)


def _as_2d_float(energy: np.ndarray) -> np.ndarray:
    values = np.asarray(energy, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"energy must be a 2D (H, W) array, got shape {values.shape}")
    return values


def _stretch_to_uint8(values: np.ndarray) -> np.ndarray:
    value_range = values.max() - values.min()
    if value_range == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    stretched = (values - values.min()) / value_range * UINT8_MAX
    return np.clip(np.rint(stretched), 0, UINT8_MAX).astype(np.uint8)


def draw_seams(
    img: np.ndarray, removed: np.ndarray, color: tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """Return a BGR (H, W, 3) copy of img with the marked pixels painted in color.

    removed is either a bool (H, W) map of pixels to paint or a single vertical seam
    (H,) of column indices.
    """
    canvas = _to_bgr(img)
    canvas[_seam_mask(removed, canvas.shape[:2])] = color
    return canvas


def _seam_mask(removed: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    removed = np.asarray(removed)
    if removed.ndim == 1:
        return _mask_from_seam(removed, image_shape)
    if removed.shape != image_shape or removed.dtype != np.bool_:
        raise ValueError(
            f"removed must be a bool map of shape {image_shape} or a seam of shape "
            f"({image_shape[0]},), got {removed.dtype} {removed.shape}"
        )
    return removed


def _mask_from_seam(seam: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    if seam.shape != (height,) or not np.issubdtype(seam.dtype, np.integer):
        raise ValueError(f"seam must be an integer array of shape ({height},), got {seam.shape}")
    if seam.min() < 0 or seam.max() >= width:
        raise ValueError(f"seam columns must lie in [0, {width}), got [{seam.min()}, {seam.max()}]")
    mask = np.zeros(image_shape, dtype=bool)
    mask[np.arange(height), seam] = True
    return mask


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Return a new (H, W, 3) BGR uint8 array from a gray or BGR uint8 image."""
    if img.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {img.dtype}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()
    raise ValueError(f"image must have shape (H, W) or (H, W, 3), got {img.shape}")


def comparison_grid(
    images: list[np.ndarray],
    labels: list[str],
    cols: int | None = None,
    cell_height: int = 320,
) -> np.ndarray:
    """Lay images out in a labelled grid. Returns BGR uint8; labels must be ASCII."""
    raise NotImplementedError


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 15) -> None:
    """Write a list of BGR uint8 frames as an animated GIF."""
    raise NotImplementedError
