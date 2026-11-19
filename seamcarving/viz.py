"""Visualisation helpers shared by all seam-carving experiments.

Conventions: images are (H, W, 3) BGR uint8 or (H, W) gray uint8, energy maps are
(H, W) float64, a vertical seam is an (H,) int array with seam[r] = column in row r.
No function modifies its inputs.
"""

from pathlib import Path

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
    """Paint seams on a BGR copy. ``removed``: bool (H, W) or a single seam (H,)."""
    raise NotImplementedError


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
