"""Visualisation helpers. Owner: Berkay."""

from pathlib import Path

import numpy as np


def energy_to_image(energy: np.ndarray, log: bool = False) -> np.ndarray:
    """Min-max stretch an energy map to uint8 (H, W); log=True: log(1 + e - e.min()) first."""
    raise NotImplementedError


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
