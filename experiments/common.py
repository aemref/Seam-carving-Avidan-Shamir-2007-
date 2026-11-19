"""Shared dataset and output helpers for the experiment scripts. Owner: Emre."""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
MASKS_DIR = ROOT / "data" / "masks"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"


def list_images(category: str | None = None, person: str | None = None) -> list[Path]:
    """List ``data/images/<category>_<nn>_<person>.jpg`` files, sorted by name."""
    raise NotImplementedError


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8."""
    raise NotImplementedError


def load_mask(image_path: str | Path, kind: str = "remove") -> np.ndarray | None:
    """Load an image's mask as bool (H, W); None if missing. kind: "remove" | "protect"."""
    raise NotImplementedError


def save_figure(name: str, img: np.ndarray) -> Path:
    """Write ``results/figures/<name>.png`` and return its path."""
    raise NotImplementedError


def save_table(name: str, rows: Iterable[Sequence], header: Sequence[str]) -> Path:
    """Write ``results/tables/<name>.csv`` and return its path."""
    raise NotImplementedError
