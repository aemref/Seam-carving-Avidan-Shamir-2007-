"""Shared dataset and output helpers for the experiment scripts. Owner: Emre.

The four directory constants are read when a function is called, not when it is defined,
so tests can point them at a temporary folder with ``monkeypatch.setattr``.
"""

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
MASKS_DIR = ROOT / "data" / "masks"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

CATEGORIES = ("manzara", "nesne", "insan", "mimari", "yogun")
PEOPLE = ("emre", "ahsen", "mirza", "furkan", "berkay")

# <category>_<nn>_<person>.jpg, e.g. manzara_01_emre.jpg
_IMAGE_NAME = re.compile(r"(?P<category>[a-z]+)_(?P<nn>\d{2})_(?P<person>[a-z]+)\.jpg")


def list_images(category: str | None = None, person: str | None = None) -> list[Path]:
    """List ``IMAGES_DIR/<category>_<nn>_<person>.jpg`` files, sorted by file name.

    ``category`` and ``person`` filter the result when given. Files whose name does not
    follow the pattern, or whose category is not one of ``CATEGORIES``, are skipped.
    Returns an empty list when the directory is missing or has no matching file.
    Raises ``ValueError`` for a ``category`` not in ``CATEGORIES`` or a ``person`` not in
    ``PEOPLE``, so a typo fails loudly instead of silently returning no images.
    """
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
    if person is not None and person not in PEOPLE:
        raise ValueError(f"unknown person {person!r}; expected one of {PEOPLE}")
    images_dir = Path(IMAGES_DIR)
    if not images_dir.is_dir():
        return []
    found = []
    for path in images_dir.iterdir():
        match = _IMAGE_NAME.fullmatch(path.name)
        if match is None or match["category"] not in CATEGORIES or not path.is_file():
            continue
        if category is not None and match["category"] != category:
            continue
        if person is not None and match["person"] != person:
            continue
        found.append(path)
    return sorted(found, key=lambda p: p.name)


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
