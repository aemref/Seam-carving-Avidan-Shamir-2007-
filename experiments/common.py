"""Shared dataset and output helpers for the experiment scripts. Owner: Emre.

The four directory constants are read when a function is called, not when it is defined,
so tests can point them at a temporary folder with ``monkeypatch.setattr``.
"""

import csv
import io
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
MASKS_DIR = ROOT / "data" / "masks"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

CATEGORIES = ("manzara", "nesne", "insan", "mimari", "yogun")
PEOPLE = ("emre", "ahsen", "mirza", "furkan", "berkay")
MASK_KINDS = ("remove", "protect")

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


def _decode(path: str | Path, flags: int) -> np.ndarray:
    """Read and decode an image file with OpenCV.

    ``cv2.imread`` fails silently on non-ASCII paths on Windows ("Masaüstü"), so the bytes
    are read by NumPy and decoded in memory. Raises ``FileNotFoundError`` for a missing
    file and ``ValueError`` when OpenCV cannot decode it (``cv2.imread`` would return None).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags) if data.size else None
    if img is None:
        raise ValueError(f"cannot decode image file: {path}")
    return img


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as BGR uint8 ``(H, W, 3)``; grayscale files get three equal channels.

    Works with non-ASCII paths on every platform. Raises ``FileNotFoundError`` if the file
    does not exist and ``ValueError`` if it cannot be decoded.
    """
    return _decode(path, cv2.IMREAD_COLOR)


def load_mask(image_path: str | Path, kind: str = "remove") -> np.ndarray | None:
    """Load ``MASKS_DIR/<image stem>_<kind>.png`` as a bool ``(H, W)`` mask (pixel > 127).

    ``kind`` is ``"remove"`` or ``"protect"``; anything else raises ``ValueError``.
    Returns ``None`` when the mask file does not exist. The image itself is read to check
    the size: a mask whose ``(H, W)`` differs from the image's raises ``ValueError``.
    Colour or alpha mask files are converted to grayscale before thresholding.
    """
    if kind not in MASK_KINDS:
        raise ValueError(f"unknown mask kind {kind!r}; expected one of {MASK_KINDS}")
    image_path = Path(image_path)
    mask_path = Path(MASKS_DIR) / f"{image_path.stem}_{kind}.png"
    if not mask_path.is_file():
        return None
    mask = _decode(mask_path, cv2.IMREAD_GRAYSCALE) > 127
    image_shape = load_image(image_path).shape[:2]
    if mask.shape != image_shape:
        raise ValueError(
            f"mask {mask_path.name} is {mask.shape[0]}x{mask.shape[1]} (HxW) but image "
            f"{image_path.name} is {image_shape[0]}x{image_shape[1]}"
        )
    return mask


def _output_path(folder: str | Path, name: str, suffix: str) -> Path:
    """Return ``folder/<name><suffix>``, creating ``folder``; ``name`` is a bare file stem."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"name must be a plain file name without folders, got {name!r}")
    if name.lower().endswith(suffix):
        raise ValueError(f"name must not include the {suffix} extension, got {name!r}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}{suffix}"


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an image array to uint8: bool -> 0/255, other numbers -> clip(rint(x), 0, 255)."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    if not (np.issubdtype(img.dtype, np.integer) or np.issubdtype(img.dtype, np.floating)):
        raise ValueError(f"cannot save an image of dtype {img.dtype}")
    values = img.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("image contains NaN or inf")
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def save_figure(name: str, img: np.ndarray) -> Path:
    """Write ``img`` as ``FIG_DIR/<name>.png`` (lossless) and return the written path.

    ``img`` is ``(H, W)`` gray, ``(H, W, 3)`` BGR or ``(H, W, 4)`` BGRA. A matplotlib RGBA
    buffer must be converted to BGR(A) first. Non-uint8 input is converted, never rescaled:
    bool becomes 0/255 and numbers become ``np.clip(np.rint(x), 0, 255)``, so a float image
    in [0, 1] must be multiplied by 255 by the caller. ``name`` is a file stem without
    folders or extension, e.g. ``"d2_shrink_manzara_01_emre_25"``. The folder is created
    if needed and an existing file is overwritten; the same input gives the same bytes.
    Works with non-ASCII paths. Raises ``ValueError`` for a bad name, shape or dtype.
    """
    arr = _to_uint8(img)
    if arr.size == 0 or not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (3, 4))):
        raise ValueError(
            f"expected a non-empty (H, W), (H, W, 3) or (H, W, 4) image, got {arr.shape}"
        )
    path = _output_path(FIG_DIR, name, ".png")
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr))
    if not ok:
        raise ValueError(f"OpenCV could not encode {path.name}")
    buf.tofile(path)
    return path


def save_table(name: str, rows: Iterable[Sequence], header: Sequence[str]) -> Path:
    """Write ``header`` and ``rows`` as ``TABLE_DIR/<name>.csv`` and return the written path.

    UTF-8 without BOM, comma separated, ``\\n`` line endings on every platform. Values are
    written with ``str()`` (``None`` becomes an empty cell), so round floats before passing
    them if a fixed precision is wanted. Every row must have exactly ``len(header)`` values;
    otherwise ``ValueError`` is raised and no file is written. ``name`` is a file stem
    without folders or extension, e.g. ``"d2_shrink"``. The folder is created if needed and
    an existing file is overwritten; the same input gives the same bytes.
    """
    header = list(header)
    if not header:
        raise ValueError("header must not be empty")
    rows = list(rows)
    for i, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise TypeError(f"row {i} is a string, expected a sequence of values")
        if len(row) != len(header):
            raise ValueError(f"row {i} has {len(row)} values but the header has {len(header)}")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path = _output_path(TABLE_DIR, name, ".csv")
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    return path
