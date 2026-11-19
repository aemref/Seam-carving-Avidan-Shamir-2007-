"""Visualisation helpers shared by all seam-carving experiments.

Conventions: images are (H, W, 3) BGR uint8 or (H, W) gray uint8, energy maps are
(H, W) float64, a vertical seam is an (H,) int array with seam[r] = column in row r.
No function modifies its inputs.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

UINT8_MAX = 255
MILLISECONDS_PER_SECOND = 1000

GRID_GAP = 10
GRID_BACKGROUND = (255, 255, 255)
DEFAULT_COLUMNS_PER_ROW = 5

LABEL_STRIP_HEIGHT = 32
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.6
LABEL_MIN_FONT_SCALE = 0.4
LABEL_THICKNESS = 1
LABEL_COLOR = (0, 0, 0)
LABEL_HORIZONTAL_PADDING = 4


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
    """Lay out labelled images on a white BGR uint8 canvas.

    Every image is scaled to cell_height (aspect ratio kept) with its label in a strip
    below. Cells fill rows left to right; cols=None puts up to 5 images in one row and
    wraps longer lists into rows of 5. Each grid column is as wide as its widest cell,
    narrower cells are centred in it. Labels must be ASCII (cv2.putText cannot draw
    e.g. Turkish characters); a label wider than its image first shrinks its font
    down to LABEL_MIN_FONT_SCALE, then widens the cell instead of being clipped.
    """
    _validate_grid_input(images, labels, cols, cell_height)
    cells = [_labelled_cell(img, label, cell_height) for img, label in zip(images, labels)]
    columns_per_row = cols or DEFAULT_COLUMNS_PER_ROW
    rows = [
        cells[start : start + columns_per_row] for start in range(0, len(cells), columns_per_row)
    ]
    return _assemble_grid(rows)


def _validate_grid_input(
    images: list[np.ndarray], labels: list[str], cols: int | None, cell_height: int
) -> None:
    if not images:
        raise ValueError("images must not be empty")
    if len(images) != len(labels):
        raise ValueError(f"got {len(images)} images but {len(labels)} labels")
    for label in labels:
        if not label.isascii():
            raise ValueError(f"label {label!r} contains non-ASCII characters cv2 cannot draw")
    if cols is not None and cols < 1:
        raise ValueError(f"cols must be at least 1, got {cols}")
    if cell_height < 1:
        raise ValueError(f"cell_height must be at least 1, got {cell_height}")


def _labelled_cell(img: np.ndarray, label: str, cell_height: int) -> np.ndarray:
    """Return a (cell_height + LABEL_STRIP_HEIGHT, W', 3) image-plus-label cell."""
    scaled = _scale_to_height(_to_bgr(img), cell_height)
    font_scale = _fitting_font_scale(label, scaled.shape[1])
    cell_width = max(scaled.shape[1], _padded_text_width(label, font_scale))
    cell = _blank_canvas(cell_height + LABEL_STRIP_HEIGHT, cell_width)
    _paste_centered(cell, scaled, top=0, slot_left=0, slot_width=cell_width)
    _draw_label(cell, label, font_scale, top=cell_height)
    return cell


def _scale_to_height(img: np.ndarray, height: int) -> np.ndarray:
    width = max(1, round(img.shape[1] * height / img.shape[0]))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _fitting_font_scale(label: str, available_width: int) -> float:
    natural_width = _padded_text_width(label, LABEL_FONT_SCALE)
    if natural_width <= available_width:
        return LABEL_FONT_SCALE
    shrunk_scale = LABEL_FONT_SCALE * available_width / natural_width
    return max(shrunk_scale, LABEL_MIN_FONT_SCALE)


def _padded_text_width(label: str, font_scale: float) -> int:
    (text_width, _), _ = cv2.getTextSize(label, LABEL_FONT, font_scale, LABEL_THICKNESS)
    return text_width + 2 * LABEL_HORIZONTAL_PADDING


def _draw_label(cell: np.ndarray, label: str, font_scale: float, top: int) -> None:
    (text_width, text_height), _ = cv2.getTextSize(label, LABEL_FONT, font_scale, LABEL_THICKNESS)
    left = (cell.shape[1] - text_width) // 2
    baseline_row = top + (LABEL_STRIP_HEIGHT + text_height) // 2
    cv2.putText(
        cell,
        label,
        (left, baseline_row),
        LABEL_FONT,
        font_scale,
        LABEL_COLOR,
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def _assemble_grid(rows: list[list[np.ndarray]]) -> np.ndarray:
    column_widths = _column_widths(rows)
    row_height = rows[0][0].shape[0]
    canvas_height = len(rows) * row_height + (len(rows) + 1) * GRID_GAP
    canvas_width = sum(column_widths) + (len(column_widths) + 1) * GRID_GAP
    canvas = _blank_canvas(canvas_height, canvas_width)
    column_lefts = np.cumsum([GRID_GAP] + [width + GRID_GAP for width in column_widths[:-1]])
    for row_index, row in enumerate(rows):
        top = GRID_GAP + row_index * (row_height + GRID_GAP)
        for cell, left, width in zip(row, column_lefts, column_widths):
            _paste_centered(canvas, cell, top=top, slot_left=int(left), slot_width=width)
    return canvas


def _column_widths(rows: list[list[np.ndarray]]) -> list[int]:
    column_count = len(rows[0])
    return [
        max(row[column].shape[1] for row in rows if column < len(row))
        for column in range(column_count)
    ]


def _blank_canvas(height: int, width: int) -> np.ndarray:
    return np.full((height, width, 3), GRID_BACKGROUND, dtype=np.uint8)


def _paste_centered(
    canvas: np.ndarray, patch: np.ndarray, top: int, slot_left: int, slot_width: int
) -> None:
    height, width = patch.shape[:2]
    left = slot_left + (slot_width - width) // 2
    canvas[top : top + height, left : left + width] = patch


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 15) -> None:
    """Write BGR or gray uint8 frames, all of equal (H, W), as an endlessly looping GIF.

    Note: Pillow merges consecutive identical frames into one longer frame.
    """
    _validate_gif_input(frames, fps)
    first, *rest = [_to_rgb_pil_image(frame) for frame in frames]
    frame_duration_ms = round(MILLISECONDS_PER_SECOND / fps)
    first.save(Path(path), save_all=True, append_images=rest, duration=frame_duration_ms, loop=0)


def _validate_gif_input(frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("frames must not be empty")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    frame_sizes = {frame.shape[:2] for frame in frames}
    if len(frame_sizes) > 1:
        raise ValueError(f"all frames must have the same (H, W), got {sorted(frame_sizes)}")


def _to_rgb_pil_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(_to_bgr(frame), cv2.COLOR_BGR2RGB))
