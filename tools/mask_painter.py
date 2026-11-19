"""Paint object-removal and protection masks for seam carving with the mouse."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

import numpy as np

Point = tuple[float, float]
"""A (row, col) position; may lie outside the image while the mouse is dragged off it."""

GREEN_BGR = (0, 255, 0)
RED_BGR = (0, 0, 255)

MAX_UNDO_STEPS = 50


class MaskLayer(Enum):
    """The two masks a user can paint, with their file suffix and overlay color."""

    REMOVE = ("remove", GREEN_BGR)
    PROTECT = ("protect", RED_BGR)

    def __init__(self, file_suffix: str, color_bgr: tuple[int, int, int]) -> None:
        self.file_suffix = file_suffix
        self.color_bgr = color_bgr


def paint_stroke(mask: np.ndarray, points: Sequence[Point], radius: float) -> np.ndarray:
    """Return a copy of ``mask`` with a round brush dragged through ``points``.

    Args:
        mask: (H, W) bool mask; not modified.
        points: (row, col) positions in image coordinates, in drag order. Consecutive
            points are joined, so fast drags leave no gaps. Off-image parts are clipped.
        radius: brush radius in image pixels.

    Returns:
        (H, W) bool mask with every pixel within ``radius`` of the polyline set.
    """
    painted = mask.copy()
    for start, end in _segments(points):
        _stamp_segment(painted, start, end, radius)
    return painted


def _segments(points: Sequence[Point]) -> list[tuple[Point, Point]]:
    if len(points) == 1:
        return [(points[0], points[0])]
    return list(pairwise(points))


def _stamp_segment(mask: np.ndarray, start: Point, end: Point, radius: float) -> None:
    """Set, in place, all pixels of ``mask`` within ``radius`` of segment start-end."""
    height, width = mask.shape
    top = max(int(np.floor(min(start[0], end[0]) - radius)), 0)
    bottom = min(int(np.ceil(max(start[0], end[0]) + radius)) + 1, height)
    left = max(int(np.floor(min(start[1], end[1]) - radius)), 0)
    right = min(int(np.ceil(max(start[1], end[1]) + radius)) + 1, width)
    if top >= bottom or left >= right:
        return
    rows, cols = np.ogrid[top:bottom, left:right]
    within_brush = _squared_distance_to_segment(rows, cols, start, end) <= radius**2
    mask[top:bottom, left:right] |= within_brush


def _squared_distance_to_segment(
    rows: np.ndarray, cols: np.ndarray, start: Point, end: Point
) -> np.ndarray:
    """Squared distance from each (row, col) grid pixel to the segment start-end."""
    step_r, step_c = end[0] - start[0], end[1] - start[1]
    offset_r, offset_c = rows - start[0], cols - start[1]
    length_sq = step_r**2 + step_c**2
    if length_sq == 0:
        along = 0.0
    else:
        along = np.clip((offset_r * step_r + offset_c * step_c) / length_sq, 0.0, 1.0)
    return (offset_r - along * step_r) ** 2 + (offset_c - along * step_c) ** 2


@dataclass
class _ActiveStroke:
    layer: MaskLayer
    radius: float
    last_point: Point


@dataclass(frozen=True)
class _PackedMasks:
    """Bit-packed copy of both masks, so a long undo history stays small in memory."""

    shape: tuple[int, int]
    bits: dict[MaskLayer, np.ndarray]

    @classmethod
    def pack(cls, masks: dict[MaskLayer, np.ndarray]) -> _PackedMasks:
        some_mask = next(iter(masks.values()))
        return cls(some_mask.shape, {layer: np.packbits(m) for layer, m in masks.items()})

    def unpack(self) -> dict[MaskLayer, np.ndarray]:
        pixel_count = self.shape[0] * self.shape[1]
        return {
            layer: np.unpackbits(bits, count=pixel_count).reshape(self.shape).astype(bool)
            for layer, bits in self.bits.items()
        }


class PaintSession:
    """Editable REMOVE/PROTECT mask pair with stroke painting, undo and clear.

    Pure NumPy state: it knows nothing about windows or files. All points and radii
    are in full-resolution image coordinates. Painting one layer never touches the
    other; the seam carving algorithm resolves overlap ("protection wins").
    """

    def __init__(self, remove_mask: np.ndarray, protect_mask: np.ndarray) -> None:
        """Start a session from two (H, W) masks (copied; any nonzero counts as marked)."""
        if remove_mask.shape != protect_mask.shape:
            raise ValueError(
                f"mask shapes differ: remove {remove_mask.shape}, protect {protect_mask.shape}"
            )
        self._masks = {
            MaskLayer.REMOVE: remove_mask.astype(bool, copy=True),
            MaskLayer.PROTECT: protect_mask.astype(bool, copy=True),
        }
        self._undo_stack: deque[_PackedMasks] = deque(maxlen=MAX_UNDO_STEPS)
        self._stroke: _ActiveStroke | None = None
        self._saved_masks = self._copy_masks()

    @classmethod
    def blank(cls, shape: tuple[int, int]) -> PaintSession:
        """Start a session with two empty (H, W) masks."""
        return cls(np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))

    def mask(self, layer: MaskLayer) -> np.ndarray:
        """Return a read-only (H, W) bool view of one layer's current mask."""
        view = self._masks[layer].view()
        view.flags.writeable = False
        return view

    def begin_stroke(self, layer: MaskLayer, point: Point, radius: float) -> None:
        """Start a new undoable stroke on ``layer`` and paint a dot at ``point``."""
        self._push_undo_state()
        self._stroke = _ActiveStroke(layer, radius, point)
        _stamp_segment(self._masks[layer], point, point, radius)

    def extend_stroke(self, point: Point) -> None:
        """Paint from the stroke's last point to ``point``; ignored outside a stroke."""
        if self._stroke is None:
            return
        stroke = self._stroke
        _stamp_segment(self._masks[stroke.layer], stroke.last_point, point, stroke.radius)
        stroke.last_point = point

    def end_stroke(self) -> None:
        self._stroke = None

    def undo(self) -> None:
        """Restore both masks to before the last stroke or clear; no-op if none."""
        self.end_stroke()
        if self._undo_stack:
            self._masks = self._undo_stack.pop().unpack()

    def clear(self) -> None:
        """Empty both masks as one undoable step; no-op if both are already empty."""
        self.end_stroke()
        if not any(mask.any() for mask in self._masks.values()):
            return
        self._push_undo_state()
        self._masks = {layer: np.zeros_like(mask) for layer, mask in self._masks.items()}

    def mark_saved(self) -> None:
        """Record the current masks as the state that is on disk."""
        self._saved_masks = self._copy_masks()

    def has_unsaved_changes(self) -> bool:
        return any(
            not np.array_equal(self._masks[layer], self._saved_masks[layer])
            for layer in MaskLayer
        )

    def _push_undo_state(self) -> None:
        self._undo_stack.append(_PackedMasks.pack(self._masks))

    def _copy_masks(self) -> dict[MaskLayer, np.ndarray]:
        return {layer: mask.copy() for layer, mask in self._masks.items()}
