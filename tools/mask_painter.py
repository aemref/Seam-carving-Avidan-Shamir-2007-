"""Paint object-removal and protection masks for seam carving with the mouse."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np

Point = tuple[float, float]
"""A (row, col) position; may lie outside the image while the mouse is dragged off it."""


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
