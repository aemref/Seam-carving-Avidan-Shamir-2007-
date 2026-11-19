"""Tests for the GUI-free core of tools/mask_painter.py (synthetic data only)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

MASK_PAINTER_PATH = Path(__file__).resolve().parent.parent / "tools" / "mask_painter.py"


def _import_mask_painter():
    spec = importlib.util.spec_from_file_location("mask_painter", MASK_PAINTER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


mp = _import_mask_painter()

IMAGE_SHAPE = (40, 60)
CENTER = (20, 30)
BRUSH_RADIUS = 5


def blank_mask():
    return np.zeros(IMAGE_SHAPE, dtype=bool)


def pixels_within(radius, center, shape=IMAGE_SHAPE):
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius**2


def test_single_point_stroke_paints_disk_of_brush_radius():
    painted = mp.paint_stroke(blank_mask(), [CENTER], BRUSH_RADIUS)

    np.testing.assert_array_equal(painted, pixels_within(BRUSH_RADIUS, CENTER))


def test_stroke_does_not_modify_input_mask():
    original = blank_mask()

    mp.paint_stroke(original, [CENTER], BRUSH_RADIUS)

    assert not original.any()


def test_two_point_stroke_covers_whole_segment_without_gaps():
    start, end = (20, 5), (20, 55)

    painted = mp.paint_stroke(blank_mask(), [start, end], radius=1)

    assert painted[20, 5:56].all()
    assert not painted[20 + 2].any()


def test_diagonal_fast_drag_leaves_no_gaps():
    start, end = (2, 2), (37, 57)
    steps = 200

    painted = mp.paint_stroke(blank_mask(), [start, end], radius=1)

    rows = np.linspace(start[0], end[0], steps).round().astype(int)
    cols = np.linspace(start[1], end[1], steps).round().astype(int)
    assert painted[rows, cols].all()


def test_stroke_near_border_is_clipped_without_error():
    corner_points = [(0, 0), (-10, -10), (IMAGE_SHAPE[0] + 5, IMAGE_SHAPE[1] + 5)]

    painted = mp.paint_stroke(blank_mask(), corner_points, BRUSH_RADIUS)

    assert painted.shape == IMAGE_SHAPE
    assert painted[0, 0] and painted[-1, -1]


def test_stroke_entirely_outside_image_paints_nothing():
    far_away = [(-100, -100), (-50, -200)]

    painted = mp.paint_stroke(blank_mask(), far_away, BRUSH_RADIUS)

    assert not painted.any()
