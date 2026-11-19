"""Tests for the GUI-free core of tools/mask_painter.py (synthetic data only)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

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
GRAY_LEVEL = 100


def blank_mask():
    return np.zeros(IMAGE_SHAPE, dtype=bool)


def gray_image():
    return np.full((*IMAGE_SHAPE, 3), GRAY_LEVEL, dtype=np.uint8)


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


def test_session_stroke_paints_disk_on_chosen_layer_only():
    session = mp.PaintSession.blank(IMAGE_SHAPE)

    session.begin_stroke(mp.MaskLayer.REMOVE, CENTER, BRUSH_RADIUS)
    session.end_stroke()

    np.testing.assert_array_equal(
        session.mask(mp.MaskLayer.REMOVE), pixels_within(BRUSH_RADIUS, CENTER)
    )
    assert not session.mask(mp.MaskLayer.PROTECT).any()


def test_extending_a_stroke_connects_successive_mouse_positions():
    session = mp.PaintSession.blank(IMAGE_SHAPE)

    session.begin_stroke(mp.MaskLayer.PROTECT, (10, 5), 1)
    session.extend_stroke((10, 50))
    session.end_stroke()

    assert session.mask(mp.MaskLayer.PROTECT)[10, 5:51].all()


def test_remove_and_protect_layers_are_independent():
    session = mp.PaintSession.blank(IMAGE_SHAPE)
    session.begin_stroke(mp.MaskLayer.REMOVE, CENTER, BRUSH_RADIUS)
    remove_before = session.mask(mp.MaskLayer.REMOVE).copy()

    session.begin_stroke(mp.MaskLayer.PROTECT, CENTER, BRUSH_RADIUS + 3)

    np.testing.assert_array_equal(session.mask(mp.MaskLayer.REMOVE), remove_before)
    assert session.mask(mp.MaskLayer.PROTECT)[CENTER]


def test_undo_restores_state_before_last_stroke_exactly():
    session = mp.PaintSession.blank(IMAGE_SHAPE)
    session.begin_stroke(mp.MaskLayer.REMOVE, (10, 10), BRUSH_RADIUS)
    session.end_stroke()
    remove_before = session.mask(mp.MaskLayer.REMOVE).copy()
    protect_before = session.mask(mp.MaskLayer.PROTECT).copy()
    session.begin_stroke(mp.MaskLayer.REMOVE, (30, 50), BRUSH_RADIUS)
    session.extend_stroke((5, 5))
    session.end_stroke()

    session.undo()

    np.testing.assert_array_equal(session.mask(mp.MaskLayer.REMOVE), remove_before)
    np.testing.assert_array_equal(session.mask(mp.MaskLayer.PROTECT), protect_before)


def test_undo_with_empty_history_is_harmless():
    session = mp.PaintSession.blank(IMAGE_SHAPE)

    session.undo()

    assert not session.mask(mp.MaskLayer.REMOVE).any()


def test_clear_empties_both_masks_and_is_undoable():
    session = mp.PaintSession.blank(IMAGE_SHAPE)
    session.begin_stroke(mp.MaskLayer.REMOVE, (10, 10), BRUSH_RADIUS)
    session.begin_stroke(mp.MaskLayer.PROTECT, (30, 40), BRUSH_RADIUS)
    painted_remove = session.mask(mp.MaskLayer.REMOVE).copy()

    session.clear()
    cleared_anything = any(session.mask(layer).any() for layer in mp.MaskLayer)
    session.undo()

    assert not cleared_anything
    np.testing.assert_array_equal(session.mask(mp.MaskLayer.REMOVE), painted_remove)
    assert session.mask(mp.MaskLayer.PROTECT).any()


def test_session_copies_input_masks():
    remove_mask = blank_mask()
    session = mp.PaintSession(remove_mask, blank_mask())

    session.begin_stroke(mp.MaskLayer.REMOVE, CENTER, BRUSH_RADIUS)

    assert not remove_mask.any()


def test_exposed_mask_is_read_only():
    session = mp.PaintSession.blank(IMAGE_SHAPE)

    with pytest.raises(ValueError):
        session.mask(mp.MaskLayer.REMOVE)[CENTER] = True


def test_unsaved_changes_follow_painting_saving_and_undo():
    session = mp.PaintSession.blank(IMAGE_SHAPE)
    session.begin_stroke(mp.MaskLayer.REMOVE, CENTER, BRUSH_RADIUS)
    dirty_after_stroke = session.has_unsaved_changes()

    session.mark_saved()
    clean_after_save = not session.has_unsaved_changes()
    session.undo()

    assert dirty_after_stroke and clean_after_save
    assert session.has_unsaved_changes()


def test_session_rejects_masks_of_different_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        mp.PaintSession(blank_mask(), np.zeros((1, 1), dtype=bool))


def test_brush_radius_steps_up_and_down():
    assert mp.larger_brush_radius(mp.DEFAULT_BRUSH_RADIUS) > mp.DEFAULT_BRUSH_RADIUS
    assert mp.smaller_brush_radius(mp.DEFAULT_BRUSH_RADIUS) < mp.DEFAULT_BRUSH_RADIUS


def test_brush_radius_is_clamped_to_limits():
    assert mp.smaller_brush_radius(mp.MIN_BRUSH_RADIUS) == mp.MIN_BRUSH_RADIUS
    assert mp.larger_brush_radius(mp.MAX_BRUSH_RADIUS) == mp.MAX_BRUSH_RADIUS


def test_overlay_is_bgr_uint8_of_image_size_and_keeps_unmarked_pixels():
    image = gray_image()
    remove_mask = pixels_within(BRUSH_RADIUS, CENTER)

    preview = mp.render_overlay(image, remove_mask, blank_mask())

    assert preview.shape == image.shape and preview.dtype == np.uint8
    np.testing.assert_array_equal(preview[~remove_mask], image[~remove_mask])
    assert (preview[remove_mask] != image[remove_mask]).any(axis=1).all()


def test_overlay_tints_remove_green_and_protect_red():
    image = gray_image()
    remove_mask, protect_mask = blank_mask(), blank_mask()
    remove_mask[5, 5] = protect_mask[10, 10] = True

    preview = mp.render_overlay(image, remove_mask, protect_mask)

    blue, green, red = (int(v) for v in preview[5, 5])
    assert green > GRAY_LEVEL and blue < GRAY_LEVEL and red < GRAY_LEVEL
    blue, green, red = (int(v) for v in preview[10, 10])
    assert red > GRAY_LEVEL and blue < GRAY_LEVEL and green < GRAY_LEVEL


def test_overlay_draws_protect_on_top_where_masks_overlap():
    image = gray_image()
    both = pixels_within(BRUSH_RADIUS, CENTER)

    overlapping = mp.render_overlay(image, both, both)
    protect_only = mp.render_overlay(image, blank_mask(), both)

    np.testing.assert_array_equal(overlapping, protect_only)


def test_overlay_does_not_modify_image():
    image = gray_image()

    mp.render_overlay(image, pixels_within(BRUSH_RADIUS, CENTER), blank_mask())

    assert (image == GRAY_LEVEL).all()
