"""Unit tests for seamcarving.enlarge: fake-carve unit tests and real-module tests (5.1)."""

import numpy as np
import pytest

from seamcarving import enlarge
from seamcarving.carve import ShrinkResult
from seamcarving.enlarge import (
    amplify_content,
    enlarge_height,
    enlarge_width,
    insert_vertical_seams,
    inserted_mask,
)


def fake_shrink_width(img, k, energy_fn=None, seam_fn=None, carry=()):
    h, w = img.shape[:2]
    cols = np.arange(k) * (w // max(k, 1))  # k distinct, evenly spaced columns
    removed = np.zeros((h, w), dtype=bool)
    removed[:, cols] = True
    order = np.full((h, w), -1, dtype=np.int32)
    order[:, cols] = np.arange(k)
    image = img[~removed].reshape(h, w - k, *img.shape[2:])
    return ShrinkResult(image=image, carry=(), removed=removed, order=order)


def fake_shrink_height(img, k, energy_fn=None, seam_fn=None, carry=()):
    """Transpose wrapper around fake_shrink_width; removed/order in original (H, W) layout."""
    res = fake_shrink_width(np.swapaxes(img, 0, 1), k, energy_fn, seam_fn, carry)
    return ShrinkResult(
        image=np.swapaxes(res.image, 0, 1),
        carry=(),
        removed=res.removed.T,
        order=res.order.T,
    )


@pytest.fixture
def recorded(monkeypatch):
    """Patch carve.shrink_width with the fake and record every call's k and callbacks."""
    calls = []

    def recording(img, k, energy_fn=None, seam_fn=None, carry=()):
        calls.append({"k": k, "width": img.shape[1], "energy_fn": energy_fn, "seam_fn": seam_fn})
        return fake_shrink_width(img, k, energy_fn, seam_fn, carry)

    monkeypatch.setattr("seamcarving.carve.shrink_width", recording)
    return calls


def random_chosen(rng, h, w, k):
    """bool (H, W) with exactly k random True values per row."""
    chosen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        chosen[r, rng.choice(w, size=k, replace=False)] = True
    return chosen


# ---------------------------------------------------------------- insert_vertical_seams


@pytest.mark.parametrize("tail", [(), (3,)])
def test_insert_shape_and_dtype(tail):
    rng = np.random.default_rng(0)
    h, w, k = 6, 9, 3
    img = rng.integers(0, 256, size=(h, w, *tail), dtype=np.uint8)
    out = insert_vertical_seams(img, random_chosen(rng, h, w, k))
    assert out.shape == (h, w + k, *tail)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("tail", [(), (3,)])
def test_insert_preserves_originals(tail):
    rng = np.random.default_rng(1)
    h, w, k = 7, 10, 4
    img = rng.integers(0, 256, size=(h, w, *tail), dtype=np.uint8)
    chosen = random_chosen(rng, h, w, k)
    out = insert_vertical_seams(img, chosen)
    np.testing.assert_array_equal(out[~inserted_mask(chosen)].reshape(h, w, *tail), img)


def test_insert_hand_computed_including_last_column():
    img = np.array([[10, 20, 30, 40], [5, 7, 9, 11]], dtype=np.uint8)
    chosen = np.zeros((2, 4), dtype=bool)
    chosen[0, 1] = True  # right neighbour: (20 + 30) / 2 = 25
    chosen[1, 3] = True  # last column, left neighbour: (11 + 9) / 2 = 10
    expected = np.array([[10, 20, 25, 30, 40], [5, 7, 9, 11, 10]], dtype=np.uint8)
    np.testing.assert_array_equal(insert_vertical_seams(img, chosen), expected)


def test_insert_two_adjacent_selected_pixels():
    img = np.array([[10, 20, 30, 40]], dtype=np.uint8)
    chosen = np.array([[False, True, True, False]])
    # col 1 -> (20 + 30) / 2 = 25, col 2 -> (30 + 40) / 2 = 35; both use original neighbours
    expected = np.array([[10, 20, 25, 30, 35, 40]], dtype=np.uint8)
    np.testing.assert_array_equal(insert_vertical_seams(img, chosen), expected)


def test_insert_colour_averages_per_channel():
    img = np.array([[[0, 100, 200], [10, 50, 255], [7, 7, 7]]], dtype=np.uint8)
    chosen = np.array([[True, False, False]])
    out = insert_vertical_seams(img, chosen)
    np.testing.assert_array_equal(out[0, 1], [5, 75, 228])  # 227.5 -> 228 (np.rint)


def test_insert_rounding_follows_np_rint():
    # 10.5 -> 10 and 11.5 -> 12 (round half to even); 252.5 -> 252 without uint8 overflow
    img = np.array([[10, 11, 12], [250, 255, 0]], dtype=np.uint8)
    chosen = np.array([[True, True, False], [True, False, False]])
    chosen[1, 2] = True
    out = insert_vertical_seams(img, chosen)
    assert np.rint(10.5) == 10 and np.rint(11.5) == 12
    assert out[0, 1] == np.rint((10 + 11) / 2) == 10
    assert out[0, 3] == np.rint((11 + 12) / 2) == 12
    assert out[1, 1] == np.rint((250 + 255) / 2) == 252
    assert out[1, 4] == 128  # last column: (0 + 255) / 2 = 127.5 -> 128


def test_insert_float_image_not_rounded():
    img = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    out = insert_vertical_seams(img, np.array([[True, False, False]]))
    assert out.dtype == np.float32
    assert out[0, 1] == pytest.approx(10.5)


def test_insert_flat_image_stays_flat():
    rng = np.random.default_rng(2)
    img = np.full((5, 8, 3), (12, 130, 250), dtype=np.uint8)
    out = insert_vertical_seams(img, random_chosen(rng, 5, 8, 3))
    assert out.shape == (5, 11, 3)
    assert np.all(out == img[0, 0])


def test_insert_unequal_counts_raises():
    img = np.zeros((2, 4), dtype=np.uint8)
    chosen = np.array([[True, False, False, False], [True, True, False, False]])
    with pytest.raises(ValueError):
        insert_vertical_seams(img, chosen)


@pytest.mark.parametrize("shape", [(2, 5), (3, 4), (4,)])
def test_insert_wrong_shape_raises(shape):
    img = np.zeros((2, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        insert_vertical_seams(img, np.zeros(shape, dtype=bool))


def test_insert_non_bool_chosen_raises():
    img = np.zeros((2, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        insert_vertical_seams(img, np.zeros((2, 4), dtype=np.int64))


def test_insert_k0_returns_equal_copy():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = insert_vertical_seams(img, np.zeros((3, 4), dtype=bool))
    np.testing.assert_array_equal(out, img)
    assert out is not img
    assert not np.shares_memory(out, img)


def test_insert_does_not_modify_inputs():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(4, 6, 3), dtype=np.uint8)
    chosen = random_chosen(rng, 4, 6, 2)
    img_before, chosen_before = img.copy(), chosen.copy()
    insert_vertical_seams(img, chosen)
    np.testing.assert_array_equal(img, img_before)
    np.testing.assert_array_equal(chosen, chosen_before)


# ---------------------------------------------------------------- inserted_mask


def test_inserted_mask_counts_and_positions():
    chosen = np.array([[True, False, True, True, False], [False, False, False, False, True]])
    chosen[1, 0] = chosen[1, 1] = True
    mask = inserted_mask(chosen)
    assert mask.shape == (2, 8) and mask.dtype == np.bool_
    np.testing.assert_array_equal(mask.sum(axis=1), [3, 3])
    # row 0: c = 0, 2, 3 -> 0+0+1, 2+1+1, 3+2+1 ; row 1: c = 0, 1, 4 -> 1, 3, 7
    np.testing.assert_array_equal(np.flatnonzero(mask[0]), [1, 4, 6])
    np.testing.assert_array_equal(np.flatnonzero(mask[1]), [1, 3, 7])


def test_inserted_mask_matches_inserted_values():
    img = np.array([[0, 10, 20, 30, 40]], dtype=np.uint8)
    chosen = np.array([[True, False, True, False, True]])
    out = insert_vertical_seams(img, chosen)
    np.testing.assert_array_equal(out[inserted_mask(chosen)], [5, 25, 35])


def test_inserted_mask_unequal_counts_raises():
    with pytest.raises(ValueError):
        inserted_mask(np.array([[True, True], [True, False]]))


# ---------------------------------------------------------------- enlarge_width (fake carve)


@pytest.mark.parametrize("tail", [(), (3,)])
def test_enlarge_width_shape_with_fake(recorded, tail):
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(5, 10, *tail), dtype=np.uint8)
    out = enlarge_width(img, 4)
    assert out.shape == (5, 14, *tail)
    assert out.dtype == np.uint8


def test_enlarge_width_single_step(recorded):
    img = np.zeros((3, 10), dtype=np.uint8)
    enlarge_width(img, 4, max_step=0.5)  # floor(0.5 * 10) = 5 >= 4
    assert [c["k"] for c in recorded] == [4]


@pytest.mark.parametrize(
    ("w", "k", "max_step", "steps"),
    [
        (10, 12, 0.5, [5, 7]),  # 10 -> 15 (floor 7.5 = 7) -> 22
        (10, 6, 0.2, [2, 2, 2]),  # floor(2.0), floor(2.4), floor(2.8)
        (10, 3, 0.05, [1, 1, 1]),  # floor(0.5) = 0 -> at least 1
        (8, 10, 0.75, [6, 4]),  # floor(6.0) = 6, then min(4, floor(10.5))
        (10, 10, 1.0, [9, 1]),  # k = W: capped at W - 1, the rest in the next step
        (10, 15, 1.0, [9, 6]),  # k = 1.5 W: 10 -> 19, then min(6, 19, 18)
        (8, 6, 1.0, [6]),  # +75% with max_step=1.0 is still a single step
    ],
)
def test_enlarge_width_multi_step(recorded, w, k, max_step, steps):
    img = np.zeros((4, w, 3), dtype=np.uint8)
    out = enlarge_width(img, k, max_step=max_step)
    assert [c["k"] for c in recorded] == steps
    assert [c["width"] for c in recorded] == list(w + np.cumsum([0] + steps[:-1]))
    assert out.shape == (4, w + k, 3)


def test_enlarge_width_values_with_fake(recorded):
    img = np.array([[0, 10, 20, 30]], dtype=np.uint8)
    out = enlarge_width(img, 2)  # fake removes cols 0 and 2
    np.testing.assert_array_equal(out, [[0, 5, 10, 20, 25, 30]])


def test_enlarge_width_passes_callbacks(recorded):
    def energy_fn(img, carry):
        return np.zeros(img.shape[:2])

    def seam_fn(energy):
        return np.zeros(energy.shape[0], dtype=np.int64)

    enlarge_width(np.zeros((2, 6), dtype=np.uint8), 2, energy_fn, seam_fn)
    assert recorded[0]["energy_fn"] is energy_fn
    assert recorded[0]["seam_fn"] is seam_fn


def test_enlarge_width_k0_returns_copy(recorded):
    img = np.arange(20, dtype=np.uint8).reshape(4, 5)
    out = enlarge_width(img, 0)
    np.testing.assert_array_equal(out, img)
    assert out is not img
    assert recorded == []


def test_enlarge_width_does_not_modify_input(recorded):
    img = np.arange(30, dtype=np.uint8).reshape(3, 10)
    before = img.copy()
    enlarge_width(img, 7)
    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize(("k", "max_step"), [(-1, 0.5), (2, 0.0), (2, -0.5), (2, 1.5)])
def test_enlarge_width_invalid_arguments(recorded, k, max_step):
    with pytest.raises(ValueError):
        enlarge_width(np.zeros((2, 4), dtype=np.uint8), k, max_step=max_step)


# ------------------------------------------------------- enlarge_height / amplify (fake carve)


def test_enlarge_height_matches_transposed_width(recorded):
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=(8, 5, 3), dtype=np.uint8)
    out = enlarge_height(img, 3)
    assert out.shape == (11, 5, 3)
    expected = np.swapaxes(enlarge_width(np.swapaxes(img, 0, 1), 3), 0, 1)
    np.testing.assert_array_equal(out, expected)


def test_amplify_content_shape_with_fakes(monkeypatch):
    calls = []

    def width(img, k, energy_fn=None, seam_fn=None, carry=()):
        calls.append(("width", img.shape[:2], k))
        return fake_shrink_width(img, k, energy_fn, seam_fn, carry)

    def height(img, k, energy_fn=None, seam_fn=None, carry=()):
        calls.append(("height", img.shape[:2], k))
        return fake_shrink_height(img, k, energy_fn, seam_fn, carry)

    monkeypatch.setattr("seamcarving.carve.shrink_width", width)
    monkeypatch.setattr("seamcarving.carve.shrink_height", height)
    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, size=(10, 15, 3), dtype=np.uint8)
    out = amplify_content(img, scale=1.2)  # 12 x 18, then back to 10 x 15
    assert out.shape == img.shape and out.dtype == img.dtype
    assert calls == [("width", (12, 18), 3), ("height", (12, 15), 2)]


def test_amplify_content_scale_one_and_invalid():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = amplify_content(img, scale=1.0)
    np.testing.assert_array_equal(out, img)
    assert out is not img
    with pytest.raises(ValueError):
        amplify_content(img, scale=0.9)


def test_enlarge_uses_carve_module_attribute():
    # enlarge must call carve.shrink_width at call time so tests can monkeypatch it
    assert enlarge.carve.__name__ == "seamcarving.carve"


# ------------------------------------------------ real carve/energy/seam modules (Dalga 2b, 5.1)

BG = 128


def textured_square_image(h, w, top, left, size, channels=3, seed=0):
    """Flat background with a random-texture square (high energy inside and on its border)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, channels), BG, dtype=np.uint8)
    img[top : top + size, left : left + size] = rng.integers(160, 256, size=(size, size, channels))
    return img


@pytest.mark.parametrize("tail", [(), (3,)])
def test_real_enlarge_width_shape(tail):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(12, 16, *tail), dtype=np.uint8)
    out = enlarge_width(img, 5)
    assert out.shape == (12, 21, *tail)
    assert out.dtype == np.uint8


def test_real_enlarge_width_multi_step_shape():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(10, 12, 3), dtype=np.uint8)
    out = enlarge_width(img, 15, max_step=0.5)  # steps 6, 9
    assert out.shape == (10, 27, 3)


@pytest.mark.parametrize("tail", [(), (3,)])
def test_real_enlarge_height_shape(tail):
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(14, 10, *tail), dtype=np.uint8)
    out = enlarge_height(img, 4)
    assert out.shape == (18, 10, *tail)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("tail", [(), (3,)])
def test_real_amplify_content_same_shape(tail):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(15, 20, *tail), dtype=np.uint8)
    out = amplify_content(img, scale=1.2)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_real_square_keeps_width_after_enlarge_width():
    # 30 columns, square of width 6 at cols 12..17; about 2 columns on each side of the square
    # carry gradient energy, leaving ~20 zero-energy background columns for 6 seams.
    size = 6
    img = textured_square_image(20, 30, top=7, left=12, size=size)
    out = enlarge_width(img, 6)
    assert out.shape == (20, 36, 3)
    non_bg_cols = np.flatnonzero(np.any(out != BG, axis=(0, 2)))
    assert non_bg_cols.size == size
    assert np.all(np.diff(non_bg_cols) == 1)  # square is still contiguous
    c0 = non_bg_cols[0]
    np.testing.assert_array_equal(
        out[7 : 7 + size, c0 : c0 + size], img[7 : 7 + size, 12 : 12 + size]
    )


def test_real_square_keeps_height_after_enlarge_height():
    size = 6
    img = textured_square_image(30, 20, top=12, left=7, size=size)
    out = enlarge_height(img, 6)
    assert out.shape == (36, 20, 3)
    non_bg_rows = np.flatnonzero(np.any(out != BG, axis=(1, 2)))
    assert non_bg_rows.size == size
    r0 = non_bg_rows[0]
    np.testing.assert_array_equal(
        out[r0 : r0 + size, 7 : 7 + size], img[12 : 12 + size, 7 : 7 + size]
    )


@pytest.mark.parametrize("k_factor", [1.0, 1.5])
def test_real_enlarge_width_max_step_one_large_k(k_factor):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(10, 12, 3), dtype=np.uint8)
    k = int(k_factor * 12)
    out = enlarge_width(img, k, max_step=1.0)
    assert out.shape == (10, 12 + k, 3)


def test_real_enlarge_height_max_step_one_k_equals_h():
    img = np.random.default_rng(1).integers(0, 256, size=(8, 10), dtype=np.uint8)
    assert enlarge_height(img, 8, max_step=1.0).shape == (16, 10)


@pytest.mark.parametrize("max_step", [0.5, 1.0])
def test_single_column_or_row_raises_clear_error(max_step):
    col = np.zeros((5, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="1 pixel wide"):
        enlarge_width(col, 3, max_step=max_step)
    row = np.zeros((1, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="1 pixel high"):
        enlarge_height(row, 3, max_step=max_step)
    np.testing.assert_array_equal(enlarge_width(col, 0), col)  # k = 0 is still a plain copy
