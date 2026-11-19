"""Unit tests for seamcarving.enlarge (Dalga 1: no real carve module needed)."""

import numpy as np
import pytest

from seamcarving.carve import ShrinkResult
from seamcarving.enlarge import (
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
