"""Integration tests: real energy, seam search, shrinking and enlarging working together."""

import numpy as np
import pytest

from seamcarving.carve import shrink_height, shrink_width
from seamcarving.energy import compute_energy
from seamcarving.enlarge import enlarge_width, inserted_mask
from seamcarving.seam import find_vertical_seam, is_valid_seam


def make_image(h, w, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    shape = (h, w) if channels is None else (h, w, channels)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def square_scene(h=30, w=50, r0=10, c0=20, side=10, seed=0):
    """Flat grey background with a high-energy random-texture square; returns (img, mask)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    img[r0:r0 + side, c0:c0 + side] = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
    square = np.zeros((h, w), dtype=bool)
    square[r0:r0 + side, c0:c0 + side] = True
    return img, square


def test_square_survives_width_shrink():
    h, w, side = 30, 50, 10
    img, square = square_scene(h, w, side=side)
    k = 30  # fewer than the w - side = 40 background columns
    res = shrink_width(img, k)
    assert res.image.shape == (h, w - k, 3)
    assert not (res.removed & square).any()
    # The square appears unchanged, as one contiguous block, in the output.
    kept_square = img[square].reshape(side, side, 3)
    out_cols = np.cumsum(~res.removed, axis=1) - 1  # output column of each kept pixel
    c_out = out_cols[10, 20]
    np.testing.assert_array_equal(res.image[10:10 + side, c_out:c_out + side], kept_square)


def test_square_survives_height_shrink():
    img, square = square_scene(50, 30, r0=20, c0=10, side=10, seed=1)
    res = shrink_height(img, 30)
    assert res.image.shape == (20, 30, 3)
    assert not (res.removed & square).any()


@pytest.mark.parametrize("channels", [None, 3])
def test_default_shrink_width(channels):
    h, w, k = 16, 20, 6
    img = make_image(h, w, channels)
    carry = (np.random.default_rng(1).random((h, w)),)
    res = shrink_width(img, k, carry=carry)
    assert res.image.shape == (h, w - k) + img.shape[2:]
    assert res.image.dtype == img.dtype
    np.testing.assert_array_equal(res.removed.sum(axis=1), np.full(h, k))
    np.testing.assert_array_equal(res.removed, res.order >= 0)
    for r in range(h):
        np.testing.assert_array_equal(np.sort(res.order[r][res.removed[r]]), np.arange(k))
    # Each step removed a connected seam: the first seam is a valid seam in original coords.
    first = np.argmax(res.order == 0, axis=1)
    assert is_valid_seam(first, w)
    np.testing.assert_array_equal(
        img[~res.removed].reshape((h, w - k) + img.shape[2:]), res.image
    )
    np.testing.assert_array_equal(carry[0][~res.removed].reshape(h, w - k), res.carry[0])


def test_default_first_seam_matches_find_vertical_seam():
    img = make_image(12, 15)
    res = shrink_width(img, 1)
    expected = find_vertical_seam(compute_energy(img))
    np.testing.assert_array_equal(np.argmax(res.removed, axis=1), expected)


@pytest.mark.parametrize("channels", [None, 3])
def test_default_shrink_height(channels):
    h, w, k = 20, 16, 6
    img = make_image(h, w, channels, seed=2)
    carry = (np.random.default_rng(3).integers(0, 2, size=(h, w)).astype(bool),)
    res = shrink_height(img, k, carry=carry)
    assert res.image.shape == (h - k, w) + img.shape[2:]
    assert res.image.dtype == img.dtype
    np.testing.assert_array_equal(res.removed.sum(axis=0), np.full(w, k))
    np.testing.assert_array_equal(res.removed, res.order >= 0)
    for c in range(w):
        np.testing.assert_array_equal(img[~res.removed[:, c], c], res.image[:, c])
        np.testing.assert_array_equal(carry[0][~res.removed[:, c], c], res.carry[0][:, c])


@pytest.mark.parametrize("channels", [None, 3])
def test_enlarge_width_keeps_original_pixels(channels):
    h, w, k = 14, 18, 5  # k <= w / 2: a single insertion step
    img = make_image(h, w, channels, seed=4)
    out = enlarge_width(img, k)
    assert out.shape == (h, w + k) + img.shape[2:]
    assert out.dtype == img.dtype
    chosen = shrink_width(img, k).removed
    mask = inserted_mask(chosen)
    assert mask.shape == (h, w + k)
    np.testing.assert_array_equal(mask.sum(axis=1), np.full(h, k))
    np.testing.assert_array_equal(out[~mask].reshape(img.shape), img)


def test_enlarge_width_multi_step_size():
    img = make_image(10, 12, seed=5)
    out = enlarge_width(img, 12)  # +100 %: needs more than one step
    assert out.shape == (10, 24, 3)
    assert out.dtype == np.uint8
