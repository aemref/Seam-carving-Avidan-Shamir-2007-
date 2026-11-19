"""Unit tests for seamcarving.baselines; energy is always passed explicitly."""

import cv2
import numpy as np
import pytest

from seamcarving.baselines import (
    best_crop,
    center_crop,
    remove_columns,
    scale,
)


def make_image(h=12, w=16, channels=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (h, w) if channels is None else (h, w, channels)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def window_energy(energy, r0, c0, new_h, new_w):
    return energy[r0:r0 + new_h, c0:c0 + new_w].sum()


# scale --------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("new_size", [(6, 8), (20, 30), (6, 30), (12, 16)])
def test_scale_shape_and_dtype(channels, new_size):
    img = make_image(12, 16, channels)
    out = scale(img, *new_size)
    assert out.shape == new_size + img.shape[2:]
    assert out.dtype == img.dtype


@pytest.mark.parametrize(
    ("new_size", "flag"),
    [((6, 8), cv2.INTER_AREA), ((24, 32), cv2.INTER_CUBIC), ((6, 32), cv2.INTER_CUBIC)],
)
def test_scale_default_interpolation(new_size, flag):
    img = make_image(12, 16, 3)
    expected = cv2.resize(img, (new_size[1], new_size[0]), interpolation=flag)
    np.testing.assert_array_equal(scale(img, *new_size), expected)


def test_scale_explicit_interpolation_and_no_mutation():
    img = make_image(12, 16, 3)
    before = img.copy()
    out = scale(img, 6, 8, interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(out, cv2.resize(img, (8, 6), interpolation=cv2.INTER_NEAREST))
    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize("new_size", [(0, 8), (6, 0), (-1, 8)])
def test_scale_invalid_size(new_size):
    with pytest.raises(ValueError):
        scale(make_image(), *new_size)


# center_crop --------------------------------------------------------------------


@pytest.mark.parametrize("channels", [None, 3])
def test_center_crop(channels):
    img = make_image(12, 16, channels)
    out = center_crop(img, 6, 9)
    assert out.shape == (6, 9) + img.shape[2:]
    assert out.dtype == img.dtype
    np.testing.assert_array_equal(out, img[3:9, 3:12])
    assert not np.shares_memory(out, img)


@pytest.mark.parametrize("new_size", [(13, 16), (12, 17), (0, 5), (5, -2)])
def test_center_crop_invalid(new_size):
    with pytest.raises(ValueError):
        center_crop(make_image(12, 16), *new_size)


# best_crop ----------------------------------------------------------------------


@pytest.mark.parametrize("channels", [None, 3])
def test_best_crop_contains_high_energy_block(channels):
    h, w, new_h, new_w = 12, 16, 6, 7
    img = make_image(h, w, channels)
    rng = np.random.default_rng(1)
    energy = rng.random((h, w))
    energy[1:5, 10:15] += 100.0  # off-center block that fits in the window
    out = best_crop(img, new_h, new_w, energy=energy)
    assert out.shape == (new_h, new_w) + img.shape[2:]
    assert out.dtype == img.dtype

    # Locate the crop in the image, then check it holds the block.
    matches = [
        (r, c)
        for r in range(h - new_h + 1)
        for c in range(w - new_w + 1)
        if np.array_equal(img[r:r + new_h, c:c + new_w], out)
    ]
    assert len(matches) == 1
    r0, c0 = matches[0]
    assert r0 <= 1 and r0 + new_h >= 5 and c0 <= 10 and c0 + new_w >= 15

    center = window_energy(energy, (h - new_h) // 2, (w - new_w) // 2, new_h, new_w)
    assert window_energy(energy, r0, c0, new_h, new_w) >= center


def test_best_crop_matches_brute_force():
    h, w, new_h, new_w = 9, 11, 4, 5
    img = np.arange(h * w).reshape(h, w)
    energy = np.random.default_rng(2).random((h, w))
    best = max(
        ((r, c) for r in range(h - new_h + 1) for c in range(w - new_w + 1)),
        key=lambda rc: window_energy(energy, *rc, new_h, new_w),
    )
    np.testing.assert_array_equal(
        best_crop(img, new_h, new_w, energy=energy),
        img[best[0]:best[0] + new_h, best[1]:best[1] + new_w],
    )


def test_best_crop_tie_takes_first_window():
    img = np.arange(6 * 8).reshape(6, 8)
    np.testing.assert_array_equal(best_crop(img, 3, 4, energy=np.ones((6, 8))), img[:3, :4])


def test_best_crop_full_size_and_invalid():
    img = make_image(5, 6)
    energy = np.ones((5, 6))
    np.testing.assert_array_equal(best_crop(img, 5, 6, energy=energy), img)
    with pytest.raises(ValueError):
        best_crop(img, 6, 6, energy=energy)
    with pytest.raises(ValueError):
        best_crop(img, 3, 3, energy=np.ones((5, 5)))


# remove_columns -----------------------------------------------------------------


@pytest.mark.parametrize("channels", [None, 3])
def test_remove_columns_removes_lowest_energy_columns(channels):
    h, w = 5, 8
    img = make_image(h, w, channels)
    col_energy = np.array([5.0, 1.0, 7.0, 0.5, 9.0, 2.0, 8.0, 6.0])
    energy = np.tile(col_energy, (h, 1))
    out = remove_columns(img, 3, energy=energy)
    assert out.shape == (h, w - 3) + img.shape[2:]
    assert out.dtype == img.dtype
    # Lowest sums: columns 3, 1, 5; the rest stay in order.
    np.testing.assert_array_equal(out, img[:, [0, 2, 4, 6, 7]])


def test_remove_columns_tie_and_k_zero():
    img = make_image(4, 6)
    energy = np.ones((4, 6))
    np.testing.assert_array_equal(remove_columns(img, 2, energy=energy), img[:, 2:])
    out = remove_columns(img, 0, energy=energy)
    np.testing.assert_array_equal(out, img)
    assert not np.shares_memory(out, img)


@pytest.mark.parametrize("k", [-1, 6, 7])
def test_remove_columns_invalid_k(k):
    with pytest.raises(ValueError):
        remove_columns(make_image(4, 6), k, energy=np.ones((4, 6)))


def test_remove_columns_bad_energy_shape():
    with pytest.raises(ValueError):
        remove_columns(make_image(4, 6), 1, energy=np.ones((4, 5)))
