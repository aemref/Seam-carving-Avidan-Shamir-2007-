"""Unit tests for seamcarving.baselines; energy is always passed explicitly."""

import cv2
import numpy as np
import pytest

from seamcarving.baselines import (
    center_crop,
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
