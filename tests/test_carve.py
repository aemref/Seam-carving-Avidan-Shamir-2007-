"""Unit tests for seamcarving.carve using stand-in energy and seam functions."""

import numpy as np
import pytest

from seamcarving.carve import (
    remove_vertical_seam,
)


def standin_energy(img, carry):
    g = img.astype(np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)
    return np.abs(np.diff(g, axis=1, append=g[:, -1:]))


def standin_seam(energy):
    """Lowest-energy straight column: a valid, degenerate seam."""
    return np.full(energy.shape[0], int(energy.sum(axis=0).argmin()), dtype=np.int64)


def zigzag_seam(energy):
    h, w = energy.shape
    return (np.arange(h) % 2) + min(w - 2, w // 2)


SEAM_FNS = [standin_seam, zigzag_seam]


def make_image(h=9, w=12, channels=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (h, w) if channels is None else (h, w, channels)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def make_carry(h, w, seed=1):
    rng = np.random.default_rng(seed)
    return (
        rng.random((h, w)),
        rng.integers(0, 2, size=(h, w)).astype(bool),
        rng.integers(0, 1000, size=(h, w, 2), dtype=np.int32),
    )


def kept(arr, removed, h_out, w_out):
    """Pixels of ``arr`` not marked in ``removed``, in row-major order."""
    return arr[~removed].reshape((h_out, w_out) + arr.shape[2:])


# remove_vertical_seam ---------------------------------------------------------


@pytest.mark.parametrize("channels", [None, 3])
def test_remove_vertical_seam_shape_dtype_values(channels):
    img = make_image(5, 6, channels)
    seam = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    out = remove_vertical_seam(img, seam)
    assert out.shape == (5, 5) + img.shape[2:]
    assert out.dtype == img.dtype
    for r in range(5):
        np.testing.assert_array_equal(out[r], np.delete(img[r], seam[r], axis=0))


def test_remove_vertical_seam_does_not_mutate():
    img = make_image(4, 5, 3)
    before = img.copy()
    seam = np.array([1, 2, 1, 0])
    seam_before = seam.copy()
    remove_vertical_seam(img, seam)
    np.testing.assert_array_equal(img, before)
    np.testing.assert_array_equal(seam, seam_before)


@pytest.mark.parametrize(
    "seam",
    [
        np.array([0, 1, 2]),  # too short
        np.array([0, 1, 2, 3, 4]),  # too long
        np.array([0, 1, 2, 6]),  # column == W
        np.array([0, -1, 2, 3]),  # negative column
        np.array([0.0, 1.0, 2.0, 3.0]),  # not integer
    ],
)
def test_remove_vertical_seam_invalid(seam):
    img = make_image(4, 6)
    with pytest.raises(ValueError):
        remove_vertical_seam(img, seam)

