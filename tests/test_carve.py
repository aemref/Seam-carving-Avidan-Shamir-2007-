"""Unit tests for seamcarving.carve using stand-in energy and seam functions."""

import numpy as np
import pytest

from seamcarving.carve import (
    ShrinkResult,
    remove_vertical_seam,
    shrink_width,
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


# shrink_width -----------------------------------------------------------------


@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("seam_fn", SEAM_FNS)
def test_shrink_width_shape_dtype(channels, seam_fn):
    h, w, k = 9, 12, 4
    img = make_image(h, w, channels)
    res = shrink_width(img, k, standin_energy, seam_fn)
    assert isinstance(res, ShrinkResult)
    assert res.image.shape == (h, w - k) + img.shape[2:]
    assert res.image.dtype == img.dtype
    assert res.removed.shape == (h, w) and res.removed.dtype == bool
    assert res.order.shape == (h, w) and res.order.dtype == np.int32
    assert isinstance(res.carry, tuple) and res.carry == ()


@pytest.mark.parametrize("seam_fn", SEAM_FNS)
def test_shrink_width_removed_and_order(seam_fn):
    h, w, k = 9, 12, 5
    res = shrink_width(make_image(h, w, 3), k, standin_energy, seam_fn)
    np.testing.assert_array_equal(res.removed.sum(axis=1), np.full(h, k))
    np.testing.assert_array_equal(res.removed, res.order >= 0)
    assert np.all(res.order[~res.removed] == -1)
    for r in range(h):
        np.testing.assert_array_equal(np.sort(res.order[r][res.removed[r]]), np.arange(k))


@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("seam_fn", SEAM_FNS)
def test_shrink_width_consistency_with_carry(channels, seam_fn):
    h, w, k = 9, 12, 5
    img = make_image(h, w, channels)
    carry = make_carry(h, w)
    res = shrink_width(img, k, standin_energy, seam_fn, carry=carry)
    np.testing.assert_array_equal(kept(img, res.removed, h, w - k), res.image)
    assert len(res.carry) == len(carry)
    for orig, cut in zip(carry, res.carry, strict=True):
        assert cut.dtype == orig.dtype
        assert cut.shape == (h, w - k) + orig.shape[2:]
        np.testing.assert_array_equal(kept(orig, res.removed, h, w - k), cut)


def test_zigzag_seam_is_not_straight():
    # Guard: the zigzag removal really is non-degenerate.
    res = shrink_width(make_image(6, 10), 1, standin_energy, zigzag_seam)
    cols = np.argmax(res.removed, axis=1)
    assert len(np.unique(cols)) == 2


def test_order_matches_step_sequence():
    # Straight seams on a known energy: the original column of each step is predictable.
    h, w, k = 4, 6, 3
    img = make_image(h, w)
    res = shrink_width(img, k, lambda im, cr: np.zeros(im.shape[:2]), standin_seam)
    # standin_seam on zero energy always picks current column 0 -> original columns 0, 1, 2.
    for t in range(k):
        assert np.all(res.order[:, t] == t)
    assert np.all(res.order[:, k:] == -1)


def test_shrink_width_inputs_not_mutated():
    h, w = 7, 10
    img = make_image(h, w, 3)
    carry = make_carry(h, w)
    img_before = img.copy()
    carry_before = tuple(a.copy() for a in carry)
    shrink_width(img, 4, standin_energy, zigzag_seam, carry=carry)
    np.testing.assert_array_equal(img, img_before)
    for a, b in zip(carry, carry_before, strict=True):
        np.testing.assert_array_equal(a, b)


def test_energy_fn_sees_current_image_and_carry():
    h, w, k = 6, 9, 4
    img = make_image(h, w, 3)
    carry = make_carry(h, w)
    calls = []

    def recording_energy(im, cr):
        calls.append((im.copy(), tuple(a.copy() for a in cr)))
        return standin_energy(im, cr)

    res = shrink_width(img, k, recording_energy, zigzag_seam, carry=carry)
    assert len(calls) == k  # energy recomputed at every step

    # Replay the steps independently and compare with what energy_fn received.
    cur, cur_carry = img, carry
    for im_seen, carry_seen in calls:
        np.testing.assert_array_equal(im_seen, cur)
        assert len(carry_seen) == len(cur_carry)
        for a, b in zip(carry_seen, cur_carry, strict=True):
            np.testing.assert_array_equal(a, b)
        seam = zigzag_seam(standin_energy(cur, cur_carry))
        cur = remove_vertical_seam(cur, seam)
        cur_carry = tuple(remove_vertical_seam(a, seam) for a in cur_carry)
    np.testing.assert_array_equal(cur, res.image)


def test_shrink_width_k_zero_returns_copies():
    h, w = 5, 7
    img = make_image(h, w, 3)
    carry = make_carry(h, w)
    res = shrink_width(img, 0, standin_energy, standin_seam, carry=carry)
    np.testing.assert_array_equal(res.image, img)
    assert not np.shares_memory(res.image, img)
    for orig, out in zip(carry, res.carry, strict=True):
        np.testing.assert_array_equal(out, orig)
        assert not np.shares_memory(out, orig)
    assert res.removed.shape == (h, w) and not res.removed.any()
    assert res.order.shape == (h, w) and np.all(res.order == -1)


@pytest.mark.parametrize("k", [-1, 7, 8])
def test_shrink_width_invalid_k(k):
    with pytest.raises(ValueError):
        shrink_width(make_image(5, 7), k, standin_energy, standin_seam)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((5, 6)), np.zeros((4, 7)), np.zeros((7, 5)), np.zeros(5)],
)
def test_shrink_width_bad_carry_shape(bad):
    with pytest.raises(ValueError):
        shrink_width(make_image(5, 7), 2, standin_energy, standin_seam, carry=(bad,))


def test_shrink_width_rejects_bad_seam_fn_output():
    img = make_image(5, 7)
    with pytest.raises(ValueError):
        shrink_width(img, 1, standin_energy, lambda e: np.zeros(e.shape[0] - 1, dtype=np.int64))
    with pytest.raises(ValueError):
        shrink_width(img, 1, standin_energy, lambda e: np.full(e.shape[0], e.shape[1]))

