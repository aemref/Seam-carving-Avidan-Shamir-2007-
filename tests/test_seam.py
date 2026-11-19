"""Tests for the dynamic-programming seam search in ``seamcarving.seam``."""

import numpy as np
import pytest

from seamcarving.seam import cumulative_map, find_vertical_seam, is_valid_seam, seam_cost

# Hand-computed example. Worked by hand:
#   M row 1: 2+min(inf,1,4)=3   8+min(1,4,3)=9    1+min(4,3,5)=4    6+min(3,5,inf)=9
#   M row 2: 7+min(inf,3,9)=10  3+min(3,9,4)=6    9+min(9,4,9)=13   2+min(4,9,inf)=6
#   M row 3: 4+min(inf,10,6)=10 6+min(10,6,13)=12 5+min(6,13,6)=11  1+min(13,6,inf)=7
# Row 3, column 2 has a tie between offsets -1 and +1; the leftmost (-1) must win.
# Backtracking from M[3, 3] = 7 gives the seam [2, 2, 3, 3] with cost 3 + 1 + 2 + 1 = 7.
HAND_ENERGY = np.array(
    [
        [1, 4, 3, 5],
        [2, 8, 1, 6],
        [7, 3, 9, 2],
        [4, 6, 5, 1],
    ],
    dtype=np.float64,
)
HAND_M = np.array(
    [
        [1, 4, 3, 5],
        [3, 9, 4, 9],
        [10, 6, 13, 6],
        [10, 12, 11, 7],
    ],
    dtype=np.float64,
)
HAND_BACK = np.array(
    [
        [0, 0, 0, 0],
        [0, -1, 0, -1],
        [0, -1, 0, -1],
        [1, 0, -1, 0],
    ],
    dtype=np.int8,
)


def test_cumulative_map_hand_example():
    M, back = cumulative_map(HAND_ENERGY)
    assert M.dtype == np.float64 and M.shape == (4, 4)
    assert back.dtype == np.int8 and back.shape == (4, 4)
    np.testing.assert_array_equal(M, HAND_M)
    np.testing.assert_array_equal(back, HAND_BACK)


@pytest.mark.parametrize(
    "make_input",
    [
        lambda e: e.copy(),
        lambda e: e.astype(np.int64),
        lambda e: e.astype(np.uint8),
        lambda e: e.astype(np.float32),
        lambda e: np.ascontiguousarray(e.T).T,  # non-contiguous (Fortran-order) view
        lambda e: np.pad(e, 1)[1:-1, 1:-1],  # strided slice of a bigger array
    ],
    ids=["float64", "int64", "uint8", "float32", "transposed", "slice"],
)
def test_cumulative_map_accepts_any_layout_and_leaves_input_unchanged(make_input):
    energy = make_input(HAND_ENERGY)
    before = energy.copy()
    M, back = cumulative_map(energy)
    np.testing.assert_array_equal(energy, before)
    assert energy.dtype == before.dtype
    np.testing.assert_array_equal(M, HAND_M)
    np.testing.assert_array_equal(back, HAND_BACK)


def test_cumulative_map_read_only_input():
    energy = HAND_ENERGY.copy()
    energy.flags.writeable = False
    M, _ = cumulative_map(energy)
    np.testing.assert_array_equal(M, HAND_M)


def test_cumulative_map_recurrence_on_random_map():
    rng = np.random.default_rng(0)
    energy = rng.normal(size=(20, 15)) * 1e6  # negative and large values
    M, back = cumulative_map(energy)
    assert np.isfinite(M).all()
    np.testing.assert_array_equal(M[0], energy[0])
    assert (back[0] == 0).all()
    assert set(np.unique(back)) <= {-1, 0, 1}
    assert back[1:, 0].min() >= 0 and back[1:, -1].max() <= 0  # never point outside
    for r in range(1, 20):
        parent = np.arange(15) + back[r].astype(np.int64)
        np.testing.assert_array_equal(M[r], energy[r] + M[r - 1, parent])


@pytest.mark.parametrize("shape", [(5, 1), (5, 2), (1, 6), (1, 1)])
def test_cumulative_map_edge_shapes(shape):
    rng = np.random.default_rng(0)
    energy = rng.integers(-9, 10, size=shape).astype(np.float64)
    M, back = cumulative_map(energy)
    assert M.shape == shape and back.shape == shape
    assert np.isfinite(M).all()  # the +inf padding never leaks into M
    if shape[1] == 1:
        np.testing.assert_array_equal(M[:, 0], np.cumsum(energy[:, 0]))
        assert (back == 0).all()
    if shape[0] == 1:
        np.testing.assert_array_equal(M, energy)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(5),
        np.zeros((0, 4)),
        np.zeros((3, 0)),
        np.zeros((2, 2, 2)),
        np.array([[0.0, np.inf]]),
    ],
    ids=["1d", "no-rows", "no-cols", "3d", "inf"],
)
def test_cumulative_map_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        cumulative_map(bad)


# --- find_vertical_seam -------------------------------------------------------------------


def test_find_vertical_seam_hand_example():
    energy = HAND_ENERGY.copy()
    seam = find_vertical_seam(energy)
    assert seam.dtype == np.int64 and seam.shape == (4,)
    np.testing.assert_array_equal(seam, [2, 2, 3, 3])
    np.testing.assert_array_equal(energy, HAND_ENERGY)  # input unchanged


def test_find_vertical_seam_constant_map_is_column_zero():
    for value in (0.0, 7.5, -3.0):
        seam = find_vertical_seam(np.full((6, 9), value))
        np.testing.assert_array_equal(seam, np.zeros(6, dtype=np.int64))


def test_find_vertical_seam_parent_ties_go_left():
    # Last row: column 1 is the unique minimum. Its parents tie between offsets -1 and 0
    # (first map) and between 0 and +1 (second map); the leftmost parent must win.
    np.testing.assert_array_equal(find_vertical_seam([[0, 0, 5], [9, 0, 9]]), [0, 1])
    np.testing.assert_array_equal(find_vertical_seam([[5, 0, 0], [9, 0, 9]]), [1, 1])


def test_find_vertical_seam_shift_invariant():
    rng = np.random.default_rng(0)
    for _ in range(20):
        energy = rng.integers(0, 50, size=(12, 10)).astype(np.float64)
        seam = find_vertical_seam(energy)
        for shift in (-1000.0, -1.0, 1.0, 1e6):
            np.testing.assert_array_equal(find_vertical_seam(energy + shift), seam)


def test_find_vertical_seam_goes_through_large_negative_region():
    rng = np.random.default_rng(0)
    energy = rng.uniform(0, 255, size=(30, 40))
    energy[10:20, 31:33] = -1e7  # object to remove
    seam = find_vertical_seam(energy)
    assert np.all((seam[10:20] >= 31) & (seam[10:20] <= 32))


def test_find_vertical_seam_avoids_large_positive_region():
    # Mask weights as in object removal: -R inside the removal mask, +P on the protected
    # region, of order H**2 * 1000. Every seam that reaches the removal region from the
    # top must not cross the protected band next to it.
    rng = np.random.default_rng(0)
    n_rows = 40
    weight = n_rows**2 * 1000.0
    energy = rng.uniform(0, 255, size=(n_rows, 50))
    energy[:, 20:25] += weight  # protected vertical band
    energy[15:25, 30:33] -= weight  # removal region right of the band
    M, _ = cumulative_map(energy)
    assert np.isfinite(M).all()
    seam = find_vertical_seam(energy)
    assert np.all((seam < 20) | (seam >= 25))  # never touches the protected band
    assert np.all((seam[15:25] >= 30) & (seam[15:25] <= 32))


def test_find_vertical_seam_wide_map_column_above_int8_range():
    # The cheapest path sits at column 250; backtracking adds int8 offsets to it.
    energy = np.ones((20, 300))
    path = 250 + np.round(3 * np.sin(np.arange(20) / 3)).astype(np.int64)
    energy[np.arange(20), path] = 0.0
    np.testing.assert_array_equal(find_vertical_seam(energy), path)


@pytest.mark.parametrize("shape", [(5, 1), (5, 2), (1, 6), (1, 1)])
def test_find_vertical_seam_edge_shapes(shape):
    rng = np.random.default_rng(0)
    energy = rng.integers(-9, 10, size=shape).astype(np.float64)
    seam = find_vertical_seam(energy)
    assert seam.dtype == np.int64 and seam.shape == (shape[0],)
    assert np.all((seam >= 0) & (seam < shape[1]))
    assert np.all(np.abs(np.diff(seam)) <= 1)
    if shape[1] == 1:
        np.testing.assert_array_equal(seam, np.zeros(shape[0], dtype=np.int64))
    if shape[0] == 1:
        assert seam[0] == np.argmin(energy[0])


def test_find_vertical_seam_two_columns():
    np.testing.assert_array_equal(find_vertical_seam([[1, 0], [0, 1], [1, 0]]), [1, 0, 1])


def test_find_vertical_seam_transposed_view_matches_copy():
    # shrink_height passes a transposed view of the energy.
    rng = np.random.default_rng(0)
    energy_t = rng.normal(size=(25, 18))
    view = energy_t.T
    assert not view.flags.c_contiguous
    np.testing.assert_array_equal(find_vertical_seam(view), find_vertical_seam(view.copy()))


# --- seam_cost and is_valid_seam ----------------------------------------------------------


def test_seam_properties_and_cost_on_random_maps():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n_rows, n_cols = rng.integers(1, 30, size=2)
        integer_map = rng.integers(-100, 100, size=(n_rows, n_cols)).astype(np.float64)
        float_map = rng.normal(size=(n_rows, n_cols)) * 1e3
        for energy in (integer_map, float_map):
            seam = find_vertical_seam(energy)
            assert seam.dtype == np.int64 and seam.shape == (n_rows,)
            assert is_valid_seam(seam, n_cols) is True
            cost = seam_cost(energy, seam)
            assert type(cost) is float
            M, _ = cumulative_map(energy)
            assert cost == pytest.approx(M[-1].min(), rel=1e-12, abs=1e-9)
        # Integer-valued maps are summed exactly, so the match is exact.
        assert (
            seam_cost(integer_map, find_vertical_seam(integer_map))
            == cumulative_map(integer_map)[0][-1].min()
        )


def test_seam_cost_hand_example():
    assert seam_cost(HAND_ENERGY, np.array([2, 2, 3, 3])) == 7.0
    assert seam_cost(HAND_ENERGY.astype(np.int32), [0, 0, 1, 2]) == 1 + 2 + 3 + 5


@pytest.mark.parametrize(
    "seam",
    [[2, 2, 3], [2, 2, 3, 3, 3], [2, 2, 3, 5], [0, 2, 3, 3], [-1, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]],
    ids=["too-short", "too-long", "outside", "jump", "negative", "float"],
)
def test_seam_cost_rejects_invalid_seam(seam):
    with pytest.raises(ValueError):
        seam_cost(HAND_ENERGY, seam)


@pytest.mark.parametrize(
    ("seam", "width"),
    [
        ([0, 1, 2, 1, 0], 3),
        ([4, 4, 4], 5),
        ([0], 1),
        (np.array([3, 2, 3], dtype=np.int8), 4),
        (np.array([3, 2, 3], dtype=np.uint16), 4),  # unsigned diff 2 - 3 must not wrap
        (np.array([1, 2]), np.int64(3)),
    ],
)
def test_is_valid_seam_accepts(seam, width):
    assert is_valid_seam(seam, width) is True


@pytest.mark.parametrize(
    ("seam", "width"),
    [
        ([0, 2], 3),  # jump of two columns
        ([1, 0], 1),  # outside on the right
        ([0, -1], 3),  # negative column
        ([3, 3], 3),  # column == width
        (np.array([2, 0], dtype=np.uint8), 3),  # unsigned jump of two columns
        ([], 3),  # empty
        ([[0, 1]], 3),  # 2-D
        ([0.0, 1.0], 3),  # float dtype
        ([True, False], 3),  # bool dtype
        ([0, 1], 0),  # width 0
        ([0, 1], 2.0),  # width not an integer
        ("ab", 3),  # not numeric
        ([[0, 1], [0]], 3),  # ragged
        (np.array([0, 2**64 - 1], dtype=np.uint64), 3),  # huge unsigned value
    ],
)
def test_is_valid_seam_rejects_without_raising(seam, width):
    assert is_valid_seam(seam, width) is False
