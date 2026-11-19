"""Tests for the dynamic-programming seam search in ``seamcarving.seam``."""

import numpy as np
import pytest

from seamcarving.seam import cumulative_map

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
