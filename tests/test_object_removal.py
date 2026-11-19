import itertools
import re
from collections.abc import Iterator

import numpy as np
import pytest

from seamcarving.object_removal import apply_mask_weights, seams_needed

# Small hand-built energy map with negative values; max|e| = 5.
ENERGY = np.array(
    [
        [3.0, -2.0, 5.0, 1.0, -4.0],
        [-1.0, 4.0, -3.0, 2.0, 0.0],
        [2.0, -5.0, 1.0, -2.0, 3.0],
        [0.0, 2.0, -1.0, 4.0, -3.0],
    ]
)
# Arranged so that vertical seams can cross anywhere from 0 to 4 removal pixels.
REMOVAL_PIXELS = ((0, 1), (1, 1), (1, 2), (2, 2), (3, 1))
# Next to the removal mask, so a seam can collect removal pixels on its way through it.
PROTECTED_PIXEL = (2, 3)
OTHER_PROTECTED_PIXEL = (0, 4)
WRONG_SHAPE = (3, 5)


def mask_from_pixels(pixels: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros(ENERGY.shape, dtype=bool)
    for r, c in pixels:
        mask[r, c] = True
    return mask


def all_vertical_seams(n_rows: int, n_cols: int) -> Iterator[tuple[int, ...]]:
    """Yield every seam as a tuple of column indices, one per row, with |c[r] - c[r-1]| <= 1."""

    def extend(seam: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if len(seam) == n_rows:
            yield seam
            return
        for c in range(max(seam[-1] - 1, 0), min(seam[-1] + 2, n_cols)):
            yield from extend(seam + (c,))

    for start_col in range(n_cols):
        yield from extend((start_col,))


def seam_cost(energy: np.ndarray, seam: tuple[int, ...]) -> float:
    return float(sum(energy[r, c] for r, c in enumerate(seam)))


def seam_hits(mask: np.ndarray, seam: tuple[int, ...]) -> int:
    return sum(bool(mask[r, c]) for r, c in enumerate(seam))


def costs_by_hit_count(weighted: np.ndarray, mask: np.ndarray) -> dict[int, list[float]]:
    costs: dict[int, list[float]] = {}
    for seam in all_vertical_seams(*weighted.shape):
        costs.setdefault(seam_hits(mask, seam), []).append(seam_cost(weighted, seam))
    return costs


def test_removal_pixels_become_strictly_smallest_values():
    remove_mask = mask_from_pixels(REMOVAL_PIXELS)

    weighted = apply_mask_weights(ENERGY, remove_mask=remove_mask)

    assert weighted[remove_mask].max() < weighted[~remove_mask].min()


def test_protected_pixels_become_strictly_largest_values():
    remove_mask = mask_from_pixels(REMOVAL_PIXELS)
    protect_mask = mask_from_pixels((PROTECTED_PIXEL, OTHER_PROTECTED_PIXEL))

    weighted = apply_mask_weights(ENERGY, remove_mask, protect_mask)

    assert weighted[protect_mask].min() > weighted[~protect_mask].max()


def test_values_outside_masks_are_unchanged():
    remove_mask = mask_from_pixels(REMOVAL_PIXELS)
    protect_mask = mask_from_pixels((PROTECTED_PIXEL,))
    unmasked = ~(remove_mask | protect_mask)

    weighted = apply_mask_weights(ENERGY, remove_mask, protect_mask)

    np.testing.assert_array_equal(weighted[unmasked], ENERGY[unmasked])


def test_input_energy_is_not_modified():
    energy = ENERGY.copy()

    apply_mask_weights(
        energy, mask_from_pixels(REMOVAL_PIXELS), mask_from_pixels((PROTECTED_PIXEL,))
    )

    np.testing.assert_array_equal(energy, ENERGY)


def test_output_is_finite_float64_even_for_integer_energy():
    integer_energy = ENERGY.astype(np.int64)

    weighted = apply_mask_weights(
        integer_energy, mask_from_pixels(REMOVAL_PIXELS), mask_from_pixels((PROTECTED_PIXEL,))
    )

    assert weighted.dtype == np.float64
    assert np.all(np.isfinite(weighted))


def test_flat_zero_energy_still_gets_strictly_signed_weights():
    flat_energy = np.zeros(ENERGY.shape)
    remove_mask = mask_from_pixels(REMOVAL_PIXELS)
    protect_mask = mask_from_pixels((PROTECTED_PIXEL,))

    weighted = apply_mask_weights(flat_energy, remove_mask, protect_mask)

    assert np.all(weighted[remove_mask] < 0)
    assert np.all(weighted[protect_mask] > 0)


def test_seam_through_more_removal_pixels_is_always_cheaper():
    remove_mask = mask_from_pixels(REMOVAL_PIXELS)
    weighted = apply_mask_weights(ENERGY, remove_mask=remove_mask)

    costs = costs_by_hit_count(weighted, remove_mask)

    hit_counts = sorted(costs)
    assert hit_counts == [0, 1, 2, 3, 4]
    for fewer, more in itertools.pairwise(hit_counts):
        assert max(costs[more]) < min(costs[fewer])


@pytest.mark.parametrize("removal_pixels", [(), REMOVAL_PIXELS], ids=["alone", "with_removal"])
def test_seam_through_protected_pixel_is_always_costlier(removal_pixels):
    remove_mask = mask_from_pixels(removal_pixels)
    protect_mask = mask_from_pixels((PROTECTED_PIXEL,))
    weighted = apply_mask_weights(ENERGY, remove_mask, protect_mask)

    costs = costs_by_hit_count(weighted, protect_mask)

    assert min(costs[1]) > max(costs[0])


def test_overlapping_pixel_is_protected():
    remove_mask = mask_from_pixels(REMOVAL_PIXELS)
    overlap_pixel = REMOVAL_PIXELS[0]
    protect_mask = mask_from_pixels((overlap_pixel, PROTECTED_PIXEL))

    weighted = apply_mask_weights(ENERGY, remove_mask, protect_mask)

    assert weighted[overlap_pixel] == weighted[PROTECTED_PIXEL]


@pytest.mark.parametrize("mask_argument", ["remove_mask", "protect_mask"])
def test_mask_shape_mismatch_raises_value_error_naming_both_shapes(mask_argument):
    wrong_mask = np.zeros(WRONG_SHAPE, dtype=bool)
    expected_message = re.escape(f"{WRONG_SHAPE}") + ".*" + re.escape(f"{ENERGY.shape}")

    with pytest.raises(ValueError, match=expected_message):
        apply_mask_weights(ENERGY, **{mask_argument: wrong_mask})


def test_without_masks_returns_equal_copy():
    weighted = apply_mask_weights(ENERGY)

    np.testing.assert_array_equal(weighted, ENERGY)
    assert not np.shares_memory(weighted, ENERGY)


@pytest.mark.parametrize("non_finite_value", [np.inf, -np.inf, np.nan])
def test_non_finite_energy_raises_value_error(non_finite_value):
    energy = ENERGY.copy()
    energy[PROTECTED_PIXEL] = non_finite_value

    with pytest.raises(ValueError, match="inf or nan"):
        apply_mask_weights(energy, remove_mask=mask_from_pixels(REMOVAL_PIXELS))


MASK_SHAPE = (6, 8)


def rectangle_mask(n_rows: int, n_cols: int) -> np.ndarray:
    mask = np.zeros(MASK_SHAPE, dtype=bool)
    mask[1 : 1 + n_rows, 2 : 2 + n_cols] = True
    return mask


@pytest.mark.parametrize(
    ("rect_rows", "rect_cols"),
    [(2, 5), (4, 1), (1, 1)],
    ids=["wide_flat", "narrow_tall", "single_pixel"],
)
def test_seams_needed_counts_most_crowded_row_and_column(rect_rows, rect_cols):
    mask = rectangle_mask(rect_rows, rect_cols)

    n_vertical, n_horizontal = seams_needed(mask)

    assert (n_vertical, n_horizontal) == (rect_cols, rect_rows)


def test_seams_needed_uses_most_crowded_line_of_irregular_mask():
    mask = mask_from_pixels(REMOVAL_PIXELS)

    assert seams_needed(mask) == (2, 3)


def test_seams_needed_for_empty_mask_is_zero():
    assert seams_needed(np.zeros(MASK_SHAPE, dtype=bool)) == (0, 0)


def test_seams_needed_returns_python_ints():
    n_vertical, n_horizontal = seams_needed(rectangle_mask(2, 5))

    assert type(n_vertical) is int
    assert type(n_horizontal) is int
