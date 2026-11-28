import itertools
import re
from collections.abc import Iterator

import numpy as np
import pytest

from seamcarving.object_removal import (
    UnreachablePixelsWarning,
    apply_mask_weights,
    remove_object,
    seams_needed,
)

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


# Synthetic scene for remove_object: a smooth low-energy ramp (values <= 100), a textured
# "object" (200-255), a separate textured "protected" block (150-160) and, around the object,
# an optional one-pixel halo (180) that is not in the removal mask.
SCENE_HEIGHT, SCENE_WIDTH = 24, 36
BACKGROUND_RANGE = (30, 100)
OBJECT_VALUES = (200, 255)
PROTECTED_VALUES = (150, 160)
HALO_VALUE = 180
OBJECT_REGION = (slice(8, 16), slice(10, 16))  # 8 rows x 6 columns: vertical seams are fewer
OBJECT_WIDTH = 6
WIDE_FLAT_OBJECT_REGION = (slice(10, 12), slice(4, 18))  # 2 rows x 14 columns
HALO_REGION = (slice(7, 17), slice(9, 17))  # OBJECT_REGION grown by one pixel
PROTECTED_REGION = (slice(6, 18), slice(24, 29))


def make_scene(object_region=OBJECT_REGION, with_halo=False):
    rng = np.random.default_rng(0)
    ramp = np.linspace(*BACKGROUND_RANGE, SCENE_WIDTH)
    gray = np.rint(np.tile(ramp, (SCENE_HEIGHT, 1))).astype(np.uint8)
    if with_halo:
        gray[HALO_REGION] = HALO_VALUE
    gray[object_region] = textured_patch(rng, object_region, OBJECT_VALUES)
    gray[PROTECTED_REGION] = textured_patch(rng, PROTECTED_REGION, PROTECTED_VALUES)
    return np.repeat(gray[..., np.newaxis], 3, axis=2)


def textured_patch(rng, region, value_range):
    shape = np.zeros((SCENE_HEIGHT, SCENE_WIDTH))[region].shape
    low, high = value_range
    return rng.integers(low, high + 1, size=shape, dtype=np.uint8)


def region_mask(region):
    mask = np.zeros((SCENE_HEIGHT, SCENE_WIDTH), dtype=bool)
    mask[region] = True
    return mask


def count_values_between(img, value_range):
    low, high = value_range
    gray = img[..., 0] if img.ndim == 3 else img
    return int(np.count_nonzero((gray >= low) & (gray <= high)))


def remove_scene_object(**kwargs):
    return remove_object(make_scene(), region_mask(OBJECT_REGION), restore_size=False, **kwargs)


def test_removal_leaves_no_object_pixels():
    result = remove_scene_object()

    assert count_values_between(result.image, OBJECT_VALUES) == 0


def test_removal_narrows_image_by_seam_count():
    result = remove_scene_object()

    assert result.direction == "vertical"
    assert result.image.shape == (SCENE_HEIGHT, SCENE_WIDTH - result.n_seams, 3)
    assert result.image.dtype == np.uint8


def test_removal_needs_at_least_object_width_seams():
    result = remove_scene_object()

    assert result.n_seams >= OBJECT_WIDTH


def test_protected_block_survives_removal():
    scene = make_scene()
    protected_before = count_values_between(scene, PROTECTED_VALUES)

    result = remove_scene_object(protect_mask=region_mask(PROTECTED_REGION))

    assert count_values_between(result.image, PROTECTED_VALUES) == protected_before


def test_restore_size_returns_input_shape():
    scene = make_scene()

    result = remove_object(scene, region_mask(OBJECT_REGION), restore_size=True)

    assert result.image.shape == scene.shape
    assert result.n_seams >= OBJECT_WIDTH


def test_auto_direction_picks_horizontal_for_wide_flat_object():
    scene = make_scene(WIDE_FLAT_OBJECT_REGION)

    result = remove_object(scene, region_mask(WIDE_FLAT_OBJECT_REGION), restore_size=False)

    assert result.direction == "horizontal"
    assert result.image.shape == (SCENE_HEIGHT - result.n_seams, SCENE_WIDTH, 3)
    assert count_values_between(result.image, OBJECT_VALUES) == 0


def test_explicit_direction_overrides_auto():
    result = remove_scene_object(direction="horizontal")

    assert result.direction == "horizontal"
    assert result.image.shape == (SCENE_HEIGHT - result.n_seams, SCENE_WIDTH, 3)
    assert count_values_between(result.image, OBJECT_VALUES) == 0


def test_dilate_also_removes_halo_that_plain_mask_leaves():
    scene = make_scene(with_halo=True)
    halo = (HALO_VALUE, HALO_VALUE)

    plain = remove_object(scene, region_mask(OBJECT_REGION), restore_size=False)
    dilated = remove_object(scene, region_mask(OBJECT_REGION), restore_size=False, dilate=1)

    assert count_values_between(plain.image, halo) > 0
    assert count_values_between(dilated.image, halo) == 0


def test_works_on_grayscale_image():
    gray_scene = make_scene()[..., 0]

    result = remove_object(gray_scene, region_mask(OBJECT_REGION), restore_size=False)

    assert result.image.shape == (SCENE_HEIGHT, SCENE_WIDTH - result.n_seams)
    assert count_values_between(result.image, OBJECT_VALUES) == 0


def test_empty_removal_mask_returns_unchanged_copy():
    scene = make_scene()

    result = remove_object(scene, np.zeros((SCENE_HEIGHT, SCENE_WIDTH), dtype=bool))

    assert result.n_seams == 0
    np.testing.assert_array_equal(result.image, scene)
    assert not np.shares_memory(result.image, scene)


def test_removal_pixels_under_protection_are_not_removed():
    scene = make_scene()

    result = remove_object(
        scene, region_mask(OBJECT_REGION), protect_mask=region_mask(OBJECT_REGION)
    )

    assert result.n_seams == 0
    np.testing.assert_array_equal(result.image, scene)


def test_inputs_are_not_modified():
    scene = make_scene()
    remove_mask = region_mask(OBJECT_REGION)
    protect_mask = region_mask(PROTECTED_REGION)

    remove_object(scene, remove_mask, protect_mask, dilate=1)

    np.testing.assert_array_equal(scene, make_scene())
    np.testing.assert_array_equal(remove_mask, region_mask(OBJECT_REGION))
    np.testing.assert_array_equal(protect_mask, region_mask(PROTECTED_REGION))


def test_energy_fn_receives_carved_remove_and_protect_masks():
    carry_shapes = []

    def recording_energy_fn(img, carry):
        carry_shapes.append(tuple(mask.shape for mask in carry))
        return np.zeros(img.shape[:2])

    remove_scene_object(energy_fn=recording_energy_fn)

    assert carry_shapes[0] == ((SCENE_HEIGHT, SCENE_WIDTH),) * 2
    assert carry_shapes[1] == ((SCENE_HEIGHT, SCENE_WIDTH - 1),) * 2


def test_mask_needing_more_seams_than_the_image_allows_raises():
    full_row = np.zeros((SCENE_HEIGHT, SCENE_WIDTH), dtype=bool)
    full_row[SCENE_HEIGHT // 2] = True  # every vertical seam removes exactly one of its pixels

    with pytest.raises(RuntimeError, match="removal pixels are left"):
        remove_object(make_scene(), full_row, direction="vertical")


def test_seam_that_removes_no_mask_pixel_stops_carving_with_warning():
    remove_mask = np.zeros((SCENE_HEIGHT, SCENE_WIDTH), dtype=bool)
    remove_mask[:, -1] = True
    scene = make_scene()

    def first_column_seam(energy):
        return np.zeros(energy.shape[0], dtype=np.int64)

    with pytest.warns(UnreachablePixelsWarning) as caught:
        result = remove_object(scene, remove_mask, seam_fn=first_column_seam)

    assert caught[0].message.pixels_left == SCENE_HEIGHT
    assert result.n_seams == 0
    np.testing.assert_array_equal(result.image, scene)


# A protected bar right below the object: every downward path from the object's lowest rows
# crosses it, so those removal pixels cannot be reached without cutting protection.
TRAP_BAR_REGION = (slice(16, 17), slice(4, 22))


def test_unreachable_removal_pixels_are_kept_and_protection_holds():
    scene = make_scene()
    scene[TRAP_BAR_REGION] = PROTECTED_VALUES[0]
    protected_before = count_values_between(scene, PROTECTED_VALUES)
    protect_mask = region_mask(TRAP_BAR_REGION) | region_mask(PROTECTED_REGION)

    with pytest.warns(UnreachablePixelsWarning) as caught:
        result = remove_object(scene, region_mask(OBJECT_REGION), protect_mask, restore_size=False)

    assert caught[0].message.pixels_left > 0
    assert count_values_between(result.image, PROTECTED_VALUES) == protected_before
    assert result.n_seams < OBJECT_WIDTH + SCENE_HEIGHT


@pytest.mark.parametrize("mask_argument", ["remove_mask", "protect_mask"])
def test_remove_object_rejects_mask_of_wrong_shape(mask_argument):
    masks = {"remove_mask": region_mask(OBJECT_REGION), "protect_mask": None}
    masks[mask_argument] = np.zeros(WRONG_SHAPE, dtype=bool)

    with pytest.raises(ValueError, match="does not match"):
        remove_object(make_scene(), **masks)


@pytest.mark.parametrize(
    ("option", "value"),
    [("direction", "diagonal"), ("dilate", -1)],
    ids=["unknown_direction", "negative_dilate"],
)
def test_remove_object_rejects_bad_options(option, value):
    with pytest.raises(ValueError):
        remove_object(make_scene(), region_mask(OBJECT_REGION), **{option: value})
