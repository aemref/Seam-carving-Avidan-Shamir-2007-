import numpy as np
import pytest

from seamcarving.viz import (
    GRID_GAP,
    LABEL_STRIP_HEIGHT,
    comparison_grid,
    draw_seams,
    energy_to_image,
)

HEIGHT = 6
WIDTH = 8
CELL_HEIGHT = 40
BACKGROUND_VALUE = 50
SEAM_COLOR = (0, 0, 255)
LARGE_NEGATIVE_WEIGHT = -1e8


def make_bgr(height=HEIGHT, width=WIDTH, value=BACKGROUND_VALUE):
    return np.full((height, width, 3), value, dtype=np.uint8)


def make_gray(height=HEIGHT, width=WIDTH, value=BACKGROUND_VALUE):
    return np.full((height, width), value, dtype=np.uint8)


def make_ramp_energy():
    return np.arange(HEIGHT * WIDTH, dtype=np.float64).reshape(HEIGHT, WIDTH)


def make_seam():
    return np.arange(HEIGHT, dtype=np.int64) % WIDTH


def grid_height(row_count, cell_height=CELL_HEIGHT):
    return row_count * (cell_height + LABEL_STRIP_HEIGHT) + (row_count + 1) * GRID_GAP


def test_energy_to_image_returns_uint8_with_input_shape():
    image = energy_to_image(make_ramp_energy())

    assert image.dtype == np.uint8
    assert image.shape == (HEIGHT, WIDTH)


def test_energy_to_image_stretches_to_full_range():
    image = energy_to_image(make_ramp_energy())

    assert image.min() == 0
    assert image.max() == 255


def test_energy_to_image_constant_map_gives_zeros():
    image = energy_to_image(np.full((HEIGHT, WIDTH), 7.5))

    assert not image.any()


def test_energy_to_image_log_handles_large_negative_weights():
    energy = make_ramp_energy()
    energy[0, 0] = LARGE_NEGATIVE_WEIGHT

    image = energy_to_image(energy, log=True)

    assert image.dtype == np.uint8
    assert image[0, 0] == 0
    assert image.max() == 255


def test_energy_to_image_does_not_modify_input():
    energy = make_ramp_energy()
    original = energy.copy()

    energy_to_image(energy, log=True)

    np.testing.assert_array_equal(energy, original)


def test_energy_to_image_rejects_non_2d_input():
    with pytest.raises(ValueError):
        energy_to_image(np.zeros((HEIGHT, WIDTH, 3)))


def test_draw_seams_with_seam_paints_only_seam_pixels():
    seam = make_seam()

    result = draw_seams(make_bgr(), seam, color=SEAM_COLOR)

    expected_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    expected_mask[np.arange(HEIGHT), seam] = True
    assert np.all(result[expected_mask] == SEAM_COLOR)
    assert np.all(result[~expected_mask] == BACKGROUND_VALUE)


def test_draw_seams_with_bool_map_paints_only_marked_pixels():
    removed = np.zeros((HEIGHT, WIDTH), dtype=bool)
    removed[1:3, 2:5] = True

    result = draw_seams(make_bgr(), removed, color=SEAM_COLOR)

    assert np.all(result[removed] == SEAM_COLOR)
    assert np.all(result[~removed] == BACKGROUND_VALUE)


def test_draw_seams_converts_gray_to_bgr():
    result = draw_seams(make_gray(), make_seam())

    assert result.dtype == np.uint8
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_draw_seams_does_not_modify_input():
    image = make_bgr()
    original = image.copy()

    draw_seams(image, make_seam())

    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    "removed",
    [
        np.zeros(HEIGHT + 1, dtype=np.int64),
        np.full(HEIGHT, WIDTH, dtype=np.int64),
        np.zeros((HEIGHT, WIDTH + 1), dtype=bool),
        np.zeros((HEIGHT, WIDTH), dtype=np.int64),
    ],
    ids=["seam-wrong-length", "seam-column-out-of-range", "map-wrong-shape", "map-not-bool"],
)
def test_draw_seams_rejects_invalid_removed(removed):
    with pytest.raises(ValueError):
        draw_seams(make_bgr(), removed)


def make_mixed_images(count):
    return [make_gray() if index % 2 else make_bgr() for index in range(count)]


def make_labels(count):
    return [f"img {index}" for index in range(count)]


def test_comparison_grid_returns_bgr_uint8():
    grid = comparison_grid(make_mixed_images(2), make_labels(2), cell_height=CELL_HEIGHT)

    assert grid.dtype == np.uint8
    assert grid.ndim == 3
    assert grid.shape[2] == 3


@pytest.mark.parametrize(
    ("image_count", "cols", "expected_rows"),
    [(5, None, 1), (7, None, 2), (7, 3, 3), (1, None, 1)],
)
def test_comparison_grid_row_count_follows_cols(image_count, cols, expected_rows):
    grid = comparison_grid(
        make_mixed_images(image_count),
        make_labels(image_count),
        cols=cols,
        cell_height=CELL_HEIGHT,
    )

    assert grid.shape[0] == grid_height(expected_rows)


def test_comparison_grid_scales_cells_to_same_height_keeping_aspect_ratio():
    tall = make_bgr(height=CELL_HEIGHT * 2, width=CELL_HEIGHT)
    short = make_bgr(height=HEIGHT, width=WIDTH)
    labels = ["", ""]

    grid = comparison_grid([tall, short], labels, cell_height=CELL_HEIGHT)

    tall_cell_width = CELL_HEIGHT // 2
    short_cell_width = round(WIDTH * CELL_HEIGHT / HEIGHT)
    expected_width = tall_cell_width + short_cell_width + 3 * GRID_GAP
    assert grid.shape == (grid_height(1), expected_width, 3)


def test_comparison_grid_widens_cell_for_long_label():
    narrow_image = make_bgr(height=CELL_HEIGHT, width=2)
    long_label = "a very long label that cannot fit"

    grid = comparison_grid([narrow_image], [long_label], cell_height=CELL_HEIGHT)

    assert grid.shape[1] > 2 + 2 * GRID_GAP


def test_comparison_grid_rejects_non_ascii_label():
    with pytest.raises(ValueError, match="Görüntü"):
        comparison_grid(make_mixed_images(2), ["ok", "Görüntü"])


def test_comparison_grid_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        comparison_grid(make_mixed_images(3), make_labels(2))


def test_comparison_grid_rejects_empty_list():
    with pytest.raises(ValueError):
        comparison_grid([], [])


def test_comparison_grid_does_not_modify_inputs():
    images = make_mixed_images(3)
    originals = [image.copy() for image in images]

    comparison_grid(images, make_labels(3), cell_height=CELL_HEIGHT)

    for image, original in zip(images, originals):
        np.testing.assert_array_equal(image, original)
