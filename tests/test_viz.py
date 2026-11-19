import numpy as np
import pytest

from seamcarving.viz import draw_seams, energy_to_image

HEIGHT = 6
WIDTH = 8
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
