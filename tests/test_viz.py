import numpy as np
import pytest

from seamcarving.viz import energy_to_image

HEIGHT = 6
WIDTH = 8
LARGE_NEGATIVE_WEIGHT = -1e8


def make_ramp_energy():
    return np.arange(HEIGHT * WIDTH, dtype=np.float64).reshape(HEIGHT, WIDTH)


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
