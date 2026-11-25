import numpy as np

from experiments.d8_noise import add_gaussian_noise, add_salt_pepper, iou, median_energy
from seamcarving.energy import compute_energy


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, size=(40, 50, 3), dtype=np.uint8)


def test_gaussian_noise_is_uint8_and_has_expected_spread():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    before = img.copy()
    noisy = add_gaussian_noise(img, 15.0, np.random.default_rng(0))
    assert noisy.dtype == np.uint8 and noisy.shape == img.shape
    np.testing.assert_array_equal(img, before)
    diff = noisy.astype(np.float64) - 128.0
    assert abs(diff.mean()) < 0.5
    assert 14.0 < diff.std() < 16.0


def test_salt_pepper_hits_expected_fraction_with_0_or_255():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    noisy = add_salt_pepper(img, 0.05, np.random.default_rng(0))
    changed = np.any(noisy != img, axis=2)
    assert 0.04 < changed.mean() < 0.06
    values = noisy[changed]
    assert np.all((values == 0) | (values == 255))
    assert np.all(img == 128)


def test_noise_is_reproducible_with_seed():
    img = _img()
    a = add_gaussian_noise(img, 15.0, np.random.default_rng(0))
    b = add_gaussian_noise(img, 15.0, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)


def test_iou():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    assert iou(a, b) == 1.0
    a[0, :2] = True
    b[0, 1:3] = True
    assert iou(a, b) == 1 / 3
    assert iou(a, a) == 1.0


def test_median_energy_filters_only_for_the_energy():
    img = _img(1)
    before = img.copy()
    e = median_energy(img, ())
    assert e.shape == img.shape[:2] and e.dtype == np.float64
    np.testing.assert_array_equal(img, before)
    assert not np.array_equal(e, compute_energy(img))
