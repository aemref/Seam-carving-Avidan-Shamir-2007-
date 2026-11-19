import cv2
import numpy as np
import pytest

from seamcarving.energy import METHODS, compute_energy, to_gray


def _random_bgr(shape=(15, 21), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)


def _vertical_edge(h=12, w=20, edge=10, color=False):
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, edge:] = 255
    if color:
        img = np.dstack([img, img // 2, img])
    return img


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("color", ["gray", "sum"])
def test_flat_image_has_zero_energy(method, color):
    # The grey value of this colour is not an integer, so kernel sums leave ~1e-14.
    img = np.full((9, 13, 3), (40, 120, 200), dtype=np.uint8)
    e = compute_energy(img, method=method, color=color)
    np.testing.assert_allclose(e, 0.0, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_flat_integer_gray_image_is_exactly_zero(method):
    img = np.full((9, 13), 200, dtype=np.uint8)
    np.testing.assert_array_equal(compute_energy(img, method=method), 0.0)


@pytest.mark.parametrize("method", METHODS)
def test_flat_image_with_blur_is_near_zero(method):
    img = np.full((20, 20), 77, dtype=np.uint8)
    e = compute_energy(img, method=method, blur_sigma=1.5)
    np.testing.assert_allclose(e, 0.0, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("color", [False, True])
def test_single_vertical_edge(method, color):
    edge = 10
    img = _vertical_edge(edge=edge, color=color)
    e = compute_energy(img, method=method, color="sum" if color else "gray")
    nonzero_cols = set(np.flatnonzero(e.any(axis=0)))
    assert nonzero_cols, "edge produced no energy"
    assert nonzero_cols <= {edge - 1, edge}


def test_sobel_matches_cv2():
    f = to_gray(_random_bgr())
    ours = compute_energy(f, method="sobel", color="gray", norm="l1")
    ref = np.abs(cv2.Sobel(f, cv2.CV_64F, 0, 1)) + np.abs(cv2.Sobel(f, cv2.CV_64F, 1, 0))
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-9)


def test_sobel_on_bgr_matches_cv2_on_our_gray():
    img = _random_bgr(seed=3)
    f = img.astype(np.float64) @ np.array([0.114, 0.587, 0.299])
    ours = compute_energy(img, method="sobel")
    ref = np.abs(cv2.Sobel(f, cv2.CV_64F, 0, 1)) + np.abs(cv2.Sobel(f, cv2.CV_64F, 1, 0))
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-9)


def test_scharr_matches_cv2():
    f = to_gray(_random_bgr(seed=1))
    ours = compute_energy(f, method="scharr")
    ref = np.abs(cv2.Scharr(f, cv2.CV_64F, 0, 1)) + np.abs(cv2.Scharr(f, cv2.CV_64F, 1, 0))
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-9)


def test_laplacian_matches_cv2():
    f = to_gray(_random_bgr(seed=2))
    ours = compute_energy(f, method="laplacian")
    ref = np.abs(cv2.Laplacian(f, cv2.CV_64F, ksize=1))
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-9)


def test_forward_difference_values():
    f = np.array([[0, 1, 3], [2, 2, 2], [5, 5, 0]], dtype=np.float64)
    e = compute_energy(f, method="forward")
    # d_rows: f[r+1] - f[r], zero in last row; d_cols: f[:, c+1] - f[:, c], zero in last col.
    d_rows = np.array([[2, 1, -1], [3, 3, -2], [0, 0, 0]], dtype=np.float64)
    d_cols = np.array([[1, 2, 0], [0, 0, 0], [0, -5, 0]], dtype=np.float64)
    np.testing.assert_array_equal(e, np.abs(d_rows) + np.abs(d_cols))


def test_uint8_does_not_wrap():
    img = np.array([[250, 200]], dtype=np.uint8)
    e = compute_energy(img, method="forward")
    assert e[0, 0] == 50.0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("color", ["gray", "sum"])
@pytest.mark.parametrize("norm", ["l1", "l2"])
@pytest.mark.parametrize("kind", ["gray_u8", "bgr_u8", "gray_f", "bgr_f"])
def test_shape_dtype_nonnegative(method, color, norm, kind):
    bgr = _random_bgr(shape=(11, 14), seed=4)
    img = {
        "gray_u8": bgr[:, :, 1],
        "bgr_u8": bgr,
        "gray_f": bgr[:, :, 1].astype(np.float64) / 255.0,
        "bgr_f": bgr.astype(np.float32),
    }[kind]
    e = compute_energy(img, method=method, color=color, norm=norm)
    assert e.shape == (11, 14)
    assert e.dtype == np.float64
    assert np.all(e >= 0)
    assert np.all(np.isfinite(e))


@pytest.mark.parametrize("method", METHODS)
def test_gray_and_sum_equal_on_gray_input(method):
    img = _random_bgr(seed=5)[:, :, 0]
    np.testing.assert_array_equal(
        compute_energy(img, method=method, color="gray"),
        compute_energy(img, method=method, color="sum"),
    )


def test_sum_adds_channel_energies():
    img = _random_bgr(seed=6)
    expected = sum(compute_energy(img[:, :, ch]) for ch in range(3))
    np.testing.assert_allclose(compute_energy(img, color="sum"), expected, atol=1e-9)


@pytest.mark.parametrize("method", ["forward", "roberts", "sobel", "scharr"])
def test_l2_not_greater_than_l1(method):
    img = _random_bgr(seed=7)
    l1 = compute_energy(img, method=method, norm="l1")
    l2 = compute_energy(img, method=method, norm="l2")
    assert np.all(l2 <= l1 + 1e-9)


def test_laplacian_ignores_norm():
    img = _random_bgr(seed=8)
    np.testing.assert_array_equal(
        compute_energy(img, method="laplacian", norm="l1"),
        compute_energy(img, method="laplacian", norm="l2"),
    )


@pytest.mark.parametrize("method", METHODS)
def test_blur_lowers_energy_on_noisy_image(method):
    rng = np.random.default_rng(0)
    img = np.clip(128 + rng.normal(0, 30, size=(40, 50)), 0, 255)
    plain = compute_energy(img, method=method).mean()
    blurred = compute_energy(img, method=method, blur_sigma=1.5).mean()
    assert blurred < plain


def test_input_unchanged():
    img = _random_bgr(seed=9)
    before = img.copy()
    for method in METHODS:
        compute_energy(img, method=method, color="sum", blur_sigma=1.0)
    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "prewitt"}, {"color": "rgb"}, {"norm": "linf"}, {"blur_sigma": -1.0}],
)
def test_bad_options_raise(kwargs):
    with pytest.raises(ValueError):
        compute_energy(_random_bgr(), **kwargs)


def test_bad_image_shape_raises():
    with pytest.raises(ValueError):
        compute_energy(np.zeros((5, 5, 4)))
    with pytest.raises(ValueError):
        compute_energy(np.zeros(5))
