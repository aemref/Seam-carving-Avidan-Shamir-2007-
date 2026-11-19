import cv2
import numpy as np
import pytest

from seamcarving.filters import (
    LAPLACIAN,
    ROBERTS_ANTI,
    ROBERTS_DIAG,
    SCHARR_COLS,
    SCHARR_ROWS,
    SOBEL_COLS,
    SOBEL_ROWS,
    correlate2d,
    gaussian_kernel,
)

BORDERS = {
    "reflect": cv2.BORDER_REFLECT_101,
    "edge": cv2.BORDER_REPLICATE,
    "zero": cv2.BORDER_CONSTANT,
}

DERIVATIVE_KERNELS = [
    SOBEL_ROWS,
    SOBEL_COLS,
    SCHARR_ROWS,
    SCHARR_COLS,
    LAPLACIAN,
    ROBERTS_DIAG,
    ROBERTS_ANTI,
]


@pytest.mark.parametrize("border", list(BORDERS))
@pytest.mark.parametrize("kshape", [(3, 3), (5, 5), (7, 3), (1, 5)])
def test_matches_cv2_filter2d(border, kshape):
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, size=(17, 23))
    kernel = rng.normal(size=kshape)
    ours = correlate2d(img, kernel, border=border)
    ref = cv2.filter2D(img, cv2.CV_64F, kernel, borderType=BORDERS[border])
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-9)


@pytest.mark.parametrize("border", list(BORDERS))
def test_matches_cv2_on_uint8_input(border):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(12, 9), dtype=np.uint8)
    ours = correlate2d(img, SOBEL_COLS, border=border)
    ref = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, SOBEL_COLS, borderType=BORDERS[border])
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-9)


def test_impulse_gives_rotated_kernel():
    kernel = np.arange(9, dtype=np.float64).reshape(3, 3)
    img = np.zeros((7, 7))
    img[3, 3] = 1.0
    out = correlate2d(img, kernel, border="zero")
    np.testing.assert_array_equal(out[2:5, 2:5], np.rot90(kernel, 2))
    assert out.sum() == kernel.sum()


@pytest.mark.parametrize("kernel", DERIVATIVE_KERNELS)
def test_derivative_kernels_sum_to_zero(kernel):
    assert kernel.shape == (3, 3)
    assert kernel.dtype == np.float64
    assert kernel.sum() == 0


def test_transposed_constants():
    np.testing.assert_array_equal(SOBEL_COLS, SOBEL_ROWS.T)
    np.testing.assert_array_equal(SCHARR_COLS, SCHARR_ROWS.T)


@pytest.mark.parametrize(("sigma", "side"), [(0.5, 3), (1.0, 7), (1.5, 9), (2.0, 13), (0.1, 1)])
def test_gaussian_kernel_size_and_sum(sigma, side):
    k = gaussian_kernel(sigma)
    assert k.shape == (side, side)
    assert k.dtype == np.float64
    assert k.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(k, k.T)
    np.testing.assert_allclose(k, np.rot90(k, 2))
    assert np.unravel_index(np.argmax(k), k.shape) == (side // 2, side // 2)


def test_gaussian_kernel_rejects_bad_sigma():
    with pytest.raises(ValueError):
        gaussian_kernel(0)
    with pytest.raises(ValueError):
        gaussian_kernel(-1.0)


def test_even_kernel_raises():
    with pytest.raises(ValueError):
        correlate2d(np.zeros((5, 5)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        correlate2d(np.zeros((5, 5)), np.ones((3, 4)))


def test_bad_border_and_shape_raise():
    with pytest.raises(ValueError):
        correlate2d(np.zeros((5, 5)), np.ones((3, 3)), border="wrap")
    with pytest.raises(ValueError):
        correlate2d(np.zeros((5, 5, 3)), np.ones((3, 3)))


def test_input_unchanged_and_float64_output():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(8, 10), dtype=np.uint8)
    kernel = SOBEL_ROWS.copy()
    img_before, kernel_before = img.copy(), kernel.copy()
    out = correlate2d(img, kernel)
    np.testing.assert_array_equal(img, img_before)
    np.testing.assert_array_equal(kernel, kernel_before)
    assert out.dtype == np.float64
    assert out.shape == img.shape
