import cv2
import numpy as np
import pytest

from seamcarving.filters import correlate2d

BORDERS = {
    "reflect": cv2.BORDER_REFLECT_101,
    "edge": cv2.BORDER_REPLICATE,
    "zero": cv2.BORDER_CONSTANT,
}


@pytest.mark.parametrize("border", list(BORDERS))
@pytest.mark.parametrize("kshape", [(3, 3), (5, 5), (7, 3), (1, 5)])
def test_matches_cv2_filter2d(border, kshape):
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, size=(17, 23))
    kernel = rng.normal(size=kshape)
    ours = correlate2d(img, kernel, border=border)
    ref = cv2.filter2D(img, cv2.CV_64F, kernel, borderType=BORDERS[border])
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-9)


def test_impulse_gives_rotated_kernel():
    kernel = np.arange(9, dtype=np.float64).reshape(3, 3)
    img = np.zeros((7, 7))
    img[3, 3] = 1.0
    out = correlate2d(img, kernel, border="zero")
    np.testing.assert_array_equal(out[2:5, 2:5], np.rot90(kernel, 2))
    assert out.sum() == kernel.sum()


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
