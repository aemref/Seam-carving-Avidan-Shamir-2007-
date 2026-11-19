"""Tests for ``seamcarving.resize``.

The assertions rely only on what the interface contract guarantees (output size and dtype,
original pixels kept by seam removal, protection dominating any finite energy), never on exact
seam positions or pixel values, so they hold for any conforming energy, carve and enlarge module.
"""

import numpy as np
import pytest

from seamcarving.resize import resize

H, W = 30, 40
BLOCK = (slice(8, 18), slice(14, 24))  # 10 x 10 protected block
BLOCK_VALUE = 7  # the background never takes this value


def _scene(color: bool) -> tuple[np.ndarray, np.ndarray]:
    """Textured background (values 50..255) with a flat block of a unique value.

    A flat region has zero energy for every operator, so without protection it is where the
    cheapest seams go; with protection, mask weights dominate any finite energy.
    """
    rng = np.random.default_rng(0)
    shape = (H, W, 3) if color else (H, W)
    img = rng.integers(50, 256, shape).astype(np.uint8)
    img[BLOCK] = BLOCK_VALUE
    mask = np.zeros((H, W), bool)
    mask[BLOCK] = True
    return img, mask


def _block_intact(out: np.ndarray) -> bool:
    """True if the output holds exactly one 10 x 10 block of BLOCK_VALUE and nothing else."""
    hit = out == BLOCK_VALUE
    if hit.ndim == 3:
        hit = hit.all(axis=2)
    rows, cols = np.nonzero(hit)
    if rows.size != 100:
        return False
    box = hit[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    return box.shape == (10, 10) and bool(box.all())


@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("order", ["width_first", "height_first"])
@pytest.mark.parametrize("new_h", [22, 39])  # shrink, enlarge
@pytest.mark.parametrize("new_w", [29, 52])  # shrink, enlarge
def test_output_size_for_all_direction_combinations(color, order, new_h, new_w):
    img, _ = _scene(color)
    out = resize(img, new_h, new_w, order=order)
    assert out.shape == (new_h, new_w, *img.shape[2:])
    assert out.dtype == img.dtype


@pytest.mark.parametrize("color", [False, True])
def test_same_size_returns_copy(color):
    img, _ = _scene(color)
    out = resize(img, H, W)
    np.testing.assert_array_equal(out, img)
    assert out is not img and not np.shares_memory(out, img)


@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("order", ["width_first", "height_first"])
def test_protect_mask_keeps_block_when_shrinking(color, order):
    img, mask = _scene(color)
    out = resize(img, H - 10, W - 15, protect_mask=mask, order=order)
    assert out.shape[:2] == (H - 10, W - 15)
    assert _block_intact(out)


def test_block_is_carved_without_protection():
    # Sanity check for the scene above: the zero-energy block attracts seams.
    img, _ = _scene(False)
    out = resize(img, H, W - 15)
    assert np.count_nonzero(out == BLOCK_VALUE) < 100
