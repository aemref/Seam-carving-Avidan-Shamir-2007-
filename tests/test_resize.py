"""Tests for ``seamcarving.resize``.

The assertions rely only on what the interface contract guarantees (output size and dtype,
original pixels kept by seam removal, protection dominating any finite energy), never on exact
seam positions or pixel values, so they hold for any conforming energy, carve and enlarge module.
"""

import numpy as np
import pytest

from seamcarving import carve, energy, enlarge, seam
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
    before = img.copy()
    out = resize(img, new_h, new_w, order=order)
    assert out.shape == (new_h, new_w, *img.shape[2:])
    assert out.dtype == img.dtype
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(img, before)


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


# --- call routing: which module function gets called, in which order, with what ----------


@pytest.fixture
def calls(monkeypatch):
    """Record top-level calls to the carve / enlarge entry points (nested calls are skipped).

    resize calls them as module attributes, so wrapping the attribute sees every call while the
    real (or overlaid) implementation still does the work.
    """
    log = []
    depth = [0]

    def wrap(module, name):
        original = getattr(module, name)

        def spy(*args, **kwargs):
            outer = depth[0] == 0
            depth[0] += 1
            try:
                result = original(*args, **kwargs)
            finally:
                depth[0] -= 1
            if outer:
                log.append({"name": name, "args": args, "kwargs": kwargs, "result": result})
            return result

        monkeypatch.setattr(module, name, spy)

    for name in ("shrink_width", "shrink_height"):
        wrap(carve, name)
    for name in ("enlarge_width", "enlarge_height"):
        wrap(enlarge, name)
    return log


def _expected_calls(new_h, new_w, axes):
    names = []
    for axis in axes:
        delta = (new_w - W) if axis == "width" else (new_h - H)
        if delta:
            names.append(("shrink_" if delta < 0 else "enlarge_") + axis)
    return names


@pytest.mark.parametrize("order", ["width_first", "height_first"])
@pytest.mark.parametrize("new_h, new_w", [(22, 29), (22, 52), (39, 29), (39, 52), (30, 29)])
def test_order_is_honoured_without_mask(calls, order, new_h, new_w):
    img, _ = _scene(False)
    resize(img, new_h, new_w, order=order)
    axes = ["width", "height"] if order == "width_first" else ["height", "width"]
    assert [c["name"] for c in calls] == _expected_calls(new_h, new_w, axes)


@pytest.mark.parametrize("order", ["width_first", "height_first"])
@pytest.mark.parametrize("new_h, new_w", [(22, 52), (39, 29)])
def test_mixed_directions_with_mask_shrink_first(calls, order, new_h, new_w):
    img, mask = _scene(True)
    out = resize(img, new_h, new_w, protect_mask=mask, order=order)
    assert out.shape[:2] == (new_h, new_w)
    assert [c["name"][:6] for c in calls] == ["shrink", "enlarg"]
    assert _block_intact(calls[0]["result"].image)  # protected while shrinking
    # enlarging only inserts pixels, so every protected pixel is still in the output
    assert np.count_nonzero((out == BLOCK_VALUE).all(axis=2)) >= 100


def test_mask_and_energy_fn_routing(calls):
    img, mask = _scene(False)

    def base_energy(im, carry):
        return energy.compute_energy(im)

    resize(img, 22, 52, energy_fn=base_energy, seam_fn=seam.find_vertical_seam)
    shrink_call, enlarge_call = sorted(calls, key=lambda c: c["name"].startswith("enlarge"))
    assert shrink_call["kwargs"]["carry"] == ()  # no mask: nothing carried, nothing wrapped
    assert shrink_call["kwargs"]["energy_fn"] is base_energy
    assert enlarge_call["kwargs"]["energy_fn"] is base_energy

    calls.clear()
    resize(img, 22, 52, energy_fn=base_energy, seam_fn=seam.find_vertical_seam, protect_mask=mask)
    shrink_call, enlarge_call = calls
    assert shrink_call["name"] == "shrink_height"
    (carried,) = shrink_call["kwargs"]["carry"]
    np.testing.assert_array_equal(carried, mask)
    assert shrink_call["kwargs"]["energy_fn"] is not base_energy  # wrapped with mask weights
    assert enlarge_call["kwargs"]["energy_fn"] is base_energy  # enlarging gets the base only
    for call in calls:
        assert call["kwargs"]["seam_fn"] is seam.find_vertical_seam


@pytest.mark.parametrize("order", ["width_first", "height_first"])
def test_carved_mask_passes_to_second_axis(calls, order):
    img, mask = _scene(False)
    resize(img, H - 10, W - 15, protect_mask=mask, order=order)
    first, second = calls
    (handed_over,) = second["kwargs"]["carry"]
    assert handed_over is first["result"].carry[0]
    assert handed_over.shape == first["result"].image.shape[:2]


@pytest.mark.parametrize("new_h, new_w", [(H - 10, W), (H, W - 15), (H - 10, W - 15)])
def test_wrapped_energy_sees_matching_image_and_mask(new_h, new_w):
    # shrink_height calls energy_fn with the transposed image and transposed carry.
    seen = []

    def base_energy(im, carry):
        seen.append((im.shape[:2], carry[0].shape))
        return energy.compute_energy(im)

    img, mask = _scene(True)
    resize(img, new_h, new_w, energy_fn=base_energy, protect_mask=mask, order="height_first")
    assert len(seen) == (H - new_h) + (W - new_w)
    assert all(img_shape == mask_shape for img_shape, mask_shape in seen)


def test_enlarging_ignores_mask():
    img, mask = _scene(False)
    out = resize(img, 39, 52, protect_mask=mask)
    assert out.shape == (39, 52)


# --- validation, copies, input safety ------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -3, 2.5, 20.0, True, "20", None])
def test_invalid_target_size(bad):
    img, _ = _scene(False)
    with pytest.raises(ValueError):
        resize(img, bad, W)
    with pytest.raises(ValueError):
        resize(img, H, bad)


def test_numpy_integer_sizes_are_accepted():
    img, _ = _scene(False)
    assert resize(img, np.int64(22), np.int32(52)).shape == (22, 52)


def test_invalid_order_image_and_mask():
    img, mask = _scene(False)
    with pytest.raises(ValueError):
        resize(img, 22, 29, order="both")
    with pytest.raises(ValueError):
        resize(img, 22, 29, protect_mask=mask.astype(np.uint8))
    with pytest.raises(ValueError):
        resize(img, 22, 29, protect_mask=mask[:, :-1])
    with pytest.raises(ValueError):
        resize(img[0], 1, 20)
    with pytest.raises(ValueError):
        resize(img[..., None, None], 22, 29)


@pytest.mark.parametrize("new_h, new_w", [(22, 29), (22, 52), (39, 29)])
def test_input_and_mask_unchanged(new_h, new_w):
    img, mask = _scene(True)
    img_before, mask_before = img.copy(), mask.copy()
    resize(img, new_h, new_w, protect_mask=mask)
    np.testing.assert_array_equal(img, img_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_same_size_with_mask_is_a_contiguous_copy():
    img, mask = _scene(True)
    view = np.swapaxes(np.swapaxes(img, 0, 1).copy(), 0, 1)  # same values, not C-contiguous
    out = resize(view, H, W, protect_mask=mask)
    np.testing.assert_array_equal(out, img)
    assert out.flags.c_contiguous and not np.shares_memory(out, view)
