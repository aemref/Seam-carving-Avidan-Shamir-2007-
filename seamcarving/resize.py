"""Single entry point for content-aware resizing. Owner: Emre."""

import operator

import numpy as np

from seamcarving import carve, energy, enlarge, object_removal
from seamcarving.types import EnergyFn, SeamFn

ORDERS = ("width_first", "height_first")


def _check_size(value, name: str) -> int:
    """Return ``value`` as an int >= 1; bools, fractions and non-integers raise ValueError."""
    try:
        ok = not isinstance(value, (bool, np.bool_)) and operator.index(value) >= 1
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return operator.index(value)


def _protected(base: EnergyFn | None) -> EnergyFn:
    """Wrap ``base`` so that ``carry[0]`` (the protect mask, carved along) gets +P weights.

    ``shrink_height`` calls this with the transposed image and transposed carry, so the mask in
    ``carry[0]`` always matches the image it is called with.
    """

    def protected_energy(img: np.ndarray, carry: tuple) -> np.ndarray:
        e = energy.compute_energy(img) if base is None else base(img, carry)
        return object_removal.apply_mask_weights(e, protect_mask=carry[0])

    return protected_energy


def resize(
    img: np.ndarray,
    new_h: int,
    new_w: int,
    energy_fn: EnergyFn | None = None,
    seam_fn: SeamFn | None = None,
    protect_mask: np.ndarray | None = None,
    order: str = "width_first",
) -> np.ndarray:
    """Resize to (new_h, new_w) by removing or inserting seams.

    Each axis is handled once: a smaller target removes seams (``carve.shrink_width`` /
    ``carve.shrink_height``), a larger one inserts seams (``enlarge.enlarge_width`` /
    ``enlarge.enlarge_height``), an equal one is left alone.

    Args:
        img: (H, W[, 3]) uint8 image. Not modified.
        new_h, new_w: target size, integers >= 1.
        energy_fn: base energy ``(img, carry) -> (H, W)``; None means ``compute_energy(img)``.
        seam_fn: vertical seam finder; None means ``find_vertical_seam``.
        protect_mask: bool (H, W), pixels to keep. Applied when shrinking only: it is carved
            along with the image (``carry``) and ``energy_fn`` is wrapped with
            ``apply_mask_weights``; after the first axis the carved mask is passed to the second.
            Enlarging ignores it (seams may be inserted through protected pixels) and gets the
            unwrapped ``energy_fn``. If one axis shrinks and the other enlarges, the shrinking
            axis always goes first, whatever ``order`` says, because the mask is only valid in
            input coordinates and cannot follow an enlargement.
        order: "width_first" | "height_first". Without a protect mask it is always honoured.

    Returns:
        (new_h, new_w[, 3]) C-contiguous array with the input's dtype; a copy if the size is
        unchanged.

    Raises:
        ValueError: image not 2-D or 3-D, size not an integer >= 1, unknown ``order``, or
            ``protect_mask`` not bool or not of shape (H, W).
    """
    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be (H, W) or (H, W, C), got shape {img.shape}")
    new_h = _check_size(new_h, "new_h")
    new_w = _check_size(new_w, "new_w")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    if protect_mask is not None:
        protect_mask = np.asarray(protect_mask)
        if protect_mask.dtype != bool:
            raise ValueError(f"protect_mask must be bool, got {protect_mask.dtype}")
        if protect_mask.shape != img.shape[:2]:
            raise ValueError(
                f"protect_mask shape {protect_mask.shape} != image shape {img.shape[:2]}"
            )

    h, w = img.shape[:2]
    d_rows, d_cols = new_h - h, new_w - w
    axes = ["width", "height"] if order == "width_first" else ["height", "width"]
    if protect_mask is not None and d_rows * d_cols < 0:
        axes = ["width", "height"] if d_cols < 0 else ["height", "width"]

    shrink_energy = energy_fn if protect_mask is None else _protected(energy_fn)
    carry = () if protect_mask is None else (protect_mask,)
    out = img
    for axis in axes:
        delta = d_cols if axis == "width" else d_rows
        if delta < 0:
            shrink = carve.shrink_width if axis == "width" else carve.shrink_height
            res = shrink(out, -delta, energy_fn=shrink_energy, seam_fn=seam_fn, carry=carry)
            out, carry = res.image, res.carry
        elif delta > 0:
            grow = enlarge.enlarge_width if axis == "width" else enlarge.enlarge_height
            out = grow(out, delta, energy_fn=energy_fn, seam_fn=seam_fn)

    if out is img or np.shares_memory(out, img):
        out = out.copy()
    return np.ascontiguousarray(out)
