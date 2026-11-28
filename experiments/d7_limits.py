"""D7: limitations of seam carving (paper Sec. 6) and what a protect mask fixes.

For every ``insan``, ``mimari`` and ``yogun`` photo the width is reduced by 30% and this script
writes:

- ``d7_limits_<name>.png``: original; seam carving; scale; and, if ``<name>_protect.png`` exists,
  seam carving with the protect mask (energy weighted by ``apply_mask_weights``)
- ``d7_seams_<name>.png``: the removed seams on the original (and, with a mask, the seams of the
  protected run), so the source of each distortion is visible

Failures are saved as they are: showing them is the point of this experiment.

Run: ``python -m experiments.d7_limits``
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from experiments import common
from seamcarving import baselines, carve, viz
from seamcarving.energy import compute_energy
from seamcarving.object_removal import apply_mask_weights

CATEGORIES = ("insan", "mimari", "yogun")
SHRINK = 0.30  # width -30%
SEAM_COLOR = (0, 0, 255)  # BGR red


TABLE_HEADER = [
    "photo",
    "category",
    "width",
    "height",
    "seams_removed",
    "seconds",
    "protect_px",
    "protect_removed_plain",
    "protect_removed_masked",
    "seconds_masked",
]


def _size(img: np.ndarray) -> str:
    """ASCII size label 'W x H'."""
    return f"{img.shape[1]}x{img.shape[0]}"


def _photos(category: str) -> list[Path]:
    """All photos of a category, sorted by name; [] if there are none."""
    if not common.IMAGES_DIR.is_dir():
        return []
    return sorted(common.list_images(category=category))


def protected_energy(img: np.ndarray, carry: tuple) -> np.ndarray:
    """EnergyFn for shrink_width(..., carry=(protect,)): default energy with protect weights.

    img: (H, W[, C]) current image; carry[0]: (H, W) bool protect mask, cut along with the image.
    Returns (H, W) float64.
    """
    return apply_mask_weights(compute_energy(img), protect_mask=carry[0])


def _load_protect(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """Protect mask for a photo, or None if missing or of the wrong size."""
    mask = common.load_mask(path, kind="protect")
    if mask is None:
        return None
    if mask.shape != shape:
        print(
            f"  protect mask has shape {mask.shape}, image is {shape}; skipping the masked panel."
        )
        return None
    return mask


def run_photo(path: Path) -> tuple:
    """Produce the D7 figures for one photo and return its row for d7_limits.csv."""
    name = path.stem
    img = common.load_image(path)
    h, w = img.shape[:2]
    k = round(SHRINK * w)
    print(f"D7: {name} ({w}x{h}), -{k} cols (-30%)")

    t0 = time.perf_counter()
    res = carve.shrink_width(img, k)
    seconds = time.perf_counter() - t0
    protect_px = removed_plain = removed_masked = seconds_masked = ""
    scaled = baselines.scale(img, h, w - k)
    images = [img, res.image, scaled]
    labels = [
        f"original {_size(img)}",
        f"seam carving -30% {_size(res.image)}",
        f"scale -30% {_size(scaled)}",
    ]
    seam_images = [viz.draw_seams(img, res.removed, color=SEAM_COLOR)]
    seam_labels = [f"{k} removed seams {_size(img)}"]

    protect = _load_protect(path, (h, w))
    if protect is None:
        print("  no usable protect mask; 3 panels.")
    else:
        t0 = time.perf_counter()
        res_p = carve.shrink_width(img, k, energy_fn=protected_energy, carry=(protect,))
        seconds_masked = round(time.perf_counter() - t0, 3)
        images.append(res_p.image)
        labels.append(f"seam carving + protect mask -30% {_size(res_p.image)}")
        seam_images.append(viz.draw_seams(img, res_p.removed, color=SEAM_COLOR))
        seam_labels.append(f"{k} removed seams, protect mask {_size(img)}")
        protect_px = int(np.count_nonzero(protect))
        removed_masked = int(np.count_nonzero(res_p.removed & protect))
        removed_plain = int(np.count_nonzero(res.removed & protect))
        print(
            f"  protected pixels removed: {removed_masked} of {protect_px} "
            f"(seam carving without mask: {removed_plain})"
        )

    common.save_figure(f"d7_limits_{name}", viz.comparison_grid(images, labels, cols=len(images)))
    common.save_figure(
        f"d7_seams_{name}", viz.comparison_grid(seam_images, seam_labels, cols=len(seam_images))
    )
    category = name.split("_")[0]
    return (
        name,
        category,
        w,
        h,
        k,
        round(seconds, 3),
        protect_px,
        removed_plain,
        removed_masked,
        seconds_masked,
    )


def main() -> None:
    """Run D7 on every insan, mimari and yogun photo; skip categories without photos."""
    rows = []
    for category in CATEGORIES:
        paths = _photos(category)
        if not paths:
            print(f"D7: no '{category}' photos in {common.IMAGES_DIR}; skipping this category.")
            continue
        for path in paths:
            rows.append(run_photo(path))
    if rows:
        out = common.save_table("d7_limits", rows, TABLE_HEADER)
        print(f"D7: wrote {out}")
    print(f"D7: processed {len(rows)} photo(s).")


if __name__ == "__main__":
    main()
