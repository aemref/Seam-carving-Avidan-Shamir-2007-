"""D1: which derivative operator gives the better energy map and the better shrink?

Run: ``python -m experiments.d1_energy_operators``

For one ``manzara``, one ``mimari`` and one ``insan`` photo it writes
- ``d1_energy_maps_<name>.png``: the photo and the energy map of each method,
- ``d1_carved_<name>.png``: width -25 % with each method,
- ``d1_color_blur_<name>.png``: Sobel with gray, sum, gray + blur and l2 (maps and results),
and ``results/tables/d1_energy.csv`` with the energy time per method and the mean
original Sobel energy of the pixels each method keeps.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from experiments import common
from seamcarving import carve, viz
from seamcarving.energy import METHODS, compute_energy, make_energy_fn

CATEGORIES = ("manzara", "mimari", "insan")
PERSON = "ahsen"
SHRINK = 0.25
TIMING_REPEATS = 5

# Sobel variants compared in d1_color_blur_*: (label, compute_energy options).
VARIANTS = (
    ("sobel gray", {"method": "sobel", "color": "gray"}),
    ("sobel sum", {"method": "sobel", "color": "sum"}),
    ("sobel gray blur 1.5", {"method": "sobel", "color": "gray", "blur_sigma": 1.5}),
    ("sobel gray l2", {"method": "sobel", "color": "gray", "norm": "l2"}),
)


def pick_image(category: str) -> Path | None:
    """Our own photo of ``category`` if there is one, otherwise any team member's."""
    paths = common.list_images(category=category, person=PERSON) or common.list_images(
        category=category
    )
    return paths[0] if paths else None


def as_bgr(gray: np.ndarray) -> np.ndarray:
    """(H, W) uint8 -> (H, W, 3) uint8 so it can share a grid with photos."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def time_energy(img: np.ndarray, options: dict) -> float:
    """Median wall time of compute_energy in milliseconds."""
    runs = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        compute_energy(img, **options)
        runs.append(time.perf_counter() - start)
    return 1000.0 * float(np.median(runs))


def run_image(path: Path) -> list[list]:
    """Make the three figures for one photo and return its table rows."""
    name = path.stem
    img = common.load_image(path)
    h, w = img.shape[:2]
    k = round(SHRINK * w)
    e0 = compute_energy(img)  # reference: original Sobel energy
    size = f"{w - k}x{h}"
    print(f"[d1] {name}: {w}x{h}, removing {k} seams per method")

    rows = []
    results = {}
    settings = [(m, {"method": m}) for m in METHODS] + [
        (label, opts) for label, opts in VARIANTS if label != "sobel gray"
    ]
    for label, options in settings:
        result = carve.shrink_width(img, k, energy_fn=make_energy_fn(**options))
        results[label] = result
        rows.append(
            [
                name,
                label,
                f"{time_energy(img, options):.2f}",
                f"{e0[~result.removed].mean():.3f}",
                f"{e0.mean():.3f}",
            ]
        )
    results["sobel gray"] = results["sobel"]

    maps = [img] + [as_bgr(viz.energy_to_image(compute_energy(img, method=m))) for m in METHODS]
    common.save_figure(
        f"d1_energy_maps_{name}",
        viz.comparison_grid(maps, [f"original {w}x{h}", *METHODS], cols=3),
    )

    carved = [img] + [results[m].image for m in METHODS]
    common.save_figure(
        f"d1_carved_{name}",
        viz.comparison_grid(
            carved, [f"original {w}x{h}", *(f"{m} {size}" for m in METHODS)], cols=3
        ),
    )

    variant_maps = [as_bgr(viz.energy_to_image(compute_energy(img, **o))) for _, o in VARIANTS]
    variant_carved = [results[label].image for label, _ in VARIANTS]
    labels = [label for label, _ in VARIANTS]
    common.save_figure(
        f"d1_color_blur_{name}",
        viz.comparison_grid(
            variant_maps + variant_carved,
            [f"{lb} energy" for lb in labels] + [f"{lb} {size}" for lb in labels],
            cols=len(VARIANTS),
        ),
    )
    return rows


def main() -> None:
    rows = []
    for category in CATEGORIES:
        path = pick_image(category)
        if path is None:
            print(f"[d1] no '{category}' photo in data/images, skipping it")
            continue
        rows.extend(run_image(path))
    if not rows:
        print("[d1] no photos found, nothing to do")
        return
    out = common.save_table(
        "d1_energy",
        rows,
        ["image", "method", "energy_ms", "mean_kept_sobel_energy", "mean_all_sobel_energy"],
    )
    print(f"[d1] wrote {out}")


if __name__ == "__main__":
    main()
