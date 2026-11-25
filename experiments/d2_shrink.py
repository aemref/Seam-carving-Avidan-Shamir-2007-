"""D2: is seam carving better than scaling and cropping for shrinking? Owner: Mirza.

For every ``manzara``, ``insan`` and ``mimari`` photo the width is reduced by 10, 25 and 50 %
with seam carving, ``scale``, ``center_crop`` and ``best_crop`` (paper Fig. 5).

Outputs:
- ``results/figures/d2_shrink_<name>_<percent>.png``: original and the four results.
- ``results/figures/d2_shrink_height_<name>.png``: the same for height -25 % (first photo).
- ``results/figures/d2_seams_<name>.png``: the seams removed at -25 % drawn on the original.
- ``results/tables/d2_shrink.csv``: photo, direction, percent, seam carving time.

Seam carving runs once per photo with the largest k (50 %). Seams are removed one after
another, so the first k_p seams of that run are exactly the result of a k_p run; the 10 % and
25 % images are cut from the original with ``order < k_p``. Their time is the time the run
needed to finish k_p seams. Without photos the script prints a message and writes nothing.

Run with ``python -m experiments.d2_shrink``.
"""

import time
from pathlib import Path

import numpy as np

from experiments.common import list_images, load_image, save_figure, save_table
from seamcarving.baselines import best_crop, center_crop, scale
from seamcarving.carve import shrink_height, shrink_width
from seamcarving.energy import compute_energy
from seamcarving.viz import comparison_grid, draw_seams

CATEGORIES = ("manzara", "insan", "mimari")
PERCENTS = (10, 25, 50)
SEAMS_PERCENT = 25
HEIGHT_PERCENT = 25


def _n_remove(size: int, percent: int) -> int:
    """Number of columns (or rows) to remove for ``percent`` of ``size``."""
    return round(size * percent / 100)


class _StepTimer:
    """EnergyFn wrapper that records when each carving step starts."""

    def __init__(self) -> None:
        self.starts: list[float] = []

    def __call__(self, img: np.ndarray, carry: tuple) -> np.ndarray:
        self.starts.append(time.perf_counter())
        return compute_energy(img)


def _cut(img: np.ndarray, removed: np.ndarray, axis: int) -> np.ndarray:
    """Drop the ``removed`` pixels of ``img`` (H, W[, C]); axis 1 = width, 0 = height."""
    h, w = removed.shape
    if axis == 1:
        n_left = w - int(removed[0].sum())
        return img[~removed].reshape((h, n_left) + img.shape[2:])
    swapped = np.swapaxes(img, 0, 1)
    return np.swapaxes(_cut(swapped, removed.T, 1), 0, 1)


def _label(name: str, img: np.ndarray) -> str:
    """ASCII panel label: method name and size as W x H."""
    return f"{name} {img.shape[1]}x{img.shape[0]}"


def _save_grid(name: str, panels: list[tuple[str, np.ndarray]]) -> None:
    """Save a labelled comparison grid of ``(method, image)`` panels as ``name``.png."""
    grid = comparison_grid([p[1] for p in panels], [_label(*p) for p in panels])
    print(f"  wrote {save_figure(name, grid)}")


def _shrink_width_photo(path: Path, rows: list[list]) -> None:
    """Run all width percentages for one photo; append table rows, save figures."""
    img = load_image(path)
    h, w = img.shape[:2]
    energy = compute_energy(img)
    ks = {p: _n_remove(w, p) for p in PERCENTS}
    timer = _StepTimer()
    t0 = time.perf_counter()
    res = shrink_width(img, max(ks.values()), energy_fn=timer)
    t_end = time.perf_counter()
    k_max = max(ks.values())
    for p in PERCENTS:
        k = ks[p]
        seconds = (timer.starts[k] if k < k_max else t_end) - t0
        rows.append([path.name, "width", p, k, f"{seconds:.3f}"])
        removed = (res.order >= 0) & (res.order < k)
        carved = _cut(img, removed, axis=1)
        new_w = w - k
        _save_grid(
            f"d2_shrink_{path.stem}_{p}",
            [
                ("original", img),
                ("seam carving", carved),
                ("scale", scale(img, h, new_w)),
                ("center crop", center_crop(img, h, new_w)),
                ("best crop", best_crop(img, h, new_w, energy=energy)),
            ],
        )
        if p == SEAMS_PERCENT:
            seams = draw_seams(img, removed)
            print(f"  wrote {save_figure(f'd2_seams_{path.stem}', seams)}")


def _shrink_height_photo(path: Path, rows: list[list]) -> None:
    """Height -25 % for one photo; append a table row and save the grid."""
    img = load_image(path)
    h, w = img.shape[:2]
    k = _n_remove(h, HEIGHT_PERCENT)
    t0 = time.perf_counter()
    res = shrink_height(img, k)
    seconds = time.perf_counter() - t0
    rows.append([path.name, "height", HEIGHT_PERCENT, k, f"{seconds:.3f}"])
    new_h = h - k
    _save_grid(
        f"d2_shrink_height_{path.stem}",
        [
            ("original", img),
            ("seam carving", res.image),
            ("scale", scale(img, new_h, w)),
            ("center crop", center_crop(img, new_h, w)),
            ("best crop", best_crop(img, new_h, w, energy=compute_energy(img))),
        ],
    )


def main() -> None:
    """Run D2 on every photo found; skip cleanly when there are none."""
    photos = []
    for category in CATEGORIES:
        found = list_images(category=category)
        if not found:
            print(f"D2: no '{category}' photo under data/images; skipping that category.")
        photos.extend(found)
    if not photos:
        print(f"D2: no photos in categories {CATEGORIES} under data/images; nothing to do.")
        return
    rows: list[list] = []
    for path in photos:
        print(f"D2: {path.name} (width -{', -'.join(map(str, PERCENTS))} %)")
        _shrink_width_photo(path, rows)
    print(f"D2: {photos[0].name} (height -{HEIGHT_PERCENT} %)")
    _shrink_height_photo(photos[0], rows)
    header = ["image", "direction", "percent", "seams", "seam_carving_seconds"]
    print(f"D2: wrote {save_table('d2_shrink', rows, header)}")


if __name__ == "__main__":
    main()
