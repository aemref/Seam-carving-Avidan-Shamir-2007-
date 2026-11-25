"""D3: how much content is preserved? Paper Sec. 3.1 and Fig. 3. Owner: Mirza.

Measure: the mean of the ORIGINAL energy map ``e0`` over the pixels that are kept,
``e0[kept].mean()``; the energy is not recomputed on the result. Random removal leaves this
mean unchanged on average; content-aware removal raises it.

Four strategies remove 0 to 50 % of the width in 5 % steps:
best crop (one window), column removal, seam carving (``removed`` map) and per-row pixel
removal. The kept masks of the baselines come from ``seamcarving.baselines`` itself, applied
to a column-index image, so they match the baseline images exactly.

Photos: the first ``manzara`` and the first ``mimari`` photo.

Outputs:
- ``results/figures/d3_energy_preservation_<name>.png``: matplotlib line plot.
- ``results/figures/d3_strategies_<name>.png``: the four images at -30 % (paper Fig. 2).
- ``results/tables/d3_energy_preservation.csv``: image, percent, strategy, mean energy.

A missing ``manzara`` or ``mimari`` photo is reported and skipped; without any photo the
script writes nothing.
Run with ``python -m experiments.d3_energy_preservation``.
"""

from pathlib import Path

import numpy as np

from experiments.common import list_images, load_image, save_figure, save_table
from seamcarving.baselines import best_crop, remove_columns, remove_pixels_per_row
from seamcarving.carve import shrink_width
from seamcarving.energy import compute_energy
from seamcarving.viz import comparison_grid

CATEGORIES = ("manzara", "mimari")
PERCENTS = tuple(range(0, 51, 5))
STRATEGY_PERCENT = 30
STRATEGIES = ("best crop", "column removal", "seam carving", "pixel removal")


def _n_remove(size: int, percent: int) -> int:
    """Number of columns to remove for ``percent`` of ``size``."""
    return round(size * percent / 100)


def _column_index(h: int, w: int) -> np.ndarray:
    """(H, W) int64 image whose value is the column index."""
    return np.tile(np.arange(w, dtype=np.int64), (h, 1))


def _mask_from_columns(kept_cols: np.ndarray, w: int) -> np.ndarray:
    """bool (H, W) mask from an (H, n) array of kept original columns per row."""
    h = kept_cols.shape[0]
    mask = np.zeros((h, w), dtype=bool)
    mask[np.arange(h)[:, None], kept_cols] = True
    return mask


def _best_crop_kept(e0: np.ndarray, k: int) -> np.ndarray:
    """Kept mask of ``best_crop`` to width W-k (full height)."""
    h, w = e0.shape
    if k == 0:
        return np.ones((h, w), dtype=bool)
    return _mask_from_columns(best_crop(_column_index(h, w), h, w - k, energy=e0), w)


def _remove_columns_kept(e0: np.ndarray, k: int) -> np.ndarray:
    """Kept mask of ``remove_columns`` with k columns."""
    h, w = e0.shape
    return _mask_from_columns(remove_columns(_column_index(h, w), k, energy=e0), w)


def _pixel_removal_kept(e0: np.ndarray, k: int) -> np.ndarray:
    """Kept mask of ``remove_pixels_per_row`` with k pixels per row."""
    h, w = e0.shape
    return _mask_from_columns(remove_pixels_per_row(_column_index(h, w), k, energy=e0), w)


def _cut(img: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Keep the ``kept`` pixels of ``img`` (H, W[, C]); every row keeps the same count."""
    h = kept.shape[0]
    return img[kept].reshape((h, int(kept[0].sum())) + img.shape[2:])


def _kept_masks(e0: np.ndarray, order: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Kept masks of the four strategies for k removed columns."""
    return {
        "best crop": _best_crop_kept(e0, k),
        "column removal": _remove_columns_kept(e0, k),
        "seam carving": ~((order >= 0) & (order < k)),
        "pixel removal": _pixel_removal_kept(e0, k),
    }


def _plot(name: str, title: str, curves: dict[str, list[float]]) -> None:
    """Save a matplotlib line plot of mean energy against removed width."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
    for strategy in STRATEGIES:
        ax.plot(PERCENTS, curves[strategy], marker="o", markersize=3, label=strategy)
    ax.set_xlabel("width removed (%)")
    ax.set_ylabel("mean original energy of kept pixels")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    bgr = np.ascontiguousarray(rgba[:, :, 2::-1])  # RGBA -> BGR
    print(f"  wrote {save_figure(name, bgr)}")


def _save_strategies(name: str, img: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    """Save a grid of the original and the four strategy images as ``name``.png."""
    images = [img] + [_cut(img, masks[s]) for s in STRATEGIES]
    names = ["original", *STRATEGIES]
    labels = [f"{n} {im.shape[1]}x{im.shape[0]}" for n, im in zip(names, images, strict=True)]
    grid = comparison_grid(images, labels)
    print(f"  wrote {save_figure(name, grid)}")


def _run_photo(path: Path, rows: list[list]) -> None:
    """Measure all strategies and widths for one photo; save its figures."""
    img = load_image(path)
    e0 = compute_energy(img)
    w = e0.shape[1]
    ks = {p: _n_remove(w, p) for p in PERCENTS}
    # One seam carving run; the first k seams of it equal a k-seam run.
    order = shrink_width(img, max(ks.values())).order
    curves: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    for p in PERCENTS:
        masks = _kept_masks(e0, order, ks[p])
        for strategy in STRATEGIES:
            value = float(e0[masks[strategy]].mean())
            curves[strategy].append(value)
            rows.append([path.name, p, ks[p], strategy, f"{value:.4f}"])
    _plot(f"d3_energy_preservation_{path.stem}", path.stem, curves)
    k30 = _n_remove(w, STRATEGY_PERCENT)
    _save_strategies(f"d3_strategies_{path.stem}", img, _kept_masks(e0, order, k30))


def main() -> None:
    """Run D3 on the first manzara and first mimari photo; skip missing ones."""
    photos = []
    for category in CATEGORIES:
        found = list_images(category=category)
        if found:
            photos.append(found[0])
        else:
            print(f"D3: no '{category}' photo under data/images; skipping it.")
    if not photos:
        print("D3: no photos; nothing to do.")
        return
    rows: list[list] = []
    for path in photos:
        print(f"D3: {path.name}")
        _run_photo(path, rows)
    header = ["image", "percent", "columns_removed", "strategy", "mean_energy"]
    print(f"D3: wrote {save_table('d3_energy_preservation', rows, header)}")


if __name__ == "__main__":
    main()
