"""D4: seam insertion for enlarging (paper Sec. 4.3, Fig. 8; Sec. 4.4).

Question: why is the k-seam method needed for enlarging, and how does it compare to scaling?

For one ``manzara`` and one ``mimari`` photo (Furkan's own if present, else first by sorted name) this script writes:

- ``d4_enlarge_<name>.png``: original; naive "repeat best seam" +50%; k-seam +50%; scale +50%
- ``d4_seams_<name>.png``: the k chosen seams on the original; inserted pixels on the k-seam
  result; inserted pixels on the naive result
- ``d4_staged_<name>.png``: +75% in one step (max_step=1.0), in two steps (max_step=0.5), scale
- ``d4_amplify_<name>.png``: original and amplify_content(scale=1.3)
- ``results/tables/d4_enlarge.csv``: photo, method, seconds

Run: ``python -m experiments.d4_enlarge``
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from experiments import common
from seamcarving import baselines, carve, enlarge, viz
from seamcarving.energy import compute_energy
from seamcarving.seam import find_vertical_seam

CATEGORIES = ("manzara", "mimari")
GROW = 0.50  # D4 main comparison: +50% width
GROW_STAGED = 0.75  # staged comparison: +75% width (above the 50% limit of Sec. 4.3)
AMPLIFY_SCALE = 1.3
SEAM_COLOR = (0, 0, 255)  # BGR red
INSERT_COLOR = (0, 255, 0)  # BGR green


def _timed(fn: Callable, *args, **kwargs) -> tuple[object, float]:
    """Call fn(*args, **kwargs); return (result, wall-clock seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def _size(img: np.ndarray) -> str:
    """ASCII size label 'W x H'."""
    return f"{img.shape[1]}x{img.shape[0]}"


PREFERRED_PERSON = "furkan"


def _first_photo(category: str) -> Path | None:
    """Owner's (furkan) photo of a category if present, else first by sorted name; None if none."""
    if not common.IMAGES_DIR.is_dir():
        return None
    own = sorted(common.list_images(category=category, person=PREFERRED_PERSON))
    if own:
        return own[0]
    paths = sorted(common.list_images(category=category))
    return paths[0] if paths else None


def naive_enlarge_width(img: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Wrong enlarging, kept only for D4: find the best seam, duplicate it, repeat k times.

    Each step recomputes the energy of the growing image, finds the single minimum seam and
    inserts one copy of it with ``insert_vertical_seams``. The new averaged pixels have low energy,
    so the next step picks (almost) the same seam again and a stretched strip appears (Fig. 8).

    img: (H, W[, C]) uint8. Returns (out (H, W+k[, C]), inserted (H, W+k) bool), where
    ``inserted`` marks every pixel that was added by one of the k steps.
    """
    h = img.shape[0]
    rows = np.arange(h)
    cur = img.copy()
    inserted = np.zeros((h, img.shape[1]), dtype=bool)
    for _ in range(k):
        seam = find_vertical_seam(compute_energy(cur))
        chosen = np.zeros(cur.shape[:2], dtype=bool)
        chosen[rows, seam] = True
        new_pos = enlarge.inserted_mask(chosen)  # (H, W_cur + 1)
        grown = np.empty(new_pos.shape, dtype=bool)
        grown[new_pos] = True
        grown[~new_pos] = inserted.ravel()  # exactly one new pixel per row: row-major order kept
        inserted = grown
        cur = enlarge.insert_vertical_seams(cur, chosen)
    return cur, inserted


def run_photo(path: Path) -> list[tuple[str, str, float]]:
    """Produce all D4 figures for one photo; return timing rows (photo, method, seconds)."""
    name = path.stem
    img = common.load_image(path)
    h, w = img.shape[:2]
    k = int(GROW * w)  # floor, so the default max_step=0.5 enlarges in a single step
    k_staged = int(GROW_STAGED * w)
    rows: list[tuple[str, str, float]] = []

    def record(method: str, seconds: float) -> None:
        rows.append((name, method, round(seconds, 3)))
        print(f"  {method:<22s} {seconds:8.2f} s")

    print(f"D4: {name} ({w}x{h}), +{k} cols (+50%), +{k_staged} cols (+75%)")

    # --- +50%: naive vs k-seam vs scale ------------------------------------------------------
    (naive, naive_ins), t = _timed(naive_enlarge_width, img, k)
    record("naive_repeat_+50", t)
    kseam, t = _timed(enlarge.enlarge_width, img, k)
    record("kseam_+50", t)
    scaled, t = _timed(baselines.scale, img, h, w + k)
    record("scale_+50", t)

    common.save_figure(
        f"d4_enlarge_{name}",
        viz.comparison_grid(
            [img, naive, kseam, scaled],
            [
                f"original {_size(img)}",
                f"naive repeat best seam +50% {_size(naive)}",
                f"k-seam +50% {_size(kseam)}",
                f"scale +50% {_size(scaled)}",
            ],
            cols=2,
        ),
    )

    # --- seams: the first k removal seams, mapped to original coordinates ---------------------
    removed = carve.shrink_width(img, k).removed  # (H, W) bool, k True per row
    if not np.array_equal(enlarge.insert_vertical_seams(img, removed), kseam):
        print("  warning: enlarge_width output differs from inserting shrink_width(img, k).removed")
    common.save_figure(
        f"d4_seams_{name}",
        viz.comparison_grid(
            [
                viz.draw_seams(img, removed, color=SEAM_COLOR),
                viz.draw_seams(kseam, enlarge.inserted_mask(removed), color=INSERT_COLOR),
                viz.draw_seams(naive, naive_ins, color=INSERT_COLOR),
            ],
            [
                f"k={k} chosen seams on original {_size(img)}",
                f"k-seam inserted pixels {_size(kseam)}",
                f"naive inserted pixels {_size(naive)}",
            ],
            cols=3,
        ),
    )

    # --- +75%: one step vs two steps vs scale (Sec. 4.3) ---------------------------------------
    one_step, t = _timed(enlarge.enlarge_width, img, k_staged, max_step=1.0)
    record("kseam_+75_max_step_1.0", t)
    two_step, t = _timed(enlarge.enlarge_width, img, k_staged, max_step=0.5)
    record("kseam_+75_max_step_0.5", t)
    scaled75, t = _timed(baselines.scale, img, h, w + k_staged)
    record("scale_+75", t)
    common.save_figure(
        f"d4_staged_{name}",
        viz.comparison_grid(
            [one_step, two_step, scaled75],
            [
                f"k-seam +75% one step (max_step=1.0) {_size(one_step)}",
                f"k-seam +75% staged (max_step=0.5) {_size(two_step)}",
                f"scale +75% {_size(scaled75)}",
            ],
            cols=3,
        ),
    )

    # --- content amplification (Sec. 4.4) -------------------------------------------------------
    amplified, t = _timed(enlarge.amplify_content, img, scale=AMPLIFY_SCALE)
    record(f"amplify_{AMPLIFY_SCALE}", t)
    common.save_figure(
        f"d4_amplify_{name}",
        viz.comparison_grid(
            [img, amplified],
            [f"original {_size(img)}", f"amplify_content scale={AMPLIFY_SCALE} {_size(amplified)}"],
            cols=2,
        ),
    )
    return rows


def main() -> None:
    """Run D4 on the first manzara and the first mimari photo; skip what is missing."""
    rows: list[tuple[str, str, float]] = []
    for category in CATEGORIES:
        path = _first_photo(category)
        if path is None:
            print(f"D4: no '{category}' photo in {common.IMAGES_DIR}; skipping this category.")
            continue
        rows.extend(run_photo(path))
    if not rows:
        print("D4: nothing to do (no photos); no figures or table written.")
        return
    out = common.save_table("d4_enlarge", rows, ["photo", "method", "seconds"])
    print(f"D4: wrote {out}")


if __name__ == "__main__":
    main()
