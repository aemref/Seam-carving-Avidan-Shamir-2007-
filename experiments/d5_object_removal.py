"""D5: how cleanly does mask-driven seam carving remove an object? Owner: Berkay.

Run with ``python -m experiments.d5_object_removal`` (no arguments, headless).

Scenes: every ``nesne`` photo that has a removal mask (``<stem>_remove.png``); its protection
mask is used when it exists. Photos without a removal mask are reported and skipped.

Outputs (``<name>`` is the photo's file stem, e.g. ``nesne_01_berkay``):
    results/figures/d5_object_removal_<name>.png  input; masks overlaid; object removed;
                                                  object removed and size restored (Fig. 11-12)
    results/figures/d5_protect_<name>.png         scenes with a protection mask only: masks;
                                                  removal without and with the protection mask
    results/figures/d5_dilate_<name>.png          first scene only: removal mask grown by 0 and
                                                  by DILATE_PIXELS pixels
    results/tables/d5_object_removal.csv          one row per removal run (see CSV_HEADER)

Every run uses the default energy (``compute_energy``) and seam finder. "removed" runs keep
the narrower size; "restored" runs insert the removed number of seams again. Times are one
``time.perf_counter`` measurement per run (no repeats: a run takes seconds, and the number is
meant as an order of magnitude next to the seam count, D6 measures runtime properly).
"""

import time
import warnings
from typing import NamedTuple

import numpy as np

from experiments import common
from seamcarving import object_removal, viz

DILATE_PIXELS = 5
OVERLAY_ALPHA = 0.45
REMOVE_COLOR_BGR = (0, 255, 0)
PROTECT_COLOR_BGR = (0, 0, 255)
TABLE_NAME = "d5_object_removal"
CSV_HEADER = (
    "scene",
    "variant",
    "protect_mask",
    "dilate",
    "restore_size",
    "mask_area_percent",
    "direction",
    "n_seams",
    "pixels_left",
    "seconds",
)


class Scene(NamedTuple):
    name: str
    image: np.ndarray  # (H, W, 3) BGR uint8
    remove: np.ndarray  # (H, W) bool
    protect: np.ndarray | None  # (H, W) bool


class Run(NamedTuple):
    variant: str
    protect: np.ndarray | None
    dilate: int
    restore_size: bool
    result: object_removal.RemovalResult
    pixels_left: int  # removal pixels kept because no seam reaches them without protection
    seconds: float


def main() -> None:
    scenes = load_scenes()
    if not scenes:
        print("D5: no 'nesne' photo with a removal mask yet; nothing to do.")
        return
    rows = []
    for scene in scenes:
        rows += [table_row(scene, run) for run in run_scene(scene)]
    rows += [table_row(scenes[0], run) for run in run_dilate_comparison(scenes[0])]
    path = common.save_table(TABLE_NAME, rows, CSV_HEADER)
    print(f"D5: {len(scenes)} scene(s), {len(rows)} runs -> {path}")


def load_scenes() -> list[Scene]:
    """Load every 'nesne' photo with a removal mask; report the ones without."""
    scenes = []
    for image_path in common.list_images(category="nesne"):
        remove = common.load_mask(image_path, "remove")
        if remove is None:
            print(f"D5: skipping {image_path.name}: no removal mask")
            continue
        protect = common.load_mask(image_path, "protect")
        scenes.append(Scene(image_path.stem, common.load_image(image_path), remove, protect))
    return scenes


def run_scene(scene: Scene) -> list[Run]:
    """Write the removal figure (and the protection figure if there is a protect mask)."""
    removed = timed_run(scene, "removed", scene.protect, restore_size=False)
    restored = timed_run(scene, "restored", scene.protect, restore_size=True)
    save_removal_figure(scene, removed, restored)
    if scene.protect is None:
        return [removed, restored]
    unprotected = timed_run(scene, "removed_without_protect", None, restore_size=False)
    save_protect_figure(scene, unprotected, removed)
    return [removed, restored, unprotected]


def run_dilate_comparison(scene: Scene) -> list[Run]:
    """Write the dilate figure for one scene: removal mask as painted vs grown."""
    plain = timed_run(scene, "dilate_0", scene.protect, restore_size=False)
    grown = timed_run(
        scene, f"dilate_{DILATE_PIXELS}", scene.protect, restore_size=False, dilate=DILATE_PIXELS
    )
    common.save_figure(
        f"d5_dilate_{scene.name}",
        viz.comparison_grid(
            [plain.result.image, grown.result.image],
            [run_label("dilate 0", plain), run_label(f"dilate {DILATE_PIXELS}", grown)],
        ),
    )
    return [plain, grown]


def timed_run(
    scene: Scene,
    variant: str,
    protect: np.ndarray | None,
    restore_size: bool,
    dilate: int = 0,
) -> Run:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        result = object_removal.remove_object(
            scene.image, scene.remove, protect, restore_size=restore_size, dilate=dilate
        )
        seconds = time.perf_counter() - start
    pixels_left = unreachable_pixel_count(caught)
    return Run(variant, protect, dilate, restore_size, result, pixels_left, seconds)


def unreachable_pixel_count(caught: list[warnings.WarningMessage]) -> int:
    """Sum UnreachablePixelsWarning counts; show every other caught warning as usual."""
    pixels_left = 0
    for warning in caught:
        if isinstance(warning.message, object_removal.UnreachablePixelsWarning):
            pixels_left += warning.message.pixels_left
        else:
            warnings.showwarning(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    return pixels_left


def save_removal_figure(scene: Scene, removed: Run, restored: Run) -> None:
    images = [
        scene.image,
        mask_overlay(scene.image, scene.remove, scene.protect),
        removed.result.image,
        restored.result.image,
    ]
    labels = ["input", "masks", run_label("removed", removed), "size restored"]
    common.save_figure(f"d5_object_removal_{scene.name}", viz.comparison_grid(images, labels))


def save_protect_figure(scene: Scene, unprotected: Run, protected: Run) -> None:
    images = [
        mask_overlay(scene.image, scene.remove, scene.protect),
        unprotected.result.image,
        protected.result.image,
    ]
    labels = [
        "masks",
        run_label("no protect mask", unprotected),
        run_label("with protect mask", protected),
    ]
    common.save_figure(f"d5_protect_{scene.name}", viz.comparison_grid(images, labels))


def run_label(title: str, run: Run) -> str:
    label = f"{title}: {run.result.n_seams} {run.result.direction} seams"
    if run.pixels_left:
        label += f", {run.pixels_left} px kept"
    return label


def mask_overlay(
    image: np.ndarray, remove: np.ndarray, protect: np.ndarray | None
) -> np.ndarray:
    """Blend the removal mask green and the protection mask red onto a BGR copy."""
    blended = image.astype(np.float64)
    for mask, color in ((remove, REMOVE_COLOR_BGR), (protect, PROTECT_COLOR_BGR)):
        if mask is not None:
            blended[mask] = (1 - OVERLAY_ALPHA) * blended[mask] + OVERLAY_ALPHA * np.array(color)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def table_row(scene: Scene, run: Run) -> tuple:
    return (
        scene.name,
        run.variant,
        "yes" if run.protect is not None else "no",
        run.dilate,
        "yes" if run.restore_size else "no",
        round(100 * float(scene.remove.mean()), 2),
        run.result.direction,
        run.result.n_seams,
        run.pixels_left,
        round(run.seconds, 3),
    )


if __name__ == "__main__":
    main()
