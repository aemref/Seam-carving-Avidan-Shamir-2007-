"""D8: how does noise affect the seams, and does a pre-filter fix it?

Run: ``python -m experiments.d8_noise``

One photo gets Gaussian noise (sigma = 15) and salt-and-pepper noise (5 % of pixels).
Four cases are compared: clean; noisy; noisy with Gaussian pre-smoothing
(``blur_sigma=1.5``); noisy with a 3 x 3 median filter before the energy. The noisy
image itself is what gets carved; the pre-filter is applied only inside ``energy_fn``.

Outputs ``d8_noise_<type>.png`` (energy map and the seams removed at width -20 % for
each case, drawn on that case's input) and ``results/tables/d8_noise.csv`` (overlap of
each case's removed pixels with the clean case, intersection over union).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from experiments import common
from seamcarving import carve, viz
from seamcarving.energy import compute_energy, make_energy_fn

CATEGORIES = ("manzara", "mimari", "insan")  # the first one found is used
PERSON = "ahsen"
SHRINK = 0.20
GAUSS_SIGMA = 15.0
SALT_PEPPER_FRACTION = 0.05
BLUR_SIGMA = 1.5
MEDIAN_SIZE = 3


def pick_image() -> Path | None:
    """Our own photo from the first category that has one, otherwise any member's."""
    for person in (PERSON, None):
        for category in CATEGORIES:
            paths = common.list_images(category=category, person=person)
            if paths:
                return paths[0]
    return None


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive zero-mean Gaussian noise, clipped back to uint8."""
    noisy = img.astype(np.float64) + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def add_salt_pepper(img: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Set ``fraction`` of the pixels (all channels) to 0 or 255, half each."""
    out = img.copy()
    hit = rng.random(img.shape[:2]) < fraction
    salt = rng.random(img.shape[:2]) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out


def median_energy(img: np.ndarray, carry: tuple = ()) -> np.ndarray:
    """Sobel energy of the median-filtered image (the carved image stays unfiltered)."""
    return compute_energy(cv2.medianBlur(img, MEDIAN_SIZE))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two bool masks."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def as_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main() -> None:
    path = pick_image()
    if path is None:
        print("[d8] no photo found in data/images, nothing to do")
        return
    rng = np.random.default_rng(0)
    clean = common.load_image(path)
    h, w = clean.shape[:2]
    k = round(SHRINK * w)
    print(f"[d8] {path.stem}: {w}x{h}, removing {k} seams per case")

    clean_result = carve.shrink_width(clean, k, energy_fn=make_energy_fn())
    noisy_images = {
        "gauss": add_gaussian_noise(clean, GAUSS_SIGMA, rng),
        "saltpepper": add_salt_pepper(clean, SALT_PEPPER_FRACTION, rng),
    }
    blur_fn = make_energy_fn(blur_sigma=BLUR_SIGMA)

    rows = []
    for noise, noisy in noisy_images.items():
        cases = [
            ("clean", clean, make_energy_fn(), clean_result),
            ("noisy", noisy, make_energy_fn(), None),
            (f"noisy + gauss {BLUR_SIGMA}", noisy, blur_fn, None),
            (f"noisy + median {MEDIAN_SIZE}", noisy, median_energy, None),
        ]
        maps, seams, labels = [], [], []
        for label, src, energy_fn, result in cases:
            if result is None:
                result = carve.shrink_width(src, k, energy_fn=energy_fn)
            overlap = iou(result.removed, clean_result.removed)
            rows.append([path.stem, noise, label, f"{overlap:.4f}"])
            maps.append(as_bgr(viz.energy_to_image(energy_fn(src, ()))))
            seams.append(viz.draw_seams(src, result.removed))
            labels.append(label)
        common.save_figure(
            f"d8_noise_{noise}",
            viz.comparison_grid(
                maps + seams,
                [f"{lb} energy" for lb in labels]
                + [f"{lb} seams IoU {r[3]}" for lb, r in zip(labels, rows[-4:])],
                cols=len(cases),
            ),
        )

    out = common.save_table("d8_noise", rows, ["image", "noise", "case", "iou_with_clean"])
    print(f"[d8] wrote {out}")


if __name__ == "__main__":
    main()
