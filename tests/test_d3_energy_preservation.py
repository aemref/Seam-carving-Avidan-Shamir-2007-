"""Tests for experiments/d3_energy_preservation.py on tiny synthetic images."""

import csv

import cv2
import numpy as np
import pytest

from experiments import common
from experiments import d3_energy_preservation as d3
from seamcarving.baselines import best_crop, remove_columns, remove_pixels_per_row
from seamcarving.carve import shrink_width
from seamcarving.energy import compute_energy

IMAGE_SHAPE = (14, 20)


def _textured(seed: int = 0) -> np.ndarray:
    """(H, W, 3) uint8 image: smooth gradient plus a random textured block."""
    rng = np.random.default_rng(seed)
    h, w = IMAGE_SHAPE
    img = np.tile(np.linspace(40, 160, w, dtype=np.uint8)[None, :, None], (h, 1, 3))
    img[3:11, 6:13] = rng.integers(0, 256, size=(8, 7, 3), dtype=np.uint8)
    return img


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    for constant, folder in (
        ("IMAGES_DIR", "images"),
        ("MASKS_DIR", "masks"),
        ("FIG_DIR", "figures"),
        ("TABLE_DIR", "tables"),
    ):
        (tmp_path / folder).mkdir()
        monkeypatch.setattr(common, constant, tmp_path / folder)
    return tmp_path


def _write_photo(dataset, stem: str, seed: int = 0) -> None:
    png = dataset / "images" / f"{stem}.png"
    assert cv2.imwrite(str(png), _textured(seed))
    png.rename(dataset / "images" / f"{stem}.jpg")  # lossless bytes under the .jpg name


def _figure_names(dataset) -> list[str]:
    return sorted(path.name for path in (dataset / "figures").iterdir())


def test_main_without_photos_reports_and_writes_nothing(dataset, capsys):
    d3.main()

    assert "nothing to do" in capsys.readouterr().out
    assert _figure_names(dataset) == []
    assert not any((dataset / "tables").iterdir())


def test_main_uses_first_manzara_and_first_mimari(dataset):
    _write_photo(dataset, "manzara_01_test", seed=1)
    _write_photo(dataset, "manzara_02_test", seed=2)
    _write_photo(dataset, "mimari_01_test", seed=3)

    d3.main()

    assert _figure_names(dataset) == [
        "d3_energy_preservation_manzara_01_test.png",
        "d3_energy_preservation_mimari_01_test.png",
        "d3_strategies_manzara_01_test.png",
        "d3_strategies_mimari_01_test.png",
    ]
    with open(dataset / "tables" / "d3_energy_preservation.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 * len(d3.PERCENTS) * len(d3.STRATEGIES)
    assert {r["image"] for r in rows} == {"manzara_01_test.jpg", "mimari_01_test.jpg"}


def test_missing_category_is_reported_and_skipped(dataset, capsys):
    _write_photo(dataset, "manzara_01_test")

    d3.main()

    assert "no 'mimari' photo" in capsys.readouterr().out
    assert len(_figure_names(dataset)) == 2


@pytest.mark.parametrize("k", [0, 3, 7])
def test_kept_masks_keep_w_minus_k_per_row_and_match_baselines(k):
    img = _textured()
    e0 = compute_energy(img)
    h, w = e0.shape
    order = shrink_width(img, 10).order

    masks = d3._kept_masks(e0, order, k)

    for mask in masks.values():
        assert mask.shape == (h, w)
        np.testing.assert_array_equal(mask.sum(axis=1), np.full(h, w - k))
    np.testing.assert_array_equal(d3._cut(img, masks["seam carving"]), shrink_width(img, k).image)
    if k:
        np.testing.assert_array_equal(
            d3._cut(img, masks["column removal"]), remove_columns(img, k, energy=e0)
        )
        np.testing.assert_array_equal(
            d3._cut(img, masks["pixel removal"]), remove_pixels_per_row(img, k, energy=e0)
        )
        np.testing.assert_array_equal(
            d3._cut(img, masks["best crop"]), best_crop(img, h, w - k, energy=e0)
        )


@pytest.mark.parametrize("k", [2, 5, 9])
def test_mean_energy_ordering_bounds(k):
    img = _textured(seed=4)
    e0 = compute_energy(img)
    order = shrink_width(img, 10).order
    means = {s: e0[m].mean() for s, m in d3._kept_masks(e0, order, k).items()}

    # Dropping each row's k lowest pixels maximises the mean among all strategies that
    # keep the same number of pixels per row; dropping the k lowest columns maximises it
    # among column subsets, and a crop is a column subset.
    assert means["pixel removal"] >= max(means.values()) - 1e-12
    assert means["column removal"] >= means["best crop"] - 1e-12
    # Content-aware removal keeps more energy than the image average.
    assert means["seam carving"] > e0.mean()


def test_all_strategies_equal_original_mean_at_zero():
    img = _textured()
    e0 = compute_energy(img)
    order = shrink_width(img, 5).order

    for mask in d3._kept_masks(e0, order, 0).values():
        assert e0[mask].mean() == pytest.approx(e0.mean())
