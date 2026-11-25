"""Tests for experiments/d2_shrink.py on a tiny synthetic dataset in a temporary folder."""

import csv

import cv2
import numpy as np
import pytest

from experiments import common, d2_shrink
from seamcarving.carve import shrink_height, shrink_width

IMAGE_SHAPE = (16, 24)


def _textured(seed: int = 0, shape: tuple[int, int] = IMAGE_SHAPE) -> np.ndarray:
    """(H, W, 3) uint8 image: flat background with a random textured block."""
    rng = np.random.default_rng(seed)
    img = np.full((*shape, 3), 90, dtype=np.uint8)
    h, w = shape
    rows, cols = slice(h // 4, 3 * h // 4), slice(w // 3, 2 * w // 3)
    img[rows, cols] = rng.integers(
        0, 256, size=(rows.stop - rows.start, cols.stop - cols.start, 3), dtype=np.uint8
    )
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
    assert cv2.imwrite(str(dataset / "images" / f"{stem}.png"), _textured(seed))
    # list_images looks for .jpg; PNG bytes under a .jpg name keep the pixels exact.
    (dataset / "images" / f"{stem}.png").rename(dataset / "images" / f"{stem}.jpg")


def _figure_names(dataset) -> list[str]:
    return sorted(path.name for path in (dataset / "figures").iterdir())


def _table_rows(dataset) -> list[dict]:
    with open(dataset / "tables" / "d2_shrink.csv", newline="") as file:
        return list(csv.DictReader(file))


def test_main_without_photos_reports_and_writes_nothing(dataset, capsys):
    d2_shrink.main()

    assert "nothing to do" in capsys.readouterr().out
    assert _figure_names(dataset) == []
    assert not any((dataset / "tables").iterdir())


def test_main_writes_expected_figures_and_rows(dataset):
    _write_photo(dataset, "manzara_01_test", seed=1)
    _write_photo(dataset, "insan_01_test", seed=2)
    _write_photo(dataset, "nesne_01_test", seed=3)  # not a D2 category: ignored

    d2_shrink.main()

    expected = {
        f"d2_shrink_{stem}_{p}.png"
        for stem in ("manzara_01_test", "insan_01_test")
        for p in d2_shrink.PERCENTS
    }
    expected |= {"d2_seams_manzara_01_test.png", "d2_seams_insan_01_test.png"}
    expected.add("d2_shrink_height_manzara_01_test.png")  # first photo in category order
    assert set(_figure_names(dataset)) == expected

    rows = _table_rows(dataset)
    assert len(rows) == 2 * len(d2_shrink.PERCENTS) + 1
    h, w = IMAGE_SHAPE
    for row in rows:
        size = w if row["direction"] == "width" else h
        assert int(row["seams"]) == round(size * int(row["percent"]) / 100)
        assert float(row["seam_carving_seconds"]) >= 0.0


def test_width_times_grow_with_percent(dataset):
    _write_photo(dataset, "mimari_01_test")

    d2_shrink.main()

    times = [
        float(r["seam_carving_seconds"]) for r in _table_rows(dataset) if r["direction"] == "width"
    ]
    assert times == sorted(times)


@pytest.mark.parametrize("k", [1, 4, 9])
def test_cut_of_first_k_seams_equals_direct_k_seam_run(k):
    img = _textured(seed=5)
    long_run = shrink_width(img, 12)
    removed = (long_run.order >= 0) & (long_run.order < k)

    cut = d2_shrink._cut(img, removed, axis=1)

    np.testing.assert_array_equal(cut, shrink_width(img, k).image)


def test_cut_along_height_matches_shrink_height():
    img = _textured(seed=6)
    res = shrink_height(img, 5)

    cut = d2_shrink._cut(img, res.removed, axis=0)

    np.testing.assert_array_equal(cut, res.image)


def test_n_remove_rounds_percent_of_size():
    assert d2_shrink._n_remove(800, 25) == 200
    assert d2_shrink._n_remove(600, 10) == 60
    assert d2_shrink._n_remove(24, 50) == 12
