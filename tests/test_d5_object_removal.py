"""Tests for experiments/d5_object_removal.py on a tiny synthetic dataset in a temporary folder."""

import csv

import cv2
import numpy as np
import pytest

from experiments import common, d5_object_removal

IMAGE_SHAPE = (24, 36)
OBJECT_REGION = (slice(8, 16), slice(10, 16))
PROTECT_REGION = (slice(6, 18), slice(24, 29))
BACKGROUND_VALUE = 60
OBJECT_VALUE = 230
WITH_PROTECT = "nesne_01_berkay"
WITHOUT_PROTECT = "nesne_02_berkay"
WITHOUT_REMOVE = "nesne_03_berkay"
ROWS_PER_SCENE_WITH_PROTECT = 3  # removed, restored, removed_without_protect
ROWS_PER_SCENE_WITHOUT_PROTECT = 2  # removed, restored
DILATE_ROWS = 2  # dilate 0 and dilate DILATE_PIXELS on the first scene


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


def write_photo(dataset, stem):
    image = np.full((*IMAGE_SHAPE, 3), BACKGROUND_VALUE, dtype=np.uint8)
    image[OBJECT_REGION] = OBJECT_VALUE
    assert cv2.imwrite(str(dataset / "images" / f"{stem}.jpg"), image)


def write_mask(dataset, stem, kind, region):
    mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    mask[region] = 255
    assert cv2.imwrite(str(dataset / "masks" / f"{stem}_{kind}.png"), mask)


def add_scene(dataset, stem, with_remove=True, with_protect=False):
    write_photo(dataset, stem)
    if with_remove:
        write_mask(dataset, stem, "remove", OBJECT_REGION)
    if with_protect:
        write_mask(dataset, stem, "protect", PROTECT_REGION)


def figure_names(dataset):
    return sorted(path.name for path in (dataset / "figures").iterdir())


def table_rows(dataset):
    with open(dataset / "tables" / f"{d5_object_removal.TABLE_NAME}.csv", newline="") as file:
        return list(csv.DictReader(file))


def test_main_without_scenes_reports_and_writes_nothing(dataset, capsys):
    d5_object_removal.main()

    assert "nothing to do" in capsys.readouterr().out
    assert figure_names(dataset) == []
    assert not any((dataset / "tables").iterdir())


def test_main_writes_all_figures_for_scene_with_both_masks(dataset):
    add_scene(dataset, WITH_PROTECT, with_protect=True)

    d5_object_removal.main()

    assert figure_names(dataset) == [
        f"d5_dilate_{WITH_PROTECT}.png",
        f"d5_object_removal_{WITH_PROTECT}.png",
        f"d5_protect_{WITH_PROTECT}.png",
    ]


def test_scene_without_protect_mask_gets_no_protect_figure(dataset):
    add_scene(dataset, WITHOUT_PROTECT)

    d5_object_removal.main()

    assert f"d5_protect_{WITHOUT_PROTECT}.png" not in figure_names(dataset)
    assert f"d5_object_removal_{WITHOUT_PROTECT}.png" in figure_names(dataset)


def test_photo_without_removal_mask_is_skipped_with_message(dataset, capsys):
    add_scene(dataset, WITHOUT_REMOVE, with_remove=False)
    add_scene(dataset, WITHOUT_PROTECT)

    d5_object_removal.main()

    assert f"skipping {WITHOUT_REMOVE}.jpg" in capsys.readouterr().out
    assert {row["scene"] for row in table_rows(dataset)} == {WITHOUT_PROTECT}


def test_table_has_one_row_per_run(dataset):
    add_scene(dataset, WITH_PROTECT, with_protect=True)
    add_scene(dataset, WITHOUT_PROTECT)

    d5_object_removal.main()

    rows = table_rows(dataset)
    expected_count = ROWS_PER_SCENE_WITH_PROTECT + ROWS_PER_SCENE_WITHOUT_PROTECT + DILATE_ROWS
    assert len(rows) == expected_count
    assert list(rows[0]) == list(d5_object_removal.CSV_HEADER)


def test_table_row_records_removal_of_painted_object(dataset):
    add_scene(dataset, WITHOUT_PROTECT)
    object_height = OBJECT_REGION[0].stop - OBJECT_REGION[0].start
    object_width = OBJECT_REGION[1].stop - OBJECT_REGION[1].start
    object_area_percent = 100 * object_height * object_width / (IMAGE_SHAPE[0] * IMAGE_SHAPE[1])

    d5_object_removal.main()

    removed = next(row for row in table_rows(dataset) if row["variant"] == "removed")
    assert removed["direction"] == "vertical"
    assert int(removed["n_seams"]) >= object_width
    assert float(removed["mask_area_percent"]) == round(object_area_percent, 2)
    assert removed["protect_mask"] == "no" and removed["restore_size"] == "no"
    assert removed["pixels_left"] == "0"


def test_table_records_removal_pixels_kept_by_protection(dataset):
    trap_below_object = (slice(16, 17), slice(4, 22))
    write_photo(dataset, WITH_PROTECT)
    write_mask(dataset, WITH_PROTECT, "remove", OBJECT_REGION)
    write_mask(dataset, WITH_PROTECT, "protect", trap_below_object)

    d5_object_removal.main()

    pixels_left = {row["variant"]: int(row["pixels_left"]) for row in table_rows(dataset)}
    assert pixels_left["removed"] > 0
    assert pixels_left["removed_without_protect"] == 0


def test_mask_overlay_tints_only_masked_pixels():
    image = np.full((*IMAGE_SHAPE, 3), BACKGROUND_VALUE, dtype=np.uint8)
    remove, protect = np.zeros(IMAGE_SHAPE, bool), np.zeros(IMAGE_SHAPE, bool)
    remove[OBJECT_REGION] = True
    protect[PROTECT_REGION] = True

    overlay = d5_object_removal.mask_overlay(image, remove, protect)

    unmasked = ~(remove | protect)
    np.testing.assert_array_equal(overlay[unmasked], image[unmasked])
    blue, green, red = (int(value) for value in overlay[OBJECT_REGION][0, 0])
    assert green > BACKGROUND_VALUE > max(blue, red)
    blue, green, red = (int(value) for value in overlay[PROTECT_REGION][0, 0])
    assert red > BACKGROUND_VALUE > max(blue, green)
