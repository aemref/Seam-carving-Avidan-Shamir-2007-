"""Tests for experiments.common, run against small files in a temporary folder."""

import pytest

from experiments import common


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Point the four directory constants at fresh folders under a non-ASCII path."""
    base = tmp_path / "Görüntü İşleme"
    paths = {
        "IMAGES_DIR": base / "data" / "images",
        "MASKS_DIR": base / "data" / "masks",
        "FIG_DIR": base / "results" / "figures",
        "TABLE_DIR": base / "results" / "tables",
    }
    for attr, path in paths.items():
        monkeypatch.setattr(common, attr, path)
    return paths


def touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")


# --- list_images -------------------------------------------------------------


def test_list_images_missing_or_empty_dir(dirs):
    assert common.list_images() == []
    dirs["IMAGES_DIR"].mkdir(parents=True)
    assert common.list_images() == []


def test_list_images_sorted_and_filtered(dirs):
    images = dirs["IMAGES_DIR"]
    touch(
        images,
        "nesne_01_mirza.jpg",
        "manzara_02_emre.jpg",
        "manzara_01_emre.jpg",
        "manzara_01_ahsen.jpg",
        "yogun_01_furkan.jpg",
    )
    names = [p.name for p in common.list_images()]
    assert names == [
        "manzara_01_ahsen.jpg",
        "manzara_01_emre.jpg",
        "manzara_02_emre.jpg",
        "nesne_01_mirza.jpg",
        "yogun_01_furkan.jpg",
    ]
    assert all(p.parent == images for p in common.list_images())
    assert [p.name for p in common.list_images(category="manzara", person="emre")] == [
        "manzara_01_emre.jpg",
        "manzara_02_emre.jpg",
    ]
    assert [p.name for p in common.list_images(person="mirza")] == ["nesne_01_mirza.jpg"]
    assert common.list_images(category="mimari") == []


def test_list_images_skips_names_off_pattern(dirs):
    images = dirs["IMAGES_DIR"]
    touch(
        images,
        "manzara_01_emre.jpg",
        "manzara_1_emre.jpg",
        "manzara_001_emre.jpg",
        "manzara_01_emre.png",
        # distinct numbers: macOS file systems are case-insensitive by default
        "manzara_05_emre.JPG",
        "Manzara_06_emre.jpg",
        "manzara_07_Emre.jpg",
        "deniz_01_emre.jpg",
        "manzara_01_emre_remove.png",
        "manzara_01_emre.jpg.bak",
        ".DS_Store",
        "README.md",
    )
    (images / "nesne_01_emre.jpg").mkdir()
    assert [p.name for p in common.list_images()] == ["manzara_01_emre.jpg"]


def test_list_images_rejects_unknown_filters(dirs):
    with pytest.raises(ValueError, match="category"):
        common.list_images(category="manzra")
    with pytest.raises(ValueError, match="person"):
        common.list_images(person="Emre")


def test_list_images_reads_dir_at_call_time(dirs, tmp_path, monkeypatch):
    other = tmp_path / "other"
    touch(other, "insan_01_berkay.jpg")
    monkeypatch.setattr(common, "IMAGES_DIR", other)
    assert common.list_images() == [other / "insan_01_berkay.jpg"]
