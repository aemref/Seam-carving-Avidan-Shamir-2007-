"""Tests for experiments.common, run against small files in a temporary folder."""

import cv2
import numpy as np
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


def write_image(path, img, ext=".png"):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ext, img)
    assert ok
    buf.tofile(path)


@pytest.fixture
def no_path_io(monkeypatch):
    """Fail if cv2.imread / cv2.imwrite are used: they break on non-ASCII paths on Windows."""

    def forbidden(*args, **kwargs):
        raise AssertionError("use np.fromfile + cv2.imdecode / cv2.imencode + tofile")

    monkeypatch.setattr(cv2, "imread", forbidden)
    monkeypatch.setattr(cv2, "imwrite", forbidden)


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


# --- load_image --------------------------------------------------------------


def test_load_image_roundtrip_non_ascii_path(dirs, no_path_io):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    path = dirs["IMAGES_DIR"] / "nesne_01_emre_ğüşiöç.png"
    write_image(path, img)
    for arg in (path, str(path)):
        out = common.load_image(arg)
        assert out.dtype == np.uint8
        assert out.shape == (5, 7, 3)
        np.testing.assert_array_equal(out, img)


def test_load_image_bgr_channel_order(dirs):
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[..., 2] = 255  # pure red in BGR
    path = dirs["IMAGES_DIR"] / "red.png"
    write_image(path, img)
    np.testing.assert_array_equal(common.load_image(path), img)


def test_load_image_jpeg_and_gray(dirs):
    gray = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6) * 10
    jpg = dirs["IMAGES_DIR"] / "manzara_01_emre.jpg"
    write_image(jpg, np.dstack([gray] * 3), ext=".jpg")
    out = common.load_image(jpg)
    assert out.dtype == np.uint8 and out.shape == (4, 6, 3)
    png = dirs["IMAGES_DIR"] / "gray.png"
    write_image(png, gray)
    out = common.load_image(png)
    assert out.shape == (4, 6, 3)
    for ch in range(3):
        np.testing.assert_array_equal(out[..., ch], gray)


def test_load_image_errors(dirs):
    images = dirs["IMAGES_DIR"]
    with pytest.raises(FileNotFoundError):
        common.load_image(images / "missing.jpg")
    touch(images, "empty.jpg")
    with pytest.raises(ValueError, match="decode"):
        common.load_image(images / "empty.jpg")
    (images / "broken.jpg").write_bytes(b"not an image at all")
    with pytest.raises(ValueError, match="decode"):
        common.load_image(images / "broken.jpg")
    with pytest.raises(FileNotFoundError):
        common.load_image(images)
