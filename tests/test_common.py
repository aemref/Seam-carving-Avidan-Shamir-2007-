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


# --- load_mask ---------------------------------------------------------------


@pytest.fixture
def photo(dirs):
    path = dirs["IMAGES_DIR"] / "nesne_01_emre.jpg"
    write_image(path, np.full((4, 6, 3), 128, dtype=np.uint8), ext=".jpg")
    return path


def test_load_mask_threshold_and_kinds(dirs, photo, no_path_io):
    values = np.array([[0, 127, 128, 255, 200, 1]] * 4, dtype=np.uint8)
    write_image(dirs["MASKS_DIR"] / "nesne_01_emre_remove.png", values)
    write_image(dirs["MASKS_DIR"] / "nesne_01_emre_protect.png", 255 - values)
    for arg in (photo, str(photo)):
        mask = common.load_mask(arg)
        assert mask.dtype == np.bool_ and mask.shape == (4, 6)
        np.testing.assert_array_equal(mask, values > 127)
    protect = common.load_mask(photo, kind="protect")
    np.testing.assert_array_equal(protect, (255 - values) > 127)


def test_load_mask_colour_file(dirs, photo):
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[:, :3] = 255
    write_image(dirs["MASKS_DIR"] / "nesne_01_emre_remove.png", bgr)
    expected = np.zeros((4, 6), dtype=bool)
    expected[:, :3] = True
    np.testing.assert_array_equal(common.load_mask(photo), expected)


def test_load_mask_missing_returns_none(dirs, photo):
    assert common.load_mask(photo) is None
    write_image(dirs["MASKS_DIR"] / "nesne_01_emre_remove.png", np.zeros((4, 6), np.uint8))
    assert common.load_mask(photo, kind="protect") is None
    assert common.load_mask(photo) is not None


def test_load_mask_errors(dirs, photo):
    with pytest.raises(ValueError, match="kind"):
        common.load_mask(photo, kind="delete")
    write_image(dirs["MASKS_DIR"] / "nesne_01_emre_remove.png", np.zeros((6, 4), np.uint8))
    with pytest.raises(ValueError, match="6x4"):
        common.load_mask(photo)


def test_load_mask_reads_dir_at_call_time(dirs, photo, tmp_path, monkeypatch):
    other = tmp_path / "other_masks"
    write_image(other / "nesne_01_emre_remove.png", np.full((4, 6), 255, np.uint8))
    assert common.load_mask(photo) is None
    monkeypatch.setattr(common, "MASKS_DIR", other)
    assert common.load_mask(photo).all()


# --- save_figure -------------------------------------------------------------


def read_back(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def test_save_figure_writes_png_and_creates_dir(dirs, no_path_io):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    before = img.copy()
    path = common.save_figure("d0_test_ğüş", img)
    assert path == dirs["FIG_DIR"] / "d0_test_ğüş.png"
    assert path.is_file()
    np.testing.assert_array_equal(read_back(path), img)
    np.testing.assert_array_equal(img, before)


def test_save_figure_gray_and_bgra(dirs):
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    np.testing.assert_array_equal(read_back(common.save_figure("gray", gray)), gray)
    bgra = np.zeros((3, 4, 4), dtype=np.uint8)
    bgra[..., 3] = 255
    np.testing.assert_array_equal(read_back(common.save_figure("bgra", bgra)), bgra)


def test_save_figure_converts_dtypes(dirs):
    floats = np.array([[-3.0, 0.4, 0.5, 1.5, 254.6, 300.0]])
    out = read_back(common.save_figure("floats", floats))
    np.testing.assert_array_equal(out, [[0, 0, 0, 2, 255, 255]])  # rint rounds half to even
    ints = np.array([[-1, 7, 256]], dtype=np.int64)
    np.testing.assert_array_equal(read_back(common.save_figure("ints", ints)), [[0, 7, 255]])
    mask = np.array([[True, False]])
    np.testing.assert_array_equal(read_back(common.save_figure("mask", mask)), [[255, 0]])


def test_save_figure_is_deterministic(dirs):
    img = np.random.default_rng(0).random((16, 16, 3)) * 255
    first = common.save_figure("det", img).read_bytes()
    second = common.save_figure("det", img).read_bytes()
    assert first == second


def test_save_figure_errors(dirs):
    img = np.zeros((2, 2), np.uint8)
    for name in ("", "sub/fig", "sub\\fig", "fig.png", ".."):
        with pytest.raises(ValueError, match="name"):
            common.save_figure(name, img)
    for bad in (np.zeros((2, 2, 2), np.uint8), np.zeros((2,), np.uint8), np.zeros((0, 3))):
        with pytest.raises(ValueError, match="expected"):
            common.save_figure("bad", bad)
    with pytest.raises(ValueError, match="NaN"):
        common.save_figure("nan", np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError, match="dtype"):
        common.save_figure("complex", np.zeros((2, 2), complex))
    assert not dirs["FIG_DIR"].exists() or not any(dirs["FIG_DIR"].iterdir())


def test_save_figure_reads_dir_at_call_time(dirs, tmp_path, monkeypatch):
    other = tmp_path / "other_figs"
    monkeypatch.setattr(common, "FIG_DIR", other)
    assert common.save_figure("x", np.zeros((2, 2), np.uint8)) == other / "x.png"


# --- save_table --------------------------------------------------------------


def test_save_table_writes_utf8_csv(dirs):
    rows = [
        ("manzara_01_emre", 25, 0.125, "ğüşİ"),
        ["nesne_01_mirza", np.int64(10), np.float64(1.5), None],
        ("a,b", 0, 'say "hi"', True),
    ]
    path = common.save_table("d0_table", (r for r in rows), ["photo", "pct", "seconds", "note"])
    assert path == dirs["TABLE_DIR"] / "d0_table.csv"
    assert path.read_bytes().decode("utf-8") == (
        "photo,pct,seconds,note\n"
        "manzara_01_emre,25,0.125,ğüşİ\n"
        "nesne_01_mirza,10,1.5,\n"
        '"a,b",0,"say ""hi""",True\n'
    )


def test_save_table_header_only_and_deterministic(dirs):
    path = common.save_table("empty", [], ("a", "b"))
    assert path.read_bytes() == b"a,b\n"
    rows = [(1, 2.0), (3, 4.5)]
    first = common.save_table("det", rows, ("a", "b")).read_bytes()
    assert common.save_table("det", rows, ("a", "b")).read_bytes() == first


def test_save_table_errors_write_nothing(dirs):
    with pytest.raises(ValueError, match="row 1 has 1 values"):
        common.save_table("bad", [(1, 2), (3,)], ("a", "b"))
    with pytest.raises(TypeError, match="string"):
        common.save_table("bad", ["ab"], ("a", "b"))
    with pytest.raises(ValueError, match="header"):
        common.save_table("bad", [], [])
    with pytest.raises(ValueError, match="name"):
        common.save_table("bad.csv", [], ("a",))
    assert not (dirs["TABLE_DIR"] / "bad.csv").exists()


def test_save_table_reads_dir_at_call_time(dirs, tmp_path, monkeypatch):
    other = tmp_path / "other_tables"
    monkeypatch.setattr(common, "TABLE_DIR", other)
    assert common.save_table("x", [(1,)], ("a",)) == other / "x.csv"
