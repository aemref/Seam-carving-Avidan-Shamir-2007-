"""Tests for tools/prepare_dataset.py on synthetic images in a temporary repo root."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_dataset.py"
_SPEC = importlib.util.spec_from_file_location("prepare_dataset", _PATH)
prep = importlib.util.module_from_spec(_SPEC)
sys.modules["prepare_dataset"] = prep
_SPEC.loader.exec_module(prep)

PERSON = "furkan"


def make_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Random BGR uint8 (h, w, 3) image."""
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def put_raw(root: Path, rel: str, h: int = 60, w: int = 80, seed: int = 0) -> Path:
    """Write a synthetic photo to data/raw/<PERSON>/<rel> and return its path."""
    path = root / "data" / "raw" / PERSON / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), make_image(h, w, seed))
    return path


def run(root: Path, *extra: str) -> int:
    return prep.main(["--person", PERSON, "--root", str(root), *extra])


def read_rows(root: Path) -> tuple[list[str], list[dict[str, str]]]:
    with (root / "data" / "metadata" / f"{PERSON}.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def image_names(root: Path) -> list[str]:
    images = root / "data" / "images"
    return sorted(p.name for p in images.iterdir()) if images.is_dir() else []


def test_naming_and_numbering_per_category(tmp_path: Path) -> None:
    put_raw(tmp_path, "nesne/b.jpg")
    put_raw(tmp_path, "nesne/a.JPG")
    put_raw(tmp_path, "manzara/x.png")
    put_raw(tmp_path, "mimari/y.jpeg")
    (tmp_path / "data" / "raw" / PERSON / "nesne" / ".DS_Store").write_bytes(b"junk")
    assert run(tmp_path) == 0
    assert image_names(tmp_path) == [
        "manzara_01_furkan.jpg",
        "mimari_01_furkan.jpg",
        "nesne_01_furkan.jpg",
        "nesne_02_furkan.jpg",
    ]
    _, rows = read_rows(tmp_path)
    by_source = {r["source_name"]: r["file"] for r in rows}
    # Sorted order: a.JPG comes before b.jpg.
    assert by_source["nesne/a.JPG"] == "nesne_01_furkan.jpg"
    assert by_source["nesne/b.jpg"] == "nesne_02_furkan.jpg"


@pytest.mark.parametrize(
    ("h", "w", "expected"), [(1200, 1600, (600, 800)), (1600, 1000, (800, 500))]
)
def test_large_image_long_side_800(
    tmp_path: Path, h: int, w: int, expected: tuple[int, int]
) -> None:
    put_raw(tmp_path, "manzara/big.jpg", h=h, w=w)
    assert run(tmp_path) == 0
    out = cv2.imread(str(tmp_path / "data" / "images" / "manzara_01_furkan.jpg"))
    assert out.shape[:2] == expected
    _, rows = read_rows(tmp_path)
    assert (rows[0]["orig_width"], rows[0]["orig_height"]) == (str(w), str(h))
    assert (rows[0]["width"], rows[0]["height"]) == (str(expected[1]), str(expected[0]))


def test_small_image_keeps_size_and_pixels(tmp_path: Path) -> None:
    src = put_raw(tmp_path, "yogun/small.png", h=50, w=70)
    assert run(tmp_path) == 0
    out = cv2.imread(str(tmp_path / "data" / "images" / "yogun_01_furkan.jpg"))
    assert out.shape == (50, 70, 3)
    expected = cv2.imdecode(
        cv2.imencode(".jpg", cv2.imread(str(src)), [cv2.IMWRITE_JPEG_QUALITY, 92])[1],
        cv2.IMREAD_COLOR,
    )
    np.testing.assert_array_equal(out, expected)  # only JPEG encoding, no resampling


def test_target_size_rounding() -> None:
    assert prep.target_size(4032, 3024) == (800, 600)
    assert prep.target_size(3024, 4032) == (600, 800)
    assert prep.target_size(800, 799) == (800, 799)
    assert prep.target_size(5000, 3) == (800, 1)


def test_metadata_columns_and_rows(tmp_path: Path) -> None:
    put_raw(tmp_path, "insan/p.jpg", h=900, w=1200)
    assert run(tmp_path) == 0
    header, rows = read_rows(tmp_path)
    assert header == list(prep.COLUMNS)
    assert rows == [
        {
            "file": "insan_01_furkan.jpg",
            "category": "insan",
            "person": "furkan",
            "source_name": "insan/p.jpg",
            "orig_width": "1200",
            "orig_height": "900",
            "width": "800",
            "height": "600",
        }
    ]


def test_same_basename_in_two_categories(tmp_path: Path) -> None:
    put_raw(tmp_path, "nesne/IMG_1.jpg")
    put_raw(tmp_path, "mimari/IMG_1.jpg")
    assert run(tmp_path) == 0
    _, rows = read_rows(tmp_path)
    assert sorted(r["source_name"] for r in rows) == ["mimari/IMG_1.jpg", "nesne/IMG_1.jpg"]


def test_rerun_adds_nothing(tmp_path: Path) -> None:
    put_raw(tmp_path, "nesne/a.jpg")
    put_raw(tmp_path, "manzara/b.jpg")
    assert run(tmp_path) == 0
    csv_before = (tmp_path / "data" / "metadata" / f"{PERSON}.csv").read_bytes()
    img_before = (tmp_path / "data" / "images" / "nesne_01_furkan.jpg").read_bytes()
    assert run(tmp_path) == 0
    assert (tmp_path / "data" / "metadata" / f"{PERSON}.csv").read_bytes() == csv_before
    assert (tmp_path / "data" / "images" / "nesne_01_furkan.jpg").read_bytes() == img_before
    assert len(image_names(tmp_path)) == 2


def test_new_file_later_continues_numbering(tmp_path: Path) -> None:
    put_raw(tmp_path, "nesne/a.jpg", seed=1)
    put_raw(tmp_path, "nesne/b.jpg", seed=2)
    assert run(tmp_path) == 0
    put_raw(tmp_path, "nesne/0_first_alphabetically.jpg", seed=3)
    assert run(tmp_path) == 0
    _, rows = read_rows(tmp_path)
    assert [r["file"] for r in rows] == [
        "nesne_01_furkan.jpg",
        "nesne_02_furkan.jpg",
        "nesne_03_furkan.jpg",
    ]
    assert rows[-1]["source_name"] == "nesne/0_first_alphabetically.jpg"


def test_numbering_skips_existing_image_files(tmp_path: Path) -> None:
    images = tmp_path / "data" / "images"
    images.mkdir(parents=True)
    (images / "nesne_04_furkan.jpg").write_bytes(b"placeholder")
    (images / "nesne_09_emre.jpg").write_bytes(b"other person")
    put_raw(tmp_path, "nesne/a.jpg")
    assert run(tmp_path) == 0
    assert (images / "nesne_05_furkan.jpg").exists()


def test_invalid_category_errors_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    put_raw(tmp_path, "nesne/a.jpg")
    put_raw(tmp_path, "Manzara/b.jpg")
    assert run(tmp_path) != 0
    err = capsys.readouterr().err
    assert "Manzara" in err
    assert "manzara, nesne, insan, mimari, yogun" in err
    assert image_names(tmp_path) == []
    assert not (tmp_path / "data" / "metadata" / f"{PERSON}.csv").exists()


def test_heic_errors_with_sips_hint(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    put_raw(tmp_path, "manzara/a.jpg")
    heic = tmp_path / "data" / "raw" / PERSON / "manzara" / "IMG_0001.HEIC"
    heic.write_bytes(b"\x00\x00\x00\x18ftypheic")
    assert run(tmp_path) != 0
    err = capsys.readouterr().err
    assert "manzara/IMG_0001.HEIC" in err
    assert "sips -s format jpeg girdi.heic --out cikti.jpg" in err
    assert image_names(tmp_path) == []


def test_unsupported_extension_and_unreadable_file(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    raw = tmp_path / "data" / "raw" / PERSON / "mimari"
    raw.mkdir(parents=True)
    (raw / "notes.txt").write_text("hi")
    assert run(tmp_path) != 0
    assert "notes.txt" in capsys.readouterr().err
    (raw / "notes.txt").unlink()
    (raw / "broken.jpg").write_bytes(b"not a jpeg")
    put_raw(tmp_path, "mimari/good.jpg")
    assert run(tmp_path) != 0
    assert "broken.jpg" in capsys.readouterr().err
    assert image_names(tmp_path) == []


def test_invalid_person_name(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    for bad in ("Furkan", "furkan1", "fürkan", ""):
        assert prep.main(["--person", bad, "--root", str(tmp_path)]) == 2
    assert "--person furkan" in capsys.readouterr().err


def test_missing_raw_folder(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert run(tmp_path) == 1
    assert "does not exist" in capsys.readouterr().err


def test_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    put_raw(tmp_path, "nesne/a.jpg")
    assert run(tmp_path, "--dry-run") == 0
    assert "Would add 1" in capsys.readouterr().out
    assert image_names(tmp_path) == []


def test_output_has_no_exif_and_exif_orientation_applied(tmp_path: Path) -> None:
    src = tmp_path / "data" / "raw" / PERSON / "manzara" / "phone.jpg"
    src.parent.mkdir(parents=True)
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW to display.
    exif[0x010F] = "TestCam"  # Make
    Image.fromarray(make_image(60, 80)).save(src, exif=exif.tobytes())
    assert b"Exif" in src.read_bytes()
    assert run(tmp_path) == 0
    out_path = tmp_path / "data" / "images" / "manzara_01_furkan.jpg"
    data = out_path.read_bytes()
    assert b"Exif" not in data
    assert b"TestCam" not in data
    assert cv2.imread(str(out_path)).shape[:2] == (80, 60)  # upright: portrait after rotation
