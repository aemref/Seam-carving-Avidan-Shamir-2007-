"""Prepare one person's raw photos for the shared dataset.

Usage::

    python tools/prepare_dataset.py --person <name> [--dry-run] [--root DIR]

Reads ``data/raw/<name>/<category>/*`` and, for every source file that is not yet
registered, writes ``data/images/<category>_<nn>_<name>.jpg`` and appends a row to
``data/metadata/<name>.csv``.

Processing of one file:

1. Decode with OpenCV (``cv2.IMREAD_COLOR``). OpenCV applies the EXIF orientation
   tag while decoding (we do not pass ``IMREAD_IGNORE_ORIENTATION``), so phone photos
   come out upright. The file is read through ``np.fromfile`` + ``cv2.imdecode``
   instead of ``cv2.imread`` because ``cv2.imread`` cannot open non-ASCII paths on
   Windows (e.g. ``Masaüstü``); decoding is otherwise identical.
2. If the long side is larger than 800 px, shrink it to exactly 800 px with
   ``cv2.INTER_AREA``. Smaller images keep their pixels untouched.
3. Re-encode as JPEG quality 92 with OpenCV. The new file contains no EXIF block,
   so camera data and GPS location are dropped.

Metadata columns: ``file, category, person, source_name, orig_width, orig_height,
width, height``. ``source_name`` is the source path relative to ``data/raw/<name>/``
with forward slashes, e.g. ``nesne/IMG_1234.jpg``; files with the same basename in
different category folders therefore do not collide. A source whose ``source_name``
is already in the csv is skipped, so re-running the tool is safe. ``orig_width`` and
``orig_height`` are the decoded size, i.e. after the EXIF rotation.

Numbers continue per person and category from the highest number found in the csv
or in ``data/images/``. Files are processed in sorted order. Every source file is
checked and decoded before anything is written, so a bad folder name, a HEIC file or
an unreadable file stops the run without partial output.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

CATEGORIES: tuple[str, ...] = ("manzara", "nesne", "insan", "mimari", "yogun")
KNOWN_PEOPLE: tuple[str, ...] = ("emre", "ahsen", "mirza", "furkan", "berkay")
COLUMNS: tuple[str, ...] = (
    "file",
    "category",
    "person",
    "source_name",
    "orig_width",
    "orig_height",
    "width",
    "height",
)
MAX_SIDE = 800
JPEG_QUALITY = 92
MAX_NUMBER = 99
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
HEIC_EXTENSIONS = frozenset({".heic", ".heif"})
IGNORED_FILE_NAMES = frozenset({"thumbs.db", "desktop.ini"})
HEIC_HINT = (
    "HEIC is not supported; convert it to JPEG first, e.g. on macOS: "
    "sips -s format jpeg girdi.heic --out cikti.jpg "
    "(or set iPhone Settings > Camera > Formats > Most Compatible). "
    "Then delete the .heic file from the raw folder."
)
PERSON_RE = re.compile(r"^[a-z]+$")


class DatasetError(Exception):
    """One or more problems that stop the run; ``problems`` holds one message each."""

    def __init__(self, problems: list[str] | str, wrote_nothing: bool = True) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        self.wrote_nothing = wrote_nothing
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class SourceFile:
    """A raw photo: absolute ``path``, its ``category`` and ``source_name``."""

    path: Path
    category: str
    source_name: str


@dataclass
class Summary:
    """What a run did (or, with ``dry_run``, would do)."""

    person: str
    added: list[dict[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing_outputs: list[str] = field(default_factory=list)
    metadata_path: Path | None = None
    dry_run: bool = False


def default_root() -> Path:
    """Repository root: the parent of the ``tools/`` folder holding this file."""
    return Path(__file__).resolve().parents[1]


def person_error(person: str) -> str | None:
    """Return an error message if ``person`` is not lowercase ASCII letters, else None."""
    if PERSON_RE.fullmatch(person):
        return None
    hint = ""
    lowered = person.strip().lower()
    if PERSON_RE.fullmatch(lowered):
        hint = f" Did you mean --person {lowered}?"
    return (
        f"invalid person name {person!r}: use lowercase ASCII letters only "
        f"(a-z, no spaces, digits or Turkish letters), e.g. one of {', '.join(KNOWN_PEOPLE)}.{hint}"
    )


def is_ignored(path: Path) -> bool:
    """True for hidden files and OS clutter (``.DS_Store``, ``._x``, ``Thumbs.db``)."""
    return path.name.startswith(".") or path.name.lower() in IGNORED_FILE_NAMES


def scan_raw(raw_dir: Path) -> list[SourceFile]:
    """List the photos under ``raw_dir`` (= ``data/raw/<name>``) in sorted order.

    Raises DatasetError listing every problem: unknown category folders, files
    outside a category folder, subfolders, HEIC files and unsupported extensions.
    """
    if not raw_dir.is_dir():
        raise DatasetError(
            f"folder {raw_dir} does not exist. Create data/raw/<name>/<category>/ folders "
            f"(valid categories: {', '.join(CATEGORIES)}) and put your photos there."
        )
    problems: list[str] = []
    sources: list[SourceFile] = []
    for entry in sorted(raw_dir.iterdir()):
        if is_ignored(entry):
            continue
        if not entry.is_dir():
            problems.append(
                f"{entry.name}: file is not inside a category folder; move it to one of "
                f"{', '.join(f'{c}/' for c in CATEGORIES)}"
            )
            continue
        if entry.name not in CATEGORIES:
            problems.append(
                f"{entry.name}/: unknown category folder; valid names are exactly: "
                f"{', '.join(CATEGORIES)}"
            )
            continue
        for path in sorted(entry.iterdir()):
            if is_ignored(path):
                continue
            rel = PurePosixPath(entry.name, path.name).as_posix()
            if path.is_dir():
                problems.append(
                    f"{rel}/: subfolders are not supported; put photos directly in {entry.name}/"
                )
                continue
            ext = path.suffix.lower()
            if ext in HEIC_EXTENSIONS:
                problems.append(f"{rel}: {HEIC_HINT}")
            elif ext not in IMAGE_EXTENSIONS:
                problems.append(
                    f"{rel}: unsupported file type {path.suffix or '(none)'!r}; "
                    f"allowed: .jpg, .jpeg, .png (any case)"
                )
            else:
                sources.append(SourceFile(path=path, category=entry.name, source_name=rel))
    if problems:
        raise DatasetError(problems)
    return sources


def read_metadata(csv_path: Path) -> list[dict[str, str]]:
    """Read ``data/metadata/<name>.csv``; [] if missing. Raises DatasetError on a bad header."""
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = tuple(reader.fieldnames or ())
        if header != COLUMNS:
            raise DatasetError(
                f"{csv_path}: unexpected header {list(header)}; expected exactly {list(COLUMNS)}. "
                "Fix or remove the file (removing it re-registers every raw photo)."
            )
        return [dict(row) for row in reader]


def write_metadata(csv_path: Path, rows: list[dict[str, str]]) -> None:
    """Write ``rows`` with the ``COLUMNS`` header; atomic replace, ``\\n`` line endings."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)


def output_name(category: str, number: int, person: str) -> str:
    """``<category>_<nn>_<person>.jpg`` with a 2-digit zero-padded number."""
    return f"{category}_{number:02d}_{person}.jpg"


def highest_numbers(person: str, rows: list[dict[str, str]], images_dir: Path) -> dict[str, int]:
    """Highest used number per category for ``person``, from csv rows and existing files."""
    pattern = re.compile(rf"^({'|'.join(CATEGORIES)})_(\d+)_{re.escape(person)}\.jpg$")
    names = [row.get("file", "") for row in rows]
    if images_dir.is_dir():
        names += [p.name for p in images_dir.iterdir()]
    highest = dict.fromkeys(CATEGORIES, 0)
    for name in names:
        m = pattern.match(name)
        if m:
            highest[m.group(1)] = max(highest[m.group(1)], int(m.group(2)))
    return highest


def target_size(width: int, height: int, max_side: int = MAX_SIDE) -> tuple[int, int]:
    """(width, height) with the long side scaled down to ``max_side``; unchanged if already <=."""
    long_side = max(width, height)
    if long_side <= max_side:
        return width, height
    if width >= height:
        return max_side, max(1, round(height * max_side / width))
    return max(1, round(width * max_side / height)), max_side


def resize_long_side(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Shrink (H, W[, C]) so the long side is ``max_side`` (INTER_AREA); small: ``img``."""
    h, w = img.shape[:2]
    new_w, new_h = target_size(w, h, max_side)
    if (new_w, new_h) == (w, h):
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def read_image(path: Path) -> np.ndarray:
    """Decode ``path`` to BGR uint8 (H, W, 3), EXIF orientation applied. Raises DatasetError."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except (OSError, cv2.error) as exc:
        raise DatasetError(f"{path}: cannot read file ({exc})") from exc
    if img is None:
        raise DatasetError(
            f"{path}: cannot decode image; the file is empty, damaged or not a real JPEG/PNG"
        )
    return img


def write_jpeg(path: Path, img: np.ndarray, quality: int = JPEG_QUALITY) -> None:
    """Encode ``img`` as JPEG (no EXIF) and write it to ``path``. Raises DatasetError."""
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise DatasetError(f"{path}: OpenCV failed to encode the image")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def prepare_person(root: Path, person: str, dry_run: bool = False) -> Summary:
    """Process ``root/data/raw/<person>``; write images and metadata unless ``dry_run``.

    Raises DatasetError before writing anything if any new source file is invalid.
    """
    err = person_error(person)
    if err:
        raise DatasetError(err)
    raw_dir = root / "data" / "raw" / person
    images_dir = root / "data" / "images"
    csv_path = root / "data" / "metadata" / f"{person}.csv"

    sources = scan_raw(raw_dir)
    rows = read_metadata(csv_path)
    registered = {row["source_name"] for row in rows}
    summary = Summary(person=person, metadata_path=csv_path, dry_run=dry_run)
    summary.missing_outputs = [r["file"] for r in rows if not (images_dir / r["file"]).exists()]

    # Phase 1: decode and resize every new file; nothing is written yet.
    highest = highest_numbers(person, rows, images_dir)
    pending: list[tuple[dict[str, str], np.ndarray]] = []
    problems: list[str] = []
    for src in sources:
        if src.source_name in registered:
            summary.skipped.append(src.source_name)
            continue
        try:
            img = read_image(src.path)
        except DatasetError as exc:
            problems.extend(exc.problems)
            continue
        number = highest[src.category] + 1
        if number > MAX_NUMBER:
            problems.append(
                f"{src.source_name}: more than {MAX_NUMBER} images in category {src.category}"
            )
            continue
        highest[src.category] = number
        name = output_name(src.category, number, person)
        if (images_dir / name).exists():
            problems.append(
                f"{src.source_name}: output {name} already exists; refusing to overwrite it"
            )
            continue
        out = resize_long_side(img)
        row = {
            "file": name,
            "category": src.category,
            "person": person,
            "source_name": src.source_name,
            "orig_width": str(img.shape[1]),
            "orig_height": str(img.shape[0]),
            "width": str(out.shape[1]),
            "height": str(out.shape[0]),
        }
        pending.append((row, out))
    if problems:
        raise DatasetError(problems)

    summary.added = [row for row, _ in pending]
    if dry_run or not pending:
        return summary

    # Phase 2: write images; the csv is saved even if a later write fails, so it
    # always lists exactly the images that exist.
    written: list[dict[str, str]] = []
    try:
        for row, out in pending:
            try:
                write_jpeg(images_dir / row["file"], out)
            except OSError as exc:
                raise DatasetError(
                    f"data/images/{row['file']}: cannot write ({exc}); "
                    f"{len(written)} image(s) before it were written and registered",
                    wrote_nothing=False,
                ) from exc
            written.append(row)
    finally:
        if written:
            write_metadata(csv_path, rows + written)
    return summary


def format_summary(summary: Summary, root: Path) -> str:
    """Short human-readable report of a run."""

    def rel(p: Path) -> str:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            return str(p)

    verb = "Would add" if summary.dry_run else "Added"
    if summary.added:
        lines = [f"{verb} {len(summary.added)} image(s) for {summary.person}."]
    else:
        lines = [f"Nothing new to add for {summary.person}."]
    for row in summary.added:
        size = f"{row['orig_width']}x{row['orig_height']} -> {row['width']}x{row['height']}"
        lines.append(f"  {row['source_name']} -> data/images/{row['file']} ({size})")
    if summary.skipped:
        lines.append(
            f"Skipped {len(summary.skipped)} already registered: {', '.join(summary.skipped)}"
        )
    for name in summary.missing_outputs:
        lines.append(
            f"warning: data/images/{name} is listed in the metadata but missing; "
            "delete its csv row and re-run to regenerate it"
        )
    if summary.added and not summary.dry_run and summary.metadata_path is not None:
        lines.append(f"Metadata: {rel(summary.metadata_path)}")
    needs_mask = [r["file"] for r in summary.added if r["category"] in ("nesne", "insan")]
    if needs_mask:
        lines.append(
            "Next: paint masks with tools/mask_painter.py (nesne needs _remove.png, "
            f"insan needs _protect.png) for: {', '.join(needs_mask)}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    parser = argparse.ArgumentParser(
        description="Resize raw photos to a long side of 800 px, strip EXIF and update metadata."
    )
    parser.add_argument("--person", required=True, help="your name, lowercase ASCII, e.g. furkan")
    parser.add_argument(
        "--root", type=Path, default=None, help="repository root (default: parent of tools/)"
    )
    parser.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns 0 on success, 1 on dataset problems, 2 on bad arguments."""
    args = build_parser().parse_args(argv)
    root = (args.root or default_root()).resolve()
    err = person_error(args.person)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.person not in KNOWN_PEOPLE:
        print(
            f"warning: {args.person!r} is not one of {', '.join(KNOWN_PEOPLE)}; check the spelling",
            file=sys.stderr,
        )
    try:
        summary = prepare_person(root, args.person, dry_run=args.dry_run)
    except DatasetError as exc:
        print(f"error: {len(exc.problems)} problem(s) for {args.person}:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        if exc.wrote_nothing:
            print("Nothing was written. Fix the problems above and run again.", file=sys.stderr)
        return 1
    print(format_summary(summary, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
