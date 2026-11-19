"""Paint object-removal and protection masks for seam carving with the mouse.

Usage:
    python tools/mask_painter.py data/images/<category>_<nn>_<name>.jpg [--mask-dir DIR]

Left-drag paints the REMOVE mask (green), right-drag or Shift+left-drag paints the
PROTECT mask (red). Keys: ``[``/``]`` brush size, ``u`` undo, ``c`` clear, ``s`` save,
``q``/Esc quit. With unsaved changes the first quit only prints a warning; a second
consecutive quit press leaves without saving.

Masks are stored as ``<mask-dir>/<image-stem>_remove.png`` and ``..._protect.png``:
single-channel 0/255 PNGs with the image's height and width. In memory they are
(H, W) bool arrays. Large photos are shown scaled down, but painting always happens
on the full-resolution masks.

The painting core (PaintSession, paint_stroke, render_overlay, DisplayTransform and
the file helpers) is pure and unit tested; only MaskPainterWindow touches the GUI.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

Point = tuple[float, float]
"""A (row, col) position; may lie outside the image while the mouse is dragged off it."""

GREEN_BGR = (0, 255, 0)
RED_BGR = (0, 0, 255)
WHITE_BGR = (255, 255, 255)
BLACK_BGR = (0, 0, 0)
OVERLAY_ALPHA = 0.45

MIN_BRUSH_RADIUS = 1
MAX_BRUSH_RADIUS = 150
DEFAULT_BRUSH_RADIUS = 12
BRUSH_GROWTH_FACTOR = 1.25

MAX_UNDO_STEPS = 50
DEFAULT_MAX_DISPLAY_SIDE = 1000
MASK_ON_VALUE = 255
MASK_FILE_EXTENSION = ".png"
MASKS_DIR_NAME = "masks"

FRAME_DELAY_MS = 15
KEY_CODE_MASK = 0xFF
KEY_ESCAPE = 27
KEY_BRUSH_SMALLER = ord("[")
KEY_BRUSH_LARGER = ord("]")
KEY_UNDO = ord("u")
KEY_CLEAR = ord("c")
KEY_SAVE = ord("s")
KEY_QUIT = ord("q")

HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.45
HUD_LINE_HEIGHT = 18
HUD_MARGIN = 6
HUD_KEY_HELP = "L: remove  R/Shift+L: protect  [ ]: brush  u: undo  c: clear  s: save  q: quit"
TERMINAL_KEY_HELP = (
    "Left-drag: paint REMOVE (green) | right-drag or Shift+left-drag: paint PROTECT (red)\n"
    "[ / ]: brush smaller/larger | u: undo stroke | c: clear both masks (undoable)\n"
    "s: save masks | q or Esc: quit (press twice to discard unsaved changes)"
)


class MaskPainterError(Exception):
    """Raised when an image or mask file cannot be used; the message is user-facing."""


class MaskLayer(Enum):
    """The two masks a user can paint, with their file suffix and overlay color."""

    REMOVE = ("remove", GREEN_BGR)
    PROTECT = ("protect", RED_BGR)

    def __init__(self, file_suffix: str, color_bgr: tuple[int, int, int]) -> None:
        self.file_suffix = file_suffix
        self.color_bgr = color_bgr


def paint_stroke(mask: np.ndarray, points: Sequence[Point], radius: float) -> np.ndarray:
    """Return a copy of ``mask`` with a round brush dragged through ``points``.

    Args:
        mask: (H, W) bool mask; not modified.
        points: (row, col) positions in image coordinates, in drag order. Consecutive
            points are joined, so fast drags leave no gaps. Off-image parts are clipped.
        radius: brush radius in image pixels.

    Returns:
        (H, W) bool mask with every pixel within ``radius`` of the polyline set.
    """
    painted = mask.copy()
    for start, end in _segments(points):
        _stamp_segment(painted, start, end, radius)
    return painted


def _segments(points: Sequence[Point]) -> list[tuple[Point, Point]]:
    if len(points) == 1:
        return [(points[0], points[0])]
    return list(pairwise(points))


def _stamp_segment(mask: np.ndarray, start: Point, end: Point, radius: float) -> None:
    """Set, in place, all pixels of ``mask`` within ``radius`` of segment start-end."""
    height, width = mask.shape
    top = max(int(np.floor(min(start[0], end[0]) - radius)), 0)
    bottom = min(int(np.ceil(max(start[0], end[0]) + radius)) + 1, height)
    left = max(int(np.floor(min(start[1], end[1]) - radius)), 0)
    right = min(int(np.ceil(max(start[1], end[1]) + radius)) + 1, width)
    if top >= bottom or left >= right:
        return
    rows, cols = np.ogrid[top:bottom, left:right]
    within_brush = _squared_distance_to_segment(rows, cols, start, end) <= radius**2
    mask[top:bottom, left:right] |= within_brush


def _squared_distance_to_segment(
    rows: np.ndarray, cols: np.ndarray, start: Point, end: Point
) -> np.ndarray:
    """Squared distance from each (row, col) grid pixel to the segment start-end."""
    step_r, step_c = end[0] - start[0], end[1] - start[1]
    offset_r, offset_c = rows - start[0], cols - start[1]
    length_sq = step_r**2 + step_c**2
    if length_sq == 0:
        along = 0.0
    else:
        along = np.clip((offset_r * step_r + offset_c * step_c) / length_sq, 0.0, 1.0)
    return (offset_r - along * step_r) ** 2 + (offset_c - along * step_c) ** 2


@dataclass
class _ActiveStroke:
    layer: MaskLayer
    radius: float
    last_point: Point


@dataclass(frozen=True)
class _PackedMasks:
    """Bit-packed copy of both masks, so a long undo history stays small in memory."""

    shape: tuple[int, int]
    bits: dict[MaskLayer, np.ndarray]

    @classmethod
    def pack(cls, masks: dict[MaskLayer, np.ndarray]) -> _PackedMasks:
        some_mask = next(iter(masks.values()))
        return cls(some_mask.shape, {layer: np.packbits(m) for layer, m in masks.items()})

    def unpack(self) -> dict[MaskLayer, np.ndarray]:
        pixel_count = self.shape[0] * self.shape[1]
        return {
            layer: np.unpackbits(bits, count=pixel_count).reshape(self.shape).astype(bool)
            for layer, bits in self.bits.items()
        }


class PaintSession:
    """Editable REMOVE/PROTECT mask pair with stroke painting, undo and clear.

    Pure NumPy state: it knows nothing about windows or files. All points and radii
    are in full-resolution image coordinates. Painting one layer never touches the
    other; the seam carving algorithm resolves overlap ("protection wins").
    """

    def __init__(self, remove_mask: np.ndarray, protect_mask: np.ndarray) -> None:
        """Start a session from two (H, W) masks (copied; any nonzero counts as marked)."""
        if remove_mask.shape != protect_mask.shape:
            raise ValueError(
                f"mask shapes differ: remove {remove_mask.shape}, protect {protect_mask.shape}"
            )
        self._masks = {
            MaskLayer.REMOVE: remove_mask.astype(bool, copy=True),
            MaskLayer.PROTECT: protect_mask.astype(bool, copy=True),
        }
        self._undo_stack: deque[_PackedMasks] = deque(maxlen=MAX_UNDO_STEPS)
        self._stroke: _ActiveStroke | None = None
        self._saved_masks = self._copy_masks()

    @classmethod
    def blank(cls, shape: tuple[int, int]) -> PaintSession:
        """Start a session with two empty (H, W) masks."""
        return cls(np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))

    def mask(self, layer: MaskLayer) -> np.ndarray:
        """Return a read-only (H, W) bool view of one layer's current mask."""
        view = self._masks[layer].view()
        view.flags.writeable = False
        return view

    def begin_stroke(self, layer: MaskLayer, point: Point, radius: float) -> None:
        """Start a new undoable stroke on ``layer`` and paint a dot at ``point``."""
        self._push_undo_state()
        self._stroke = _ActiveStroke(layer, radius, point)
        _stamp_segment(self._masks[layer], point, point, radius)

    def extend_stroke(self, point: Point) -> None:
        """Paint from the stroke's last point to ``point``; ignored outside a stroke."""
        if self._stroke is None:
            return
        stroke = self._stroke
        _stamp_segment(self._masks[stroke.layer], stroke.last_point, point, stroke.radius)
        stroke.last_point = point

    def end_stroke(self) -> None:
        self._stroke = None

    def undo(self) -> None:
        """Restore both masks to before the last stroke or clear; no-op if none."""
        self.end_stroke()
        if self._undo_stack:
            self._masks = self._undo_stack.pop().unpack()

    def clear(self) -> None:
        """Empty both masks as one undoable step; no-op if both are already empty."""
        self.end_stroke()
        if not any(mask.any() for mask in self._masks.values()):
            return
        self._push_undo_state()
        self._masks = {layer: np.zeros_like(mask) for layer, mask in self._masks.items()}

    def mark_saved(self) -> None:
        """Record the current masks as the state that is on disk."""
        self._saved_masks = self._copy_masks()

    def has_unsaved_changes(self) -> bool:
        return any(
            not np.array_equal(self._masks[layer], self._saved_masks[layer])
            for layer in MaskLayer
        )

    def _push_undo_state(self) -> None:
        self._undo_stack.append(_PackedMasks.pack(self._masks))

    def _copy_masks(self) -> dict[MaskLayer, np.ndarray]:
        return {layer: mask.copy() for layer, mask in self._masks.items()}


def larger_brush_radius(radius: int) -> int:
    """Next brush radius up (about +25 %, at least +1), clamped to MAX_BRUSH_RADIUS."""
    return min(max(radius + 1, round(radius * BRUSH_GROWTH_FACTOR)), MAX_BRUSH_RADIUS)


def smaller_brush_radius(radius: int) -> int:
    """Next brush radius down (about -20 %, at least -1), clamped to MIN_BRUSH_RADIUS."""
    return max(min(radius - 1, round(radius / BRUSH_GROWTH_FACTOR)), MIN_BRUSH_RADIUS)


def render_overlay(
    image: np.ndarray, remove_mask: np.ndarray, protect_mask: np.ndarray
) -> np.ndarray:
    """Blend the masks onto the image as semi-transparent color.

    Args:
        image: (H, W, 3) BGR uint8; not modified.
        remove_mask: (H, W) bool, tinted green.
        protect_mask: (H, W) bool, tinted red and drawn on top where both are set.

    Returns:
        (H, W, 3) BGR uint8 preview; unmarked pixels equal the input image.
    """
    preview = image.copy()
    for layer, mask in ((MaskLayer.REMOVE, remove_mask), (MaskLayer.PROTECT, protect_mask)):
        preview[mask] = _tint(image[mask], layer.color_bgr)
    return preview


def _tint(pixels: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    """Blend (N, 3) uint8 pixels toward ``color_bgr`` by OVERLAY_ALPHA."""
    blended = (1.0 - OVERLAY_ALPHA) * pixels + OVERLAY_ALPHA * np.array(color_bgr)
    return np.round(blended).astype(np.uint8)


@dataclass(frozen=True)
class DisplayTransform:
    """Maps between full-resolution image pixels and the (smaller) displayed pixels.

    ``scale`` is display size / image size and never exceeds 1. Pixel centers map to
    pixel centers, so display -> image -> display is exact.
    """

    scale: float

    @classmethod
    def fit(cls, image_shape: tuple[int, ...], max_display_side: int) -> DisplayTransform:
        """Scale so the image's longer side is at most ``max_display_side`` pixels."""
        long_side = max(image_shape[:2])
        return cls(min(1.0, max_display_side / long_side))

    def display_shape(self, image_shape: tuple[int, ...]) -> tuple[int, int]:
        """(height, width) of the displayed image; floored so it maps inside the image."""
        return (
            max(1, int(image_shape[0] * self.scale)),
            max(1, int(image_shape[1] * self.scale)),
        )

    def to_image_point(self, display_point: Point) -> tuple[int, int]:
        """Image (row, col) under a display pixel; not clipped to the image."""
        row, col = display_point
        return int(self._display_to_image_index(row)), int(self._display_to_image_index(col))

    def to_display_point(self, image_point: Point) -> tuple[int, int]:
        """Display (row, col) showing an image pixel."""
        row, col = image_point
        return int(np.floor((row + 0.5) * self.scale)), int(np.floor((col + 0.5) * self.scale))

    def to_image_radius(self, display_radius: float) -> float:
        return display_radius / self.scale

    def shrink_mask(self, mask: np.ndarray) -> np.ndarray:
        """Nearest-neighbor resample of an (H, W) mask to the display shape."""
        display_height, display_width = self.display_shape(mask.shape)
        rows = self._display_to_image_index(np.arange(display_height))
        cols = self._display_to_image_index(np.arange(display_width))
        return mask[np.ix_(rows, cols)]

    def shrink_image(self, image: np.ndarray) -> np.ndarray:
        """Area-averaged (H, W, 3) image resized to the display shape."""
        display_height, display_width = self.display_shape(image.shape)
        return cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)

    def _display_to_image_index(self, display_index: float | np.ndarray) -> np.ndarray:
        """Image row/col index whose pixel contains the display pixel's center."""
        return np.floor((display_index + 0.5) / self.scale).astype(int)


@dataclass(frozen=True)
class MaskPaths:
    """Where an image's REMOVE and PROTECT masks live on disk."""

    image_stem: str
    remove: Path
    protect: Path

    @classmethod
    def for_image(cls, image_path: Path, mask_dir: Path) -> MaskPaths:
        """``<mask_dir>/<image-stem>_remove.png`` and ``..._protect.png``."""
        return cls(
            image_stem=image_path.stem,
            remove=_layer_path(image_path, mask_dir, MaskLayer.REMOVE),
            protect=_layer_path(image_path, mask_dir, MaskLayer.PROTECT),
        )

    def path_for(self, layer: MaskLayer) -> Path:
        return {MaskLayer.REMOVE: self.remove, MaskLayer.PROTECT: self.protect}[layer]


def _layer_path(image_path: Path, mask_dir: Path, layer: MaskLayer) -> Path:
    return mask_dir / f"{image_path.stem}_{layer.file_suffix}{MASK_FILE_EXTENSION}"


def default_mask_dir(image_path: Path) -> Path:
    """The ``masks`` folder next to the image's folder (data/images -> data/masks)."""
    return image_path.parent.parent / MASKS_DIR_NAME


class SaveOutcome(Enum):
    """What ``save_mask`` did, phrased for the terminal."""

    WRITTEN = "saved"
    SKIPPED_EMPTY = "empty, not written"
    DELETED_STALE = "empty, deleted stale file"


def load_image(image_path: Path) -> np.ndarray:
    """Read an image as (H, W, 3) BGR uint8, exactly like the seam carving pipeline."""
    if not image_path.is_file():
        raise MaskPainterError(f"image not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise MaskPainterError(f"cannot read image (unsupported or corrupt?): {image_path}")
    return image


def load_mask(mask_path: Path, image_shape: tuple[int, ...]) -> np.ndarray:
    """Read a mask PNG as (H, W) bool (any nonzero = marked); empty if the file is absent.

    Raises:
        MaskPainterError: if the file is unreadable or its size differs from the image.
    """
    expected_shape = tuple(image_shape[:2])
    if not mask_path.exists():
        return np.zeros(expected_shape, dtype=bool)
    stored = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if stored is None:
        raise MaskPainterError(f"cannot read mask: {mask_path}")
    if stored.shape != expected_shape:
        raise MaskPainterError(
            f"mask {mask_path} is {stored.shape[1]}x{stored.shape[0]} (WxH) but the image "
            f"is {expected_shape[1]}x{expected_shape[0]}; fix or delete the mask file"
        )
    return stored != 0


def save_mask(mask: np.ndarray, mask_path: Path) -> SaveOutcome:
    """Write an (H, W) bool mask as a 0/255 PNG; empty masks are never written.

    An empty mask whose file already exists deletes that now stale file.
    """
    if not mask.any():
        return _discard_empty_mask(mask_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * MASK_ON_VALUE):
        raise MaskPainterError(f"cannot write mask: {mask_path}")
    return SaveOutcome.WRITTEN


def _discard_empty_mask(mask_path: Path) -> SaveOutcome:
    if mask_path.exists():
        mask_path.unlink()
        return SaveOutcome.DELETED_STALE
    return SaveOutcome.SKIPPED_EMPTY


def load_session(paths: MaskPaths, image_shape: tuple[int, ...]) -> PaintSession:
    """Start a session from the masks on disk (or empty masks where none exist)."""
    return PaintSession(load_mask(paths.remove, image_shape), load_mask(paths.protect, image_shape))


def save_session(session: PaintSession, paths: MaskPaths) -> dict[Path, SaveOutcome]:
    """Save both masks, mark the session saved and report what happened per file."""
    outcomes = {}
    for layer in MaskLayer:
        mask_path = paths.path_for(layer)
        outcomes[mask_path] = save_mask(session.mask(layer), mask_path)
    session.mark_saved()
    return outcomes


class MaskPainterWindow:
    """Thin OpenCV window and event loop around a PaintSession (not unit tested)."""

    def __init__(
        self,
        image: np.ndarray,
        session: PaintSession,
        paths: MaskPaths,
        transform: DisplayTransform,
    ) -> None:
        self._session = session
        self._paths = paths
        self._transform = transform
        self._display_image = transform.shrink_image(image)
        self._window_name = f"mask painter - {paths.image_stem}"
        self._brush_radius = DEFAULT_BRUSH_RADIUS
        self._cursor: Point | None = None
        self._quit_armed = False
        self._running = True
        self._needs_redraw = True
        self._mouse_handlers: dict[int, Callable[[Point, int], None]] = {
            cv2.EVENT_LBUTTONDOWN: self._on_left_down,
            cv2.EVENT_RBUTTONDOWN: self._on_right_down,
            cv2.EVENT_MOUSEMOVE: self._on_mouse_move,
            cv2.EVENT_LBUTTONUP: self._on_button_up,
            cv2.EVENT_RBUTTONUP: self._on_button_up,
        }
        self._key_actions: dict[int, Callable[[], None]] = {
            KEY_BRUSH_SMALLER: self._shrink_brush,
            KEY_BRUSH_LARGER: self._grow_brush,
            KEY_UNDO: self._session.undo,
            KEY_CLEAR: self._session.clear,
            KEY_SAVE: self._save,
            KEY_QUIT: self._request_quit,
            KEY_ESCAPE: self._request_quit,
        }

    def run(self) -> None:
        """Show the window and process mouse and key events until the user quits."""
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self._window_name, self._on_mouse)
        print(TERMINAL_KEY_HELP)
        try:
            self._event_loop()
        finally:
            cv2.destroyAllWindows()

    def _event_loop(self) -> None:
        while self._running:
            if self._needs_redraw:
                cv2.imshow(self._window_name, self._render_frame())
                self._needs_redraw = False
            self._handle_key(cv2.waitKey(FRAME_DELAY_MS) & KEY_CODE_MASK)
            if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                self._on_window_closed()

    def _on_mouse(self, event: int, col: int, row: int, flags: int, _param: object) -> None:
        # OpenCV passes (x, y) = (col, row); from here on we only use (row, col).
        handler = self._mouse_handlers.get(event)
        self._cursor = (row, col)
        self._needs_redraw = True
        if handler is not None:
            handler((row, col), flags)

    def _on_left_down(self, display_point: Point, flags: int) -> None:
        shift_held = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        self._begin_stroke(MaskLayer.PROTECT if shift_held else MaskLayer.REMOVE, display_point)

    def _on_right_down(self, display_point: Point, _flags: int) -> None:
        self._begin_stroke(MaskLayer.PROTECT, display_point)

    def _on_mouse_move(self, display_point: Point, _flags: int) -> None:
        self._session.extend_stroke(self._transform.to_image_point(display_point))

    def _on_button_up(self, _display_point: Point, _flags: int) -> None:
        self._session.end_stroke()

    def _begin_stroke(self, layer: MaskLayer, display_point: Point) -> None:
        self._quit_armed = False
        image_radius = self._transform.to_image_radius(self._brush_radius)
        self._session.begin_stroke(
            layer, self._transform.to_image_point(display_point), image_radius
        )

    def _handle_key(self, key: int) -> None:
        action = self._key_actions.get(key)
        if action is None:
            return
        if action != self._request_quit:
            self._quit_armed = False
        action()
        self._needs_redraw = True

    def _shrink_brush(self) -> None:
        self._brush_radius = smaller_brush_radius(self._brush_radius)

    def _grow_brush(self) -> None:
        self._brush_radius = larger_brush_radius(self._brush_radius)

    def _save(self) -> None:
        for path, outcome in save_session(self._session, self._paths).items():
            print(f"{outcome.value}: {path}")

    def _request_quit(self) -> None:
        if self._quit_armed or not self._session.has_unsaved_changes():
            self._running = False
            return
        self._quit_armed = True
        print("Unsaved changes! Press s to save, or q/Esc again to quit without saving.")

    def _on_window_closed(self) -> None:
        self._running = False
        if self._session.has_unsaved_changes():
            print("Window closed with unsaved changes; masks were NOT saved.")

    def _render_frame(self) -> np.ndarray:
        """Overlay preview with brush cursor, plus a status strip below (not over) it."""
        preview = render_overlay(
            self._display_image,
            self._transform.shrink_mask(self._session.mask(MaskLayer.REMOVE)),
            self._transform.shrink_mask(self._session.mask(MaskLayer.PROTECT)),
        )
        self._draw_brush_cursor(preview)
        return np.vstack([preview, self._render_status_strip(preview.shape[1])])

    def _draw_brush_cursor(self, preview: np.ndarray) -> None:
        if self._cursor is None:
            return
        row, col = self._cursor
        center = (int(col), int(row))
        cv2.circle(preview, center, self._brush_radius, BLACK_BGR, 3, cv2.LINE_AA)
        cv2.circle(preview, center, self._brush_radius, WHITE_BGR, 1, cv2.LINE_AA)

    def _render_status_strip(self, width: int) -> np.ndarray:
        unsaved_note = "  (unsaved changes)" if self._session.has_unsaved_changes() else ""
        lines = [f"brush radius {self._brush_radius} px{unsaved_note}", HUD_KEY_HELP]
        strip = np.zeros((HUD_MARGIN + HUD_LINE_HEIGHT * len(lines), width, 3), dtype=np.uint8)
        for line_number, text in enumerate(lines, start=1):
            baseline = (HUD_MARGIN, HUD_LINE_HEIGHT * line_number)
            cv2.putText(strip, text, baseline, HUD_FONT, HUD_FONT_SCALE, WHITE_BGR, 1, cv2.LINE_AA)
        return strip


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paint seam carving masks: REMOVE (left-drag, green) and\n"
        "PROTECT (right-drag or Shift+left-drag, red).",
        epilog=TERMINAL_KEY_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="image to paint on, e.g. data/images/x.jpg")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="folder for <stem>_remove.png / <stem>_protect.png "
        "(default: the 'masks' folder next to the image's folder)",
    )
    parser.add_argument(
        "--max-display-side",
        type=int,
        default=DEFAULT_MAX_DISPLAY_SIDE,
        help="longest displayed side in pixels; masks are still painted at full "
        "resolution (default: %(default)s)",
    )
    return parser.parse_args(argv)


def build_window(args: argparse.Namespace) -> MaskPainterWindow:
    """Load the image and existing masks; raises MaskPainterError on bad input."""
    image = load_image(args.image)
    mask_dir = args.mask_dir if args.mask_dir is not None else default_mask_dir(args.image)
    paths = MaskPaths.for_image(args.image, mask_dir)
    session = load_session(paths, image.shape)
    transform = DisplayTransform.fit(image.shape, args.max_display_side)
    print(f"image {image.shape[1]}x{image.shape[0]}, display scale {transform.scale:.3f}")
    print(f"masks: {paths.remove}\n       {paths.protect}")
    return MaskPainterWindow(image, session, paths, transform)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        window = build_window(args)
    except MaskPainterError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    window.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
