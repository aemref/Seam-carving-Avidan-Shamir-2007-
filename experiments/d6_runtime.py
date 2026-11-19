"""D6: how does runtime grow with image size, and what does vectorisation buy? Owner: Emre.

Run with ``python -m experiments.d6_runtime`` (no arguments, headless).

Outputs:
    results/figures/d6_dp_example.png  greedy vs dynamic programming on the 3 x 4 map of report
                                       section 2.2 (a drawing, needs no photo; always written)
    results/figures/d6_runtime.png     runtime plots (needs a dataset photo)
    results/tables/d6_runtime.csv      every number behind the runtime plots (needs a photo)

Measurements (all on one dataset photo, resized with cv2.resize):
    1. Time per seam split into its steps (energy, DP, seam removal) at four sizes.
    2. A pure-Python DP (two nested loops, defined only here) against the vectorised
       ``seam.cumulative_map`` at two sizes. Both must give the same ``M`` and ``back``.
    3. Total time of shrinking the width by 25 % with ``carve.shrink_width`` at four sizes,
       and a linear fit of time against the number of pixels.
    4. The paper's example (Sec. 5): retarget 400 x 500 to 100 x 100 in about 2.2 s (the
       paper does not state its hardware), measured here with ``seamcarving.resize.resize``.

Timing method: ``time.perf_counter``; one (short) untimed warm-up call before each
measurement; the reported statistic is always the median of the repeats (counts in the
constants below and in the ``statistic`` / ``n`` columns of the CSV). The CSV is in long
format: one number per row, identified by ``section``, ``quantity`` and ``size``, so every
number in the report can be traced to one cell.

The linear fits: (a) median time per seam against H x W over the four sizes; (b) total
25 % shrink time against the pixels processed, the sum of H x W over the k seams (the
width shrinks by one per seam, so the total is not linear in the original H x W).

Team modules are called through their module attribute (``carve.shrink_width(...)``) so tests
and the reference overlay can swap them.
"""

import datetime
import inspect
import os
import platform
import statistics
import subprocess
import time

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure  # no pyplot: never opens a window, works headless

from experiments import common
from seamcarving import carve, energy, resize, seam

# --- experiment settings --------------------------------------------------------------------

LONG_SIDES = (300, 600, 800, 1200)  # long side of the four resized versions, pixels
STEP_SEAMS = 10  # seams removed one by one per repeat for the per-step timing
STEP_REPEATS = 3  # -> STEP_SEAMS * STEP_REPEATS samples per step and size
DP_COMPARE_SIDES = (300, 800)  # sizes for the pure-Python vs vectorised DP comparison
NAIVE_REPEATS = 3  # the pure-Python DP takes about half a second at 800
VECTOR_REPEATS = 20
SHRINK_FRACTION = 0.25  # remove 25 % of the width
SHRINK_REPEATS = 3
# Paper Sec. 5: "400 x 500" -> 100 x 100 in about 2.2 s. The paper does not say which side is
# the height. Assumption: 400 rows (height) x 500 columns (width), the (H, W) order used in
# this project. The other orientation (500 x 400) is measured too and written to the CSV.
PAPER_SOURCE_HW = (400, 500)
PAPER_TARGET_HW = (100, 100)
PAPER_SECONDS = 2.2
PAPER_REPEATS = 3

# The 3 x 4 counterexample of report section 2.2 (must stay identical to the report).
DP_EXAMPLE_ENERGY = np.array([[1, 4, 6, 8], [4, 9, 2, 1], [9, 8, 1, 6]], dtype=np.float64)
DP_EXAMPLE_GREEDY = (0, 0, 1)  # top-down greedy seam, cost 13 (checked at run time)
DP_EXAMPLE_OPTIMAL = (1, 2, 2)  # DP seam, cost 7 (checked at run time)

# Colours (Okabe-Ito / reference palette): distinguishable for colour-blind viewers too.
BLUE, ORANGE, GREEN, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#777777"
BLUE_LIGHT, ORANGE_LIGHT = "#cde2fb", "#fbdccd"

CSV_HEADER = (
    "section",
    "quantity",
    "size",
    "height",
    "width",
    "value",
    "unit",
    "statistic",
    "n",
    "notes",
)


# --- environment ----------------------------------------------------------------------------


def _os_cpu_name() -> str:
    """Ask the operating system for the processor name; "" if it does not say."""
    system = platform.system()
    if system == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    if system == "Linux":
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() in ("model name", "Hardware", "Model") and value.strip():
                    return value.strip()
        return ""
    if system == "Windows":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    return ""


def cpu_name() -> str:
    """Return the processor name on macOS, Linux or Windows; never raises."""
    try:
        name = _os_cpu_name()
    except (OSError, ImportError, ValueError, subprocess.SubprocessError):
        name = ""
    return name or platform.processor() or platform.machine() or "unknown"


def environment_rows() -> list[tuple]:
    """CSV rows describing the machine and the library versions."""
    import matplotlib

    info = {
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count() or "unknown",
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "matplotlib": matplotlib.__version__,
        "run_date": datetime.date.today().isoformat(),
    }
    return [row("env", key, value) for key, value in info.items()]


def energy_defaults() -> str:
    """The default keyword arguments of ``energy.compute_energy``, e.g. "method=sobel ..."."""
    params = inspect.signature(energy.compute_energy).parameters.values()
    return " ".join(f"{p.name}={p.default}" for p in params if p.default is not p.empty)


def method_rows() -> list[tuple]:
    """CSV rows recording how the timings were taken."""
    settings = {
        "timer": "time.perf_counter",
        "warm_up": "one untimed call before each measurement (shrink: k=1; paper: 1 seam each way)",
        "statistic": "median over repeats",
        "step_seams": STEP_SEAMS,
        "step_repeats": STEP_REPEATS,
        "naive_repeats": NAIVE_REPEATS,
        "vector_repeats": VECTOR_REPEATS,
        "shrink_fraction": SHRINK_FRACTION,
        "shrink_repeats": SHRINK_REPEATS,
        "paper_repeats": PAPER_REPEATS,
        "energy": "energy.compute_energy with its defaults: " + energy_defaults(),
    }
    return [row("method", key, value) for key, value in settings.items()]


# --- helpers --------------------------------------------------------------------------------


def row(section, quantity, value, unit="", size="", shape=None, statistic="", n="", notes=""):
    """One CSV row; ``shape`` is (H, W) of the image the number belongs to."""
    height, width = shape if shape is not None else ("", "")
    return (section, quantity, size, height, width, value, unit, statistic, n, notes)


def r4(x: float) -> float:
    """Round to 4 significant digits for the CSV."""
    return float(f"{x:.4g}")


def pick_photo():
    """Return ``(path, rule)`` for the dataset photo used in D6, or None if there is no photo.

    Deterministic: the first landscape photo (width >= height) in file-name order; if there is
    none, the first photo in file-name order. ``rule`` says which of the two was applied.
    """
    photos = common.list_images()
    for path in photos:
        h, w = common.load_image(path).shape[:2]
        if w >= h:
            return path, "first landscape photo (W >= H) in file-name order"
    if photos:
        return photos[0], "no landscape photo; first photo in file-name order"
    return None


def resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    """Resize so that the longer side is ``long_side`` pixels, keeping the aspect ratio."""
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def crop_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Centre-crop to the aspect ratio ``out_h : out_w``, then resize to exactly that size."""
    h, w = img.shape[:2]
    if w * out_h > h * out_w:  # too wide: crop columns
        crop_w = round(h * out_w / out_h)
        c0 = (w - crop_w) // 2
        img = img[:, c0 : c0 + crop_w]
    else:  # too tall: crop rows
        crop_h = round(w * out_h / out_w)
        r0 = (h - crop_h) // 2
        img = img[r0 : r0 + crop_h]
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)


def median_time(fn, repeats: int, warm_up=None) -> float:
    """Run ``warm_up`` (default: ``fn``) once untimed, then time ``fn`` ``repeats`` times.

    Returns the median in seconds. A cheaper ``warm_up`` is used for the long runs.
    """
    (warm_up or fn)()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def linear_fit(x, y) -> tuple[float, float, float]:
    """Least-squares line ``y = slope * x + intercept``; returns (slope, intercept, R^2)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = y - y.mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total)
    return float(slope), float(intercept), r2


def figure_to_bgr(fig: Figure) -> np.ndarray:
    """Render a matplotlib figure off-screen and return it as a BGR uint8 image."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


# --- pure-Python DP (only for the comparison) -----------------------------------------------


def naive_cumulative_map(energy_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The DP of ``seam.cumulative_map`` written with two nested Python loops.

    Same recurrence and the same tie rule (leftmost parent wins), so ``M`` and ``back`` must be
    identical to the vectorised version.
    """
    e = np.asarray(energy_map, dtype=np.float64)
    n_rows, n_cols = e.shape
    M = np.empty((n_rows, n_cols), dtype=np.float64)
    back = np.zeros((n_rows, n_cols), dtype=np.int8)
    for c in range(n_cols):
        M[0, c] = e[0, c]
    for r in range(1, n_rows):
        for c in range(n_cols):
            best, offset = M[r - 1, c], 0
            if c > 0 and M[r - 1, c - 1] <= best:  # left wins a tie with the middle
                best, offset = M[r - 1, c - 1], -1
            if c < n_cols - 1 and M[r - 1, c + 1] < best:  # right must be strictly smaller
                best, offset = M[r - 1, c + 1], 1
            M[r, c] = e[r, c] + best
            back[r, c] = offset
    return M, back


# --- measurements ---------------------------------------------------------------------------


def measure_steps(img: np.ndarray) -> dict:
    """Median time per seam of energy, DP and removal, removing STEP_SEAMS seams in a row."""
    e = energy.compute_energy(img)  # warm-up
    carve.remove_vertical_seam(img, seam.find_vertical_seam(e))
    samples = {"energy": [], "dp": [], "remove": [], "total": []}
    for _ in range(STEP_REPEATS):
        current = img
        for _ in range(STEP_SEAMS):
            t0 = time.perf_counter()
            e = energy.compute_energy(current)
            t1 = time.perf_counter()
            s = seam.find_vertical_seam(e)
            t2 = time.perf_counter()
            current = carve.remove_vertical_seam(current, s)
            t3 = time.perf_counter()
            samples["energy"].append(t1 - t0)
            samples["dp"].append(t2 - t1)
            samples["remove"].append(t3 - t2)
            samples["total"].append(t3 - t0)
    return {step: statistics.median(values) for step, values in samples.items()}


def compare_dp(img: np.ndarray) -> tuple[float, float]:
    """Return median seconds of (pure-Python DP, vectorised DP) on the energy of ``img``.

    Raises RuntimeError if the two do not give the same M and back.
    """
    e = energy.compute_energy(img)
    M_naive, back_naive = naive_cumulative_map(e)
    M_vec, back_vec = seam.cumulative_map(e)
    if not (np.array_equal(M_naive, M_vec) and np.array_equal(back_naive, back_vec)):
        raise RuntimeError("pure-Python DP and seam.cumulative_map disagree; comparison invalid")
    naive = median_time(lambda: naive_cumulative_map(e), NAIVE_REPEATS)
    vector = median_time(lambda: seam.cumulative_map(e), VECTOR_REPEATS)
    return naive, vector


def measure_shrink(img: np.ndarray, k: int) -> float:
    """Median seconds of ``carve.shrink_width(img, k)``."""

    def run():
        result = carve.shrink_width(img, k)
        if result.image.shape[1] != img.shape[1] - k:
            raise RuntimeError("shrink_width returned the wrong width")

    return median_time(run, SHRINK_REPEATS, warm_up=lambda: carve.shrink_width(img, 1))


def measure_paper(photo: np.ndarray) -> list[tuple]:
    """Time the paper's 400 x 500 -> 100 x 100 example in both orientations."""
    rows = []
    target_h, target_w = PAPER_TARGET_HW
    orientations = (("assumed", PAPER_SOURCE_HW), ("swapped", PAPER_SOURCE_HW[::-1]))
    for label, (src_h, src_w) in orientations:
        img = crop_resize(photo, src_h, src_w)

        def run(img=img):
            out = resize.resize(img, target_h, target_w)
            if out.shape[:2] != (target_h, target_w):
                raise RuntimeError("resize returned the wrong size")

        def warm_up(img=img):
            resize.resize(img, img.shape[0] - 1, img.shape[1] - 1)

        seconds = median_time(run, PAPER_REPEATS, warm_up)
        size = f"{src_h}x{src_w}->{target_h}x{target_w}"
        notes = "H x W = rows x columns" + (
            " (assumption for the paper)" if label == "assumed" else ""
        )
        rows.append(
            row(
                "paper",
                f"seconds_{label}",
                r4(seconds),
                "s",
                size,
                (src_h, src_w),
                "median",
                PAPER_REPEATS,
                notes,
            )
        )
        rows.append(
            row(
                "paper",
                f"paper_over_ours_{label}",
                r4(PAPER_SECONDS / seconds),
                "ratio",
                size,
                (src_h, src_w),
                "",
                "",
                "paper 2.2 s / our time",
            )
        )
        print(f"  paper example {size} ({label}): {seconds:.2f} s (paper: {PAPER_SECONDS} s)")
    rows.append(
        row(
            "paper",
            "paper_seconds",
            PAPER_SECONDS,
            "s",
            "400x500->100x100",
            None,
            "",
            "",
            "Avidan & Shamir 2007, Sec. 5; the paper does not state its hardware",
        )
    )
    return rows


# --- figures --------------------------------------------------------------------------------


def greedy_top_down(energy_map: np.ndarray) -> np.ndarray:
    """Greedy seam: cheapest pixel of row 0, then always the cheapest of the 3 pixels below."""
    n_rows, n_cols = energy_map.shape
    path = [int(np.argmin(energy_map[0]))]
    for r in range(1, n_rows):
        c = path[-1]
        lo, hi = max(c - 1, 0), min(c + 2, n_cols)
        path.append(lo + int(np.argmin(energy_map[r, lo:hi])))
    return np.array(path, dtype=np.int64)


def _draw_grid(ax, values, paths, title, caption, highlight=None):
    """Draw a table of numbers with seam paths.

    ``paths`` is a list of (seam, edge colour, cell fill, border line style).
    """
    n_rows, n_cols = values.shape
    for r in range(n_rows):
        for c in range(n_cols):
            face = "white"
            for path, _, fill, _ in paths:
                if path[r] == c:
                    face = fill
            ax.add_patch(_rect(c, r, face, "#bbbbbb", 1.0, "solid", zorder=1))
            text = f"{values[r, c]:g}"
            ax.text(c, r, text, ha="center", va="center", fontsize=24, color=INK, zorder=4)
    for path, colour, _, style in paths:
        rows = np.arange(n_rows)
        for r in rows:
            ax.add_patch(_rect(path[r], r, "none", colour, 3.5, style, zorder=3))
        # Connect each cell to the next one between their borders, so no digit is crossed.
        for r in rows[:-1]:
            ys = (r + 0.44, r + 0.56)
            ax.plot(path[r : r + 2], ys, color=colour, linestyle="-", linewidth=3.5, zorder=3)
    if highlight is not None:
        r, c, label = highlight
        ax.text(c, r + 0.3, label, ha="center", va="center", fontsize=12, color=INK, zorder=5)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols), [f"c={c}" for c in range(n_cols)], fontsize=13)
    ax.set_yticks(range(n_rows), [f"r={r}" for r in range(n_rows)], fontsize=13)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=16, color=INK, pad=10)
    ax.set_xlabel(caption, fontsize=12, color=MUTED, labelpad=10)


def _rect(c, r, face, edge, width, style, zorder):
    from matplotlib.patches import Rectangle

    inset = 0.06 if face == "none" else 0.0
    size = 1.0 - 2 * inset
    return Rectangle(
        (c - 0.5 + inset, r - 0.5 + inset),
        size,
        size,
        facecolor=face,
        edgecolor=edge,
        linewidth=width,
        linestyle=style,
        zorder=zorder,
    )


def save_dp_example() -> None:
    """Draw greedy vs DP on the section 2.2 map.

    ``M``, the DP seam and both costs come from the real ``seam`` functions, not typed in.
    """
    e = DP_EXAMPLE_ENERGY
    M, _ = seam.cumulative_map(e)
    optimal = seam.find_vertical_seam(e)
    greedy = greedy_top_down(e)
    greedy_cost, optimal_cost = seam.seam_cost(e, greedy), seam.seam_cost(e, optimal)
    if tuple(greedy) != DP_EXAMPLE_GREEDY or tuple(optimal) != DP_EXAMPLE_OPTIMAL:
        raise RuntimeError("DP example no longer matches report section 2.2")

    def terms(path):
        return " + ".join(f"{e[r, c]:g}" for r, c in enumerate(path))

    def label(name, path, cost):
        columns = ", ".join(str(int(c)) for c in path)
        return f"{name}: columns ({columns}), cost {terms(path)} = {cost:g}"

    greedy_label = label("Greedy (top-down)", greedy, greedy_cost)
    optimal_label = label("Dynamic programming", optimal, optimal_cost)

    fig = Figure(figsize=(13, 5.6), dpi=150, facecolor="white")
    left, right = fig.subplots(1, 2, gridspec_kw={"wspace": 0.25})
    paths_e = [(greedy, ORANGE, ORANGE_LIGHT, "--"), (optimal, BLUE, BLUE_LIGHT, "-")]
    _draw_grid(left, e, paths_e, "Energy e", "3 x 4 energy map, both seams")
    end_col = int(optimal[-1])
    _draw_grid(
        right,
        M,
        [(optimal, BLUE, BLUE_LIGHT, "-")],
        "Cumulative cost M = e + min(3 parents above)",
        "cumulative_map; backtrack from the smallest M in the last row",
        highlight=(M.shape[0] - 1, end_col, "smallest"),
    )
    handles = [
        left.plot([], [], color=ORANGE, linestyle="--", linewidth=3, label=greedy_label)[0],
        left.plot([], [], color=BLUE, linestyle="-", linewidth=3, label=optimal_label)[0],
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=13, frameon=False)
    fig.suptitle("Greedy seam vs dynamic programming", fontsize=18, color=INK)
    fig.subplots_adjust(top=0.86, bottom=0.25, left=0.05, right=0.98)
    path = common.save_figure("d6_dp_example", figure_to_bgr(fig))
    print(f"wrote {path}")


def save_runtime_figure(steps, shrink, dp_compare, fit_seam, fit_shrink) -> None:
    """Three panels: time per seam by step, 25 % shrink time, pure-Python vs vectorised DP."""
    fig = Figure(figsize=(16, 5.4), dpi=150, facecolor="white")
    ax_a, ax_b, ax_c = fig.subplots(1, 3, gridspec_kw={"wspace": 0.3})

    # (a) time per seam vs pixels, per step
    px = np.array([s["pixels"] for s in steps]) / 1e6
    for key, label, colour, marker in (
        ("energy", "energy (compute_energy)", BLUE, "o"),
        ("dp", "DP (find_vertical_seam)", ORANGE, "s"),
        ("remove", "removal (remove_vertical_seam)", GREEN, "^"),
        ("total", "total per seam", INK, "D"),
    ):
        ax_a.plot(
            px,
            [s[key] * 1e3 for s in steps],
            marker=marker,
            markersize=8,
            linewidth=2,
            color=colour,
            label=label,
        )
    slope, intercept, r2 = fit_seam
    xs = np.linspace(0, px.max() * 1.05, 50)
    ax_a.plot(
        xs,
        (slope * xs * 1e6 + intercept) * 1e3,
        "--",
        color=MUTED,
        linewidth=1.5,
        label=f"linear fit of total, $R^2$ = {r2:.3f}",
    )
    # One tick per measured size, labelled with the size and its pixel count.
    ax_a.set_xticks(px, [f"{s['height']}x{s['width']}\n{x:.2f}" for s, x in zip(steps, px)])
    ax_a.tick_params(axis="x", labelsize=8.5)
    ax_a.set_title("(a) Time per seam by step", fontsize=13)
    ax_a.set_xlabel("image size H x W / pixels (millions)")
    ax_a.set_ylabel("ms per seam (median)")
    ax_a.legend(fontsize=8.5, frameon=False, loc="upper left")

    # (b) total time of the 25 % shrink vs processed pixels
    work = np.array([s["work_pixels"] for s in shrink]) / 1e6
    secs = np.array([s["seconds"] for s in shrink])
    ax_b.plot(work, secs, "o", markersize=9, color=BLUE, label="shrink_width, width -25 %")
    slope, intercept, r2 = fit_shrink
    xs = np.linspace(0, work.max() * 1.05, 50)
    ax_b.plot(
        xs,
        slope * xs * 1e6 + intercept,
        "--",
        color=MUTED,
        linewidth=1.5,
        label=f"linear fit, $R^2$ = {r2:.3f}",
    )
    for s, x, y in zip(shrink, work, secs):
        ax_b.annotate(
            f"{s['height']}x{s['width']}, k={s['k']}",
            (x, y),
            textcoords="offset points",
            xytext=(10, -4),
            ha="left",
            va="top",
            fontsize=9,
            color=MUTED,
        )
    ax_b.set_title("(b) Shrinking the width by 25 %", fontsize=13)
    ax_b.set_xlabel("pixels processed, sum of H x W over the k seams (millions)")
    ax_b.set_ylabel("total seconds (median)")
    ax_b.legend(fontsize=9, frameon=False, loc="upper left")

    # (c) pure-Python vs vectorised DP
    labels = [f"{d['height']}x{d['width']}" for d in dp_compare]
    x = np.arange(len(dp_compare))
    naive_ms = [d["naive"] * 1e3 for d in dp_compare]
    vector_ms = [d["vector"] * 1e3 for d in dp_compare]
    ax_c.bar(x - 0.2, naive_ms, 0.38, color=ORANGE, label="nested Python loops")
    ax_c.bar(x + 0.2, vector_ms, 0.38, color=BLUE, label="vectorised (cumulative_map)")
    for i, d in enumerate(dp_compare):
        ax_c.text(
            x[i],
            naive_ms[i] * 1.25,
            f"{d['naive'] / d['vector']:.0f}x faster",
            ha="center",
            fontsize=10,
            color=INK,
        )
    ax_c.set_yscale("log")
    ax_c.set_ylim(min(vector_ms) / 3, max(naive_ms) * 6)
    ax_c.set_xticks(x, labels)
    ax_c.set_title("(c) DP only: loops vs vectorised", fontsize=13)
    ax_c.set_xlabel("image size H x W (same M and back verified)")
    ax_c.set_ylabel("ms per DP (median, log scale)")
    ax_c.legend(fontsize=9, frameon=False, loc="upper left")

    for ax in (ax_a, ax_b, ax_c):
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in (ax_a, ax_b):  # start at 0, with room below for the lowest point's label
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=-0.08 * ax.get_ylim()[1])
    fig.subplots_adjust(top=0.9, bottom=0.2, left=0.05, right=0.99)
    path = common.save_figure("d6_runtime", figure_to_bgr(fig))
    print(f"wrote {path}")


# --- main -----------------------------------------------------------------------------------


def run_timings(photo_path, selection_rule: str) -> None:
    """All photo-based measurements; writes d6_runtime.csv and d6_runtime.png."""
    photo = common.load_image(photo_path)
    rows = environment_rows() + method_rows()
    rows.append(
        row(
            "input",
            "photo",
            photo_path.name,
            shape=photo.shape[:2],
            notes=selection_rule,
        )
    )

    steps, shrink, dp_compare = [], [], []
    for long_side in LONG_SIDES:
        img = resize_long_side(photo, long_side)
        h, w = img.shape[:2]
        size = f"long{long_side}"
        t = measure_steps(img)
        steps.append({"height": h, "width": w, "pixels": h * w, **t})
        n = STEP_SEAMS * STEP_REPEATS
        rows.append(row("per_seam", "pixels", h * w, "px", size, (h, w)))
        for step in ("energy", "dp", "remove", "total"):
            rows.append(
                row("per_seam", f"{step}_ms", r4(t[step] * 1e3), "ms", size, (h, w), "median", n)
            )
        for step in ("energy", "dp", "remove"):
            share = 100 * t[step] / (t["energy"] + t["dp"] + t["remove"])
            rows.append(
                row(
                    "per_seam",
                    f"{step}_share",
                    round(share, 1),
                    "%",
                    size,
                    (h, w),
                    "median ratio",
                    n,
                    "share of energy + dp + remove",
                )
            )
        print(
            f"  {h}x{w}: per seam energy {t['energy'] * 1e3:.2f} ms, "
            f"DP {t['dp'] * 1e3:.2f} ms, remove {t['remove'] * 1e3:.2f} ms"
        )

        k = round(SHRINK_FRACTION * w)
        seconds = measure_shrink(img, k)
        work = sum(h * (w - i) for i in range(k))
        shrink.append({"height": h, "width": w, "k": k, "seconds": seconds, "work_pixels": work})
        rows.append(row("shrink25", "seams", k, "seams", size, (h, w)))
        rows.append(
            row(
                "shrink25",
                "work_pixels",
                work,
                "px",
                size,
                (h, w),
                notes="sum of H x W over the k seams",
            )
        )
        rows.append(
            row("shrink25", "total_s", r4(seconds), "s", size, (h, w), "median", SHRINK_REPEATS)
        )
        rows.append(
            row(
                "shrink25",
                "ms_per_seam",
                r4(seconds / k * 1e3),
                "ms",
                size,
                (h, w),
                "median / k",
                SHRINK_REPEATS,
            )
        )
        print(f"  {h}x{w}: shrink width by {k} seams in {seconds:.2f} s")

        if long_side in DP_COMPARE_SIDES:
            naive, vector = compare_dp(img)
            dp_compare.append({"height": h, "width": w, "naive": naive, "vector": vector})
            rows.append(
                row(
                    "dp_compare",
                    "naive_ms",
                    r4(naive * 1e3),
                    "ms",
                    size,
                    (h, w),
                    "median",
                    NAIVE_REPEATS,
                    "two nested Python loops",
                )
            )
            rows.append(
                row(
                    "dp_compare",
                    "vectorised_ms",
                    r4(vector * 1e3),
                    "ms",
                    size,
                    (h, w),
                    "median",
                    VECTOR_REPEATS,
                    "seam.cumulative_map",
                )
            )
            rows.append(
                row(
                    "dp_compare",
                    "speedup",
                    r4(naive / vector),
                    "x",
                    size,
                    (h, w),
                    "ratio of medians",
                )
            )
            rows.append(
                row(
                    "dp_compare",
                    "same_M_and_back",
                    True,
                    "",
                    size,
                    (h, w),
                    notes="np.array_equal on M and back",
                )
            )
            print(
                f"  {h}x{w}: DP loops {naive * 1e3:.0f} ms vs vectorised "
                f"{vector * 1e3:.2f} ms ({naive / vector:.0f}x)"
            )

    fit_seam = linear_fit([s["pixels"] for s in steps], [s["total"] for s in steps])
    fit_shrink = linear_fit([s["work_pixels"] for s in shrink], [s["seconds"] for s in shrink])
    for name, (slope, intercept, r2), x_desc, y_desc in (
        ("per_seam_total", fit_seam, "pixels H x W", "per_seam total_ms"),
        ("shrink25_total", fit_shrink, "shrink25 work_pixels", "shrink25 total_s"),
    ):
        notes = f"least squares over the {len(LONG_SIDES)} sizes: {y_desc} against {x_desc}"
        rows.append(
            row(
                "fit",
                f"{name}_slope",
                r4(slope * 1e9),
                "ns per pixel",
                "all",
                None,
                "",
                len(LONG_SIDES),
                notes,
            )
        )
        rows.append(
            row(
                "fit",
                f"{name}_intercept",
                r4(intercept * 1e3),
                "ms",
                "all",
                None,
                "",
                len(LONG_SIDES),
                notes,
            )
        )
        rows.append(
            row("fit", f"{name}_r2", round(r2, 4), "", "all", None, "", len(LONG_SIDES), notes)
        )
        print(f"  fit {name}: {slope * 1e9:.2f} ns per pixel, R^2 = {r2:.4f}")

    rows += measure_paper(photo)

    path = common.save_table("d6_runtime", rows, CSV_HEADER)
    print(f"wrote {path}")
    save_runtime_figure(steps, shrink, dp_compare, fit_seam, fit_shrink)


def main() -> None:
    print("D6: runtime")
    save_dp_example()
    # The only allowed skip: no photo in the dataset. Any other error propagates.
    picked = pick_photo()
    if picked is None:
        print(
            f"no photo in {common.IMAGES_DIR}: runtime measurements skipped "
            "(d6_runtime.png and d6_runtime.csv not written)"
        )
        return
    photo_path, selection_rule = picked
    print(f"photo: {photo_path.name} ({selection_rule})")
    run_timings(photo_path, selection_rule)


if __name__ == "__main__":
    main()
