## 4. Implementation and Runtime

### 4.1 Architecture

**Modules.** The package `seamcarving` has one module per step of the method and one owner per module. The dependency chain follows the algorithm: `filters` → `energy` (map $e$) → `seam` (map $M$, minimum seam) → `carve` (seam removal) → `enlarge` (seam insertion) and `object_removal` → `resize` → `cli`. `baselines` and `viz` sit beside the chain; each experiment is a script in `experiments/`. All signatures were fixed before any module was written, so the five members worked in parallel.

**Energy as a parameter.** The shrinking loop in `carve` repeats `energy_fn(img, carry)` → `seam_fn(energy)` → remove the seam. Both functions are parameters (defaults: `compute_energy`, `find_vertical_seam`), so `carve` knows nothing about operators or masks. Experiment D1 swaps the operator; object removal wraps the base energy so that masked pixels receive large negative or positive weights; `resize` does the same with a protection mask when shrinking. `carry` is a tuple of side arrays, such as these masks, cut with the same seam as the image; `energy_fn` receives them in their current state, so a mask stays aligned with the image.

**Horizontal seams.** `shrink_height` swaps the first two axes of the image and of every carried array, calls `shrink_width`, and swaps the results back. A horizontal seam is thus a vertical seam of the transpose, and `seam` implements vertical seams only.

**Vectorisation.** `cumulative_map` loops in Python over rows only. For row $r$, row $r-1$ of $M$ is copied into a buffer of length $W + 2$ with $+\infty$ at both ends. Three shifted views of it (left, middle and right parent) are stacked into a $3 \times W$ array, and one `argmin` along the first axis gives every column's best parent at once; since `argmin` returns the first minimum, ties go left. $M(r, \cdot)$ is the energy plus the chosen values, and the offset `best - 1` is stored as `int8` in `back`. Backtracking loops over the $H$ rows. As a development check, not part of D6, one `find_vertical_seam` call on a random $533 \times 800$ map took a median of 6.8 ms over 50 calls (Apple M4, Python 3.14, NumPy 2.5).

**NumPy-only core.** Filtering, energy, dynamic programming, seam removal and seam insertion use only NumPy. OpenCV is used for reading and writing files, colour conversion, the `cv2.resize` baseline, mask dilation, median filtering in D8, the mask-painting window, and as a reference in tests.

<!--
TODO(D6): section 4.2 waits for experiments/d6_runtime.py. Fill it in from
results/tables/d6_runtime.csv and remove this comment. Column names are not fixed yet;
the names below are the planned meaning, to be replaced with the real ones.

### 4.2 Runtime (D6)

Setup sentence: image and sizes. One photo from the data set, resized with cv2.resize to
long side 300, 600, 800 and 1200 px. Source: the size columns (H, W, number of pixels).
Hardware sentence: CPU, Python and NumPy version. Source: the environment columns.

Sentence 1, time per seam at each size, split into energy, DP and seam removal.
Source: per-step time columns (energy, DP, removal) and their total, one row per size.

Sentence 2, nested-loop DP against vectorised DP at the two sizes where both were run.
Source: the two DP time columns and their ratio. State the ratio as measured.

Sentence 3, total time to reduce the width by 25 percent at each size.
Source: the total shrink time column.

Sentence 4, linear fit of time against number of pixels: slope and R^2.
Source: the fit columns (or compute from the per-size rows and say so).

Sentence 5, comparison with the paper (Section 5): "Seam carving is linear in the number of
pixels"; 400 x 500 to 100 x 100 in about 2.2 s on 2007 hardware. Compare only the trend,
not the absolute time; hardware and image sizes differ.

Figure (one): ![Time per seam against number of pixels, with a linear fit (D6).](FIG/d6_runtime.png)

Length: 150 to 300 words. Report bad results too (for example a poor linear fit).
-->
