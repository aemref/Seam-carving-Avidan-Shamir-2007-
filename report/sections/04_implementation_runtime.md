## 4. Implementation and Runtime

### 4.1 Architecture

**Modules.** The package `seamcarving` has one module per step of the method and one owner per module. The dependency chain follows the algorithm: `filters` → `energy` (map $e$) → `seam` (map $M$, minimum seam) → `carve` (seam removal) → `enlarge` (seam insertion) and `object_removal` → `resize` → `cli`. `baselines` and `viz` sit beside the chain; each experiment is a script in `experiments/`. All signatures were fixed before any module was written, so the five members worked in parallel.

**Energy as a parameter.** The shrinking loop in `carve` repeats `energy_fn(img, carry)` → `seam_fn(energy)` → remove the seam. Both functions are parameters (defaults: `compute_energy`, `find_vertical_seam`), so `carve` knows nothing about operators or masks. Experiment D1 swaps the operator; object removal wraps the base energy so that masked pixels receive large negative or positive weights; `resize` does the same with a protection mask when shrinking. `carry` is a tuple of side arrays, such as these masks, cut with the same seam as the image; `energy_fn` receives them in their current state, so a mask stays aligned with the image.

**Horizontal seams.** `shrink_height` swaps the first two axes of the image and of every carried array, calls `shrink_width`, and swaps the results back. A horizontal seam is thus a vertical seam of the transpose, and `seam` implements vertical seams only.

**Vectorisation.** `cumulative_map` loops in Python over rows only. For row $r$, row $r-1$ of $M$ is copied into a buffer of length $W + 2$ with $+\infty$ at both ends. Three shifted views of it (left, middle and right parent) are stacked into a $3 \times W$ array, and one `argmin` along the first axis gives every column's best parent at once; since `argmin` returns the first minimum, ties go left. $M(r, \cdot)$ is the energy plus the chosen values, and the offset `best - 1` is stored as `int8` in `back`. Backtracking loops over the $H$ rows.

**NumPy-only core.** Filtering, energy (including the colour-to-grey conversion), dynamic programming, seam removal and seam insertion use only NumPy. OpenCV appears around this core: image file encoding and decoding, `cv2.resize` for the scaling baseline, for the upscaling step of content amplification and for preparing the photos, and `cv2.Sobel`, `cv2.Scharr` and `cv2.Laplacian` as references in the energy tests.

<!--
TODO(D6): section 4.2 waits for a run of experiments/d6_runtime.py on a data set photo.
Fill it in from results/tables/d6_runtime.csv, then delete this comment.
The CSV is in long format, one number per row. Header:
section,quantity,size,height,width,value,unit,statistic,n,notes
A cell is named below as (section, quantity, size); the number is in "value".
Sizes: long300, long600, long800, long1200. Timings are medians (statistic, n columns).

### 4.2 Runtime (D6)

Setup (1 to 2 sentences). Photo: (input, photo, "") value, its size from height/width.
Resized with cv2.resize to long side 300, 600, 800 and 1200 px; H and W of each size from
the height/width columns of (per_seam, pixels, longNNN). Machine: (env, cpu), (env, python),
(env, numpy). Method: (method, timer), (method, statistic), energy settings from
(method, energy).

Sentence 1, time per seam by step. For each size: (per_seam, energy_ms, longNNN),
(per_seam, dp_ms, longNNN), (per_seam, remove_ms, longNNN), (per_seam, total_ms, longNNN).
Name the step with the largest share: (per_seam, energy_share | dp_share | remove_share,
longNNN). Panel (a) of the figure.

Sentence 2, linear in the number of pixels (the paper's claim, Sec. 5: "Seam carving is
linear in the number of pixels"). Fit of per-seam total time against H x W:
(fit, per_seam_total_slope, all) in ns per pixel, (fit, per_seam_total_intercept, all) in ms,
(fit, per_seam_total_r2, all). Panel (a).

Sentence 3, shrinking the width by 25 percent. For each size: (shrink25, seams, longNNN),
(shrink25, total_s, longNNN), (shrink25, ms_per_seam, longNNN). Say explicitly that the
total is NOT linear in the original H x W, because the number of seams k also grows with W.
It is fitted against the pixels actually processed, the sum of H x (W - i) over the k seams:
(shrink25, work_pixels, longNNN), (fit, shrink25_total_slope, all),
(fit, shrink25_total_intercept, all), (fit, shrink25_total_r2, all). Panel (b).

Sentence 4, what vectorisation buys (DP only). At long300 and long800:
(dp_compare, naive_ms, longNNN), (dp_compare, vectorised_ms, longNNN),
(dp_compare, speedup, longNNN). Both versions gave identical M and back:
(dp_compare, same_M_and_back, longNNN). Panel (c).

Sentence 5, the paper's own example, measured with resize on the same photo: 400 x 500 to
100 x 100. Our time: (paper, seconds_assumed, 400x500->100x100) for 400 rows x 500 columns,
and (paper, seconds_swapped, 500x400->100x100) because the paper does not say which side is
the height. Paper: (paper, paper_seconds, 400x500->100x100), "about 2.2 seconds".
Ratios: (paper, paper_over_ours_assumed, 400x500->100x100),
(paper, paper_over_ours_swapped, 500x400->100x100). Absolute times may be given, but state
that the machines and the implementations differ: the paper's timing is from its own
implementation, published in 2007, and the paper does not name its hardware.

Figure (one, three panels: (a) time per seam by step with linear fit, (b) 25 percent width
reduction with linear fit, (c) nested-loop vs vectorised DP):
![Runtime of our implementation (D6). (a) Time per seam by step against H x W, with a linear fit. (b) Total time to reduce the width by 25 %, against the pixels processed, with a linear fit. (c) DP alone: nested Python loops against the vectorised cumulative map.](FIG/d6_runtime.png)

Length: 150 to 300 words. Report weak results too (for example a low R^2).
-->
