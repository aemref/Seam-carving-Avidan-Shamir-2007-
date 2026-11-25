## 4. Implementation and Runtime

### 4.1 Architecture

**Modules.** The package `seamcarving` has one module per step of the method and one owner per module. The dependency chain follows the algorithm: `filters` → `energy` (map $e$) → `seam` (map $M$, minimum seam) → `carve` (seam removal) → `enlarge` (seam insertion) and `object_removal` → `resize` → `cli`. `baselines` and `viz` sit beside the chain; each experiment is a script in `experiments/`. All signatures were fixed before any module was written, so the five members worked in parallel.

**Energy as a parameter.** The shrinking loop in `carve` repeats `energy_fn(img, carry)` → `seam_fn(energy)` → remove the seam. Both functions are parameters (defaults: `compute_energy`, `find_vertical_seam`), so `carve` knows nothing about operators or masks. Experiment D1 swaps the operator; object removal wraps the base energy so that masked pixels receive large negative or positive weights; `resize` does the same with a protection mask when shrinking. `carry` is a tuple of side arrays, such as these masks, cut with the same seam as the image; `energy_fn` receives them in their current state, so a mask stays aligned with the image.

**Horizontal seams.** `shrink_height` swaps the first two axes of the image and of every carried array, calls `shrink_width`, and swaps the results back. A horizontal seam is thus a vertical seam of the transpose, and `seam` implements vertical seams only.

**Vectorisation.** `cumulative_map` loops in Python over rows only. For row $r$, row $r-1$ of $M$ is copied into a buffer of length $W + 2$ with $+\infty$ at both ends. Three shifted views of it (left, middle and right parent) are stacked into a $3 \times W$ array, and one `argmin` along the first axis gives every column's best parent at once; since `argmin` returns the first minimum, ties go left. $M(r, \cdot)$ is the energy plus the chosen values, and the offset `best - 1` is stored as `int8` in `back`. Backtracking loops over the $H$ rows.

**NumPy-only core.** Filtering, energy (including the colour-to-grey conversion), dynamic programming, seam removal and seam insertion use only NumPy. OpenCV appears around this core: image file encoding and decoding, `cv2.resize` for the scaling baseline, for the upscaling step of content amplification and for preparing the photos, and `cv2.Sobel`, `cv2.Scharr` and `cv2.Laplacian` as references in the energy tests.

### 4.2 Runtime (D6)

We timed our implementation on one data set photo (`insan_01_emre.jpg`, $600 \times 800$), resized with `cv2.resize` to a long side of 300, 600, 800 and 1200 pixels, on an Apple M4 with Python 3.14.7 and NumPy 2.5.5. All times are medians; the energy is Sobel on the grey image.

One seam takes 3.1 ms at $225 \times 300$, 11.7 ms at $450 \times 600$, 20.9 ms at $600 \times 800$ and 46.6 ms at $900 \times 1200$. At the 800-pixel working size the DP takes 40 % of this, seam removal 39 % and the energy 21 %. A least-squares line through the four sizes has a slope of 43.0 ns per pixel and $R^2 = 1.000$ after rounding, in line with the paper's statement that seam carving is linear in the number of pixels (Section 5).

Reducing the width by 25 % takes 0.21 s, 1.58 s, 3.69 s and 12.27 s for 75, 150, 200 and 300 seams. This total is not linear in $H \times W$, because the number of seams also grows with the width; against the pixels actually processed it is linear again: 43.2 ns per pixel, $R^2 = 1.000$.

The same DP written as two nested Python loops takes 23.8 ms at $225 \times 300$ and 179.1 ms at $600 \times 800$; the vectorised version takes 1.55 ms and 8.42 ms, a factor of 15 and 21. Both give identical $M$ and `back`.

The paper reports about 2.2 s for retargeting a $400 \times 500$ image to $100 \times 100$. Our code needs 2.76 s for 400 rows by 500 columns and 2.53 s for 500 by 400; the paper does not say which side is the height. The times are not directly comparable: the implementations differ and the paper does not state its hardware.

![Runtime of our implementation (D6). (a) Time per seam by step against H x W, with a linear fit. (b) Total time to reduce the width by 25 %, against the pixels processed, with a linear fit. (c) DP alone: nested Python loops against the vectorised cumulative map.](FIG/d6_runtime.png)
