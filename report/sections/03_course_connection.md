# 3. Connection to the course

The project is tied to the week 5 topic *sharpening spatial filters*, through the image gradient. The paper's energy function, e1 = |∂I/∂x| + |∂I/∂y| [1, Eq. 1], is exactly the approximation of the gradient magnitude M(x, y) ≈ |g_x| + |g_y| given in the lecture (week 5, slide 56). The rest of the pipeline also relies on material from weeks 2, 3 and 6. Table 1 lists each topic and where it is used in our code and experiments.

**Table 1.** Course topics and where they are used in the project (slide numbers refer to the INF605 lecture notes [2]).

| Course topic | Week, slides | Where it is used in the project |
|---|---|---|
| Gradient and its magnitude, sum-of-absolute-values approximation | W5, 56 | Definition of the energy e1 (`energy.compute_energy`, `norm="l1"`); the `l2` option is the exact magnitude |
| Roberts and Sobel operators | W5, 57–59 | `method="roberts"` and `"sobel"`; `cv2.Sobel` is the reference in our tests |
| First and second derivatives | W5, 43–46 | `method="forward"` (forward difference); the derivative is 0 in flat regions, so seams pass there |
| Laplacian | W5, 47–51 | `method="laplacian"`, used in D1 as a counter-example of a second-derivative energy |
| Mechanics of linear filtering, correlation and convolution | W5, 5–14 | Our own `filters.correlate2d`; tested with the 180° rotated impulse response (slide 14) |
| Padding types | W5, 33 | `border="reflect"` by default, so the frame does not appear as an edge |
| Gaussian smoothing | W5, 26–31 | `filters.gaussian_kernel` and `blur_sigma` (side ⌈6σ⌉, 7 × 7 for σ = 1); noise suppression in D1 and D8 |
| Derivative kernels sum to zero | W5, 57 and 62 | Why a flat region has exactly zero energy; checked in `test_filters.py` |
| Coordinate convention, `img[r, c]` | W3, 13–14 | Row-then-column indexing everywhere; `d_rows` / `d_cols` instead of x / y |
| Interpolation: nearest neighbour, bilinear, bicubic | W3, 30–36 | The scaling baseline (`cv2.resize` with `INTER_AREA` / `INTER_CUBIC`); masks are resized with nearest neighbour |
| Neighbourhood, adjacency, paths | W3, 38–40 | A seam is an 8-connected path: \|seam[r] − seam[r−1]\| ≤ 1 |
| Contrast stretching, log transform | W6, 8 and 13–14 | Displaying energy and cumulative cost maps (`viz.energy_to_image`, min-max and optional log) |
| Noise models and adding noise | W6, 33–37 | D8: Gaussian (σ = 15) and salt-and-pepper (5 %) noise on a photo |
| Mean, median, max / min filters | W6, 41–46 | D8: Gaussian and median pre-filters before the energy; `cv2.dilate` (max filter) to widen removal masks |
| Perception is misleading, a numerical measure is needed | W2, 15 | Evaluation pairs side-by-side figures with numbers: mean energy of the kept pixels, seam overlap, run time |

Three of these links are central to our part of the work.

**The energy is a gradient.** Our default energy correlates the grey image with the two Sobel kernels of slide 57 and adds the absolute responses, as in the approximation of slide 56. We implemented the correlation ourselves as the "sum of products" of slides 5–12, and verified it against `cv2.filter2D`.

**Flat means cheap.** Because every first-derivative kernel sums to zero (slides 57 and 62), a region of constant intensity has zero energy, and the dynamic programming step sends the seams through it. This is the reason seam carving works well on sky, water and grass, and badly on dense texture where no flat region exists.

**Noise and smoothing.** A derivative amplifies noise. Experiment D8 adds the noise models of week 6 to a photo and compares the seams with and without a Gaussian or median pre-filter, which connects the smoothing filters of weeks 5 and 6 to the quality of the result.
