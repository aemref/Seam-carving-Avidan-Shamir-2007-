## 2.1 Energy map

The energy map decides which pixels are cheap to remove. The paper's basic energy is the sum of the absolute partial derivatives of the image I [1, Eq. 1]:

$$e_1(I) = \left|\frac{\partial I}{\partial x}\right| + \left|\frac{\partial I}{\partial y}\right|$$

A pixel in a uniform region has near-zero derivatives and low energy, so removing it changes little. Edges and texture have high energy and are avoided by the seams. Following our row/column indexing we write the derivatives as `d_rows` and `d_cols`, so e = |d_rows| + |d_cols|; the `l2` option uses sqrt(d_rows² + d_cols²) instead.

**Derivative operators.** `compute_energy` supports five operators:

- *forward difference*: d_cols[r, c] = f[r, c+1] − f[r, c], and the same along rows (zero in the last column and row);
- *Roberts*: the two diagonal differences of a 2 × 2 neighbourhood;
- *Sobel* (the default): 3 × 3 kernels that differentiate in one direction and smooth in the other;
- *Scharr*: the same layout as Sobel with weights 3, 10, 3, giving a more isotropic response;
- *Laplacian*: |∇²f|, a second derivative, as a counter-example; it is zero on linear ramps and responds to both sides of an edge.

Every first-derivative kernel sums to zero, which is why a flat region has exactly zero energy.

**Our own filtering.** The derivatives are computed with our `correlate2d` function, not with `cv2.Sobel` or `cv2.filter2D`. The image is padded with `np.pad`; for each kernel element, the padded image shifted by that element's offset is multiplied by its weight and added to the output. The loop runs over the 9 kernel elements, never over pixels. In our tests the result equals `cv2.filter2D` to within 1e-9 for all three border types, and our Sobel and Scharr energies match `cv2.Sobel` and `cv2.Scharr`.

**Implementation choices.**

- *Data type*: the image is converted to `float64` first; with `uint8`, 200 − 250 wraps to 206.
- *Borders*: reflect padding (`BORDER_REFLECT_101`). Zero padding creates an artificial edge around the frame, and seams would avoid the outer columns for no reason.
- *Colour*: `gray` converts BGR to grey (0.114 B + 0.587 G + 0.299 R); `sum` adds the energies of the three channels, which also catches edges between regions of equal brightness but different colour.
- *Pre-smoothing*: with `blur_sigma > 0` the image is first smoothed with a normalised Gaussian kernel of side ⌈6σ⌉ (made odd) to suppress noise.

The choice of default operator is made experimentally in Section 6.1.
