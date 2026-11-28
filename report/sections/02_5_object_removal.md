### 2.5 Object Removal with Masks

**Mechanism.** The user paints a removal mask and, optionally, a protection mask (`tools/mask_painter.py`). The seam search is unchanged (paper, Section 4.6): before every seam, the energy is computed as usual and masked pixels are overwritten, removal pixels with $-R$ and protected pixels with $+P$; protection wins where the masks overlap. The masks are carved together with the image, so they stay aligned with it.

**Choosing the weights.** Fixed constants such as $\pm 10^6$ need not dwarf every energy map, so we derive the weights from the current one. Let $m = \max |e|$ (at least 1, so that a flat map still gets nonzero weights). A vertical seam has $H$ pixels, so the base energies along any two seams differ by at most $2Hm$. Choosing

$$R > 2Hm$$

makes a seam that crosses one more removal pixel always cheaper, even when the base energy is negative. A seam through a single protected pixel can gain at most $(H-1)(R+m)$ elsewhere, so

$$P > H(R + m)$$

makes it more expensive than any seam that avoids protection. The code uses $R = (2H+1)\,m$ and $P = H(R+m) + m$. Both are finite, as the dynamic programming of Section 2.2 requires. For horizontal seams the image is transposed, so $H$ becomes the width.

**Direction.** Vertical seams need as many seams as the mask has pixels in its most crowded row, horizontal seams as many as in its most crowded column. `direction="auto"` takes the smaller count (the paper's "smaller vertical or horizontal diameter"), vertical on a tie. Seams are then removed one by one until no removal pixel is left.

**Stopping.** Removal pixels directly above a wide protected region cannot be reached without crossing it. By the choice of $P$, the optimal seam then removes no removal pixel at all; we discard it, keep those pixels (protection wins) and report their number. As a safeguard, the loop never removes more than $W-1$ (or $H-1$) seams.

**Shadows and halos.** A painted mask often misses the shadow or the blurred edge. With `dilate` $= k$, the removal mask is first grown by $k$ pixels with `cv2.dilate` over a $(2k+1) \times (2k+1)$ square, the max filter of Week 6, slide 46.

**Restoring the size.** After $n$ seams have been removed, $n$ seams are inserted with the method of Section 2.4 to return to the original size (paper, Fig. 12). The insertion does not use the protection mask.

**What changes.** Every seam crosses the whole image, so distant pixels shift too: the result is a rearranged image, not a local inpainting (paper, Section 4.6).
