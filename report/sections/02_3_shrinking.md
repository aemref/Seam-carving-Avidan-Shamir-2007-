## Shrinking

To reduce the width of an image by k pixels, we remove k vertical seams one after another
(paper Section 4.1). Each step computes the energy map of the current image, finds the
minimum-cost vertical seam with dynamic programming and removes it. Removal builds a boolean
(H, W) mask that is False exactly at (r, seam[r]) in every row r, selects the remaining
pixels and reshapes them to (H, W-1). The same function cuts
side arrays, such as a protection mask, that must stay aligned with the image.

The energy is recomputed after every removal. Removing a seam makes pixels that were two
columns apart into neighbours, so the gradients along the seam change.

Removed pixels lose their original positions, so we cut an index map together with the
image. At the start, idx[r, c] = c. Just before step t, idx[r, seam[r]] is the original
column of the pixel removed in row r. We write t at that position in an order map of the
original size and then cut idx with the same seam. After k steps the order map gives, for
every original pixel, the step that removed it or -1 if it was kept; removed = order >= 0
has exactly k True values per row. This map is used to draw all seams on the original
photo and to choose the seams duplicated when enlarging.

Height needs no separate code. The image and the side arrays are transposed (rows and
columns swapped, channels kept last), the width is shrunk, and the image, side arrays and
both maps are transposed back. The maps then have exactly k True values per column.

We compare seam carving with baselines that give the same output size:

- **Scaling** resizes the whole image with OpenCV: area interpolation when no side grows,
  bicubic otherwise.
- **Centre crop** keeps the central window of the target size.
- **Best crop** keeps the window with the highest total energy (paper Fig. 2b). Window sums
  come from an integral image, four look-ups per window.
- **Column removal** deletes the k columns with the lowest total energy and keeps the others
  in order (Fig. 2c).

For the energy-preservation measure we also delete the k lowest-energy pixels of each row
independently (Fig. 2e). By construction this keeps the highest mean energy, but the removed
pixels of neighbouring rows are not connected, so image structure breaks.
