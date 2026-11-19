## Finding the Cheapest Seam

- A vertical seam is an 8-connected path with one pixel per row (Week 3, slides 38–40)
- $M(r, c) = e(r, c) + \min\big(M(r-1, c-1),\, M(r-1, c),\, M(r-1, c+1)\big)$, then backtrack from the smallest value in the last row
- One seam costs $O(H \cdot W)$; ties go to the leftmost column
- Greedy (orange, dashed) costs 13, DP (blue) costs 7; greedy from the bottom row also misses (8)

![Top-down greedy seam (cost 13) and DP seam (cost 7) on the 3 × 4 example, with the energy e and the cumulative cost M](FIG/d6_dp_example.png)

Notes: One pixel per row, at most one column sideways: an 8-connected path from Week 3. M of a pixel is the cheapest way to reach it from the top. Each cell looks only at the three cells above, so one seam is one pass. Greedy, the orange path, takes the cheap top pixel and gets stuck: cost 13. DP keeps every column open and backtracks from the smallest M in the last row: the blue path, cost 7.

## Runtime

- The Python loop runs over rows only; each row is one NumPy step: shift, stack, `argmin`
- Horizontal seams reuse the same code on the transposed image

<!--
TODO(D6): finish this slide after experiments/d6_runtime.py has run on a data set photo,
then delete this comment. Cells of results/tables/d6_runtime.csv as (section, quantity, size):
Bullet 3: one seam at the 800 px working size, total and DP part:
  (per_seam, total_ms, long800), (per_seam, dp_ms, long800).
Bullet 4: vectorised DP against nested Python loops at long800:
  (dp_compare, naive_ms, long800), (dp_compare, vectorised_ms, long800),
  (dp_compare, speedup, long800).
Figure line to add (three panels: time per seam by step, 25 % width reduction, loops vs
vectorised DP): ![Runtime (D6)](FIG/d6_runtime.png)
Notes sentence to add: time per seam grows linearly with H x W, slope and R^2 from
(fit, per_seam_total_slope, all) and (fit, per_seam_total_r2, all), as the paper says
(Sec. 5). Keep Notes at 80 words or fewer.
-->

Notes: The slow way is two nested Python loops over rows and columns. We loop only over rows. For each row we shift the row above left and right, stack the three copies and take one argmin, so NumPy chooses the best parent for every column at once. Horizontal seams need no extra code: we transpose the image, find a vertical seam and transpose back.
