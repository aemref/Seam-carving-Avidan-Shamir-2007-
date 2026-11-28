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
- One seam at 600 × 800: 20.9 ms, of which the DP is 8.4 ms; 43 ns per pixel, linear
- Vectorised DP against nested loops at 600 × 800: 8.4 ms against 179 ms, 21 times faster

![Runtime (D6): time per seam by step, 25 % width reduction, loops against vectorised DP](FIG/d6_runtime.png)

Notes: The slow way is two nested Python loops over rows and columns. We loop only over rows: shift the row above left and right, stack the three copies and take one argmin, so NumPy picks the best parent for every column at once. That makes the DP 21 times faster at 600 by 800. Time per seam grows linearly with the number of pixels, as the paper says.
