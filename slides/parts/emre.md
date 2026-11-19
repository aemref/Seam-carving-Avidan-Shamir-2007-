## Finding the Cheapest Seam

- A vertical seam is an 8-connected path with one pixel per row (Week 3, slides 38–40)
- $M(r, c) = e(r, c) + \min\big(M(r-1, c-1),\, M(r-1, c),\, M(r-1, c+1)\big)$, then backtrack from the smallest value in the last row
- One seam costs $O(H \cdot W)$; ties go to the leftmost column
- Greedy is not optimal: cost 13 top-down, 8 bottom-up, 7 with DP

![Greedy path (cost 13) and DP path (cost 7) on the 3 × 4 example](FIG/d6_dp_example.png)

Notes: A seam takes one pixel per row and moves at most one column: an 8-connected path from Week 3. M of a pixel is the cheapest way to reach it from the top row. Each cell looks only at the three cells above, so one seam is one pass over the image. Greedy takes the cheap top pixel and gets stuck in expensive pixels: cost 13. DP keeps every column open until the last row and finds 7.

## Runtime

- The Python loop runs over rows only; each row is one NumPy step: shift, stack, `argmin`
- Horizontal seams reuse the same code on the transposed image
- Development check, not D6: one seam on a random 533 × 800 map, median 6.8 ms (Apple M4)

<!--
TODO(D6): finish this slide after experiments/d6_runtime.py has run, then remove this comment.
Fourth bullet, from results/tables/d6_runtime.csv: time per seam at the largest size and the
speed-up of the vectorised DP over the nested-loop DP (DP time columns, their ratio).
Figure line to add: ![Time per seam against number of pixels (D6)](FIG/d6_runtime.png)
Notes sentence to add: whether time grows linearly with the number of pixels (fit columns),
compared with the paper's statement that seam carving is linear in the number of pixels.
Keep Notes at 80 words or fewer.
-->

Notes: The slow way is two nested Python loops over rows and columns. We loop only over rows. For each row we shift the row above left and right, stack the three copies and take one argmin, so NumPy chooses the best parent for every column at once. Horizontal seams need no extra code: we transpose the image, find a vertical seam and transpose back.
