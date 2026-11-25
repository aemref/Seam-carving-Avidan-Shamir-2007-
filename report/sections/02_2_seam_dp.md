### 2.2 Minimum Seam and Dynamic Programming

**Seams.** The paper indexes a vertical seam pixel as $(x(i), i)$ with $x$ the column. We index an $H \times W$ image as $(r, c)$, row first, and write $s(r)$ for the column a vertical seam takes in row $r$ (`seam[r]` in the code). A vertical seam has one pixel per row and moves at most one column between rows (paper, Eq. 2); a horizontal seam exchanges rows and columns (Eq. 3); the optimal seam minimises the summed energy (Eq. 4):

$$s = \{(r, s(r))\}_{r=0}^{H-1}, \qquad |s(r) - s(r-1)| \le 1,$$

$$t = \{(t(c), c)\}_{c=0}^{W-1}, \qquad |t(c) - t(c-1)| \le 1,$$

$$E(s) = \sum_{r=0}^{H-1} e\big(r, s(r)\big), \qquad s^* = \arg\min_s E(s).$$

**Seams as 8-connected paths.** In Week 3 (slides 38–40), $N_8(p)$ is the set of all eight neighbours of $p$, and $q$ is 8-adjacent to $p$ if $q \in N_8(p)$. Consecutive seam pixels differ by one row and at most one column, so each is 8-adjacent to the next: a seam is an 8-connected path from top to bottom, as the paper states, using only the three lower neighbours. With 4-adjacency instead, a seam could only be a straight column.

**Dynamic programming.** Let $M(r, c)$ be the cost of the cheapest seam from row 0 to pixel $(r, c)$:

$$M(0, c) = e(0, c), \qquad M(r, c) = e(r, c) + \min\big(M(r-1, c-1),\ M(r-1, c),\ M(r-1, c+1)\big),$$

where neighbours outside the image count as $+\infty$. Every seam reaching $(r, c)$ passes through one of the three pixels above, and its upper part must itself be cheapest; hence the recurrence is exact. It also holds for negative energies (mask weights, Section 2.5). We store the offset $b(r, c) \in \{-1, 0, +1\}$ of the chosen predecessor (`back` in the code) and backtrack from the bottom row (paper, Section 3):

$$s(H-1) = \arg\min_c M(H-1, c), \qquad s(r-1) = s(r) + b\big(r, s(r)\big).$$

Both $\arg\min$ steps break ties towards the leftmost column.

**Complexity.** Each entry of $M$ needs one addition and a three-way minimum, so one seam costs $O(H \cdot W)$ time and memory; the paper likewise calls seam carving linear in the number of pixels (Section 5). Exhaustive search would test up to $W \cdot 3^{H-1}$ seams.

**Why a greedy choice fails.** A greedy rule starts at the cheapest pixel of row 0 and, in each later row, steps to the cheapest 8-adjacent pixel in the next row. On the map below it picks columns $(0, 0, 1)$ with cost 13; run upwards from the bottom row, it picks $(2, 3, 2)$ with cost 8. Dynamic programming finds $(1, 2, 2)$ with cost 7. Enumerating all 26 valid seams shows that 7 is the unique minimum, and no choice made by either greedy rule or by dynamic programming involves a tie. Greedy commits to a cheap start without looking ahead; dynamic programming keeps the best partial seam for every column until the last row.

|         | $c=0$ | $c=1$ | $c=2$ | $c=3$ |
|---------|-------|-------|-------|-------|
| $r=0$   | *1*   | **4** | 6     | 8     |
| $r=1$   | *4*   | 9     | **2** | 1     |
| $r=2$   | 9     | *8*   | **1** | 6     |

Table: Energy $e$ of the counterexample. Italic: top-down greedy seam, $1+4+8=13$. Bold: optimal seam, $4+2+1=7$.

|         | $c=0$ | $c=1$ | $c=2$ | $c=3$ |
|---------|-------|-------|-------|-------|
| $r=0$   | 1     | 4     | 6     | 8     |
| $r=1$   | 5     | 10    | 6     | 7     |
| $r=2$   | 14    | 13    | 7     | 12    |

Table: Cumulative cost $M$ of the same map.
