"""Minimum seam search with dynamic programming (paper Eq. 2-4). Owner: Emre.

A vertical seam picks one pixel per row, and consecutive rows differ by at most one column
(8-connected path, Eq. 2). Its cost is the sum of the energy along it (Eq. 4). The minimum is
found with the dynamic program

    M(r, c) = e(r, c) + min( M(r-1, c-1), M(r-1, c), M(r-1, c+1) ),    M(0, c) = e(0, c)

followed by backtracking from the smallest value in the last row.

Tie rule (part of the contract, other modules rely on it): ties always go to the leftmost
candidate, both in the last row and among the three parents (offsets -1, 0, +1 in that order).
On a constant energy map the seam is therefore column 0 in every row.

The energy may contain negative and very large values (mask weights); it never contains inf.
"""

import numpy as np


def _as_energy(energy: np.ndarray) -> np.ndarray:
    """Return ``energy`` as a 2-D float64 array without modifying (or copying, if possible) it."""
    e = np.asarray(energy, dtype=np.float64)
    if e.ndim != 2 or e.size == 0:
        raise ValueError(f"energy must be a non-empty (H, W) array, got shape {e.shape}")
    if not np.isfinite(e).all():
        raise ValueError("energy must not contain inf or nan")
    return e


def cumulative_map(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the cumulative minimum cost map.

    ``M[r, c]`` is the cost of the cheapest seam that starts in row 0 and ends at ``(r, c)``.
    Each row is computed from the row above in one vectorised step; the only Python loop runs
    over the rows. A missing neighbour at the left or right border counts as ``+inf``, so it is
    never chosen (the middle parent always exists, so ``M`` stays finite).

    Args:
        energy: (H, W) energy map. Any real dtype is accepted and converted to float64; the
            input is never modified. Negative and very large values are allowed.

    Returns:
        M: (H, W) float64 cumulative cost; ``M[0] == energy[0]``.
        back: (H, W) int8 column offset of the chosen parent in the row above: -1 / 0 / +1.
            Row 0 is all zeros. On ties the leftmost parent wins.

    Raises:
        ValueError: if ``energy`` is not a non-empty 2-D array or contains inf / nan.
    """
    e = _as_energy(energy)
    n_rows, n_cols = e.shape
    M = np.empty((n_rows, n_cols), dtype=np.float64)
    back = np.zeros((n_rows, n_cols), dtype=np.int8)
    M[0] = e[0]

    # The previous row with one +inf on each side, so every column has three parents.
    padded = np.full(n_cols + 2, np.inf)
    cols = np.arange(n_cols)
    for r in range(1, n_rows):
        padded[1:-1] = M[r - 1]
        # Row k of `parents` is the parent at offset k - 1 (left, middle, right).
        parents = np.stack([padded[:-2], padded[1:-1], padded[2:]])
        best = np.argmin(parents, axis=0)  # first minimum = leftmost parent
        M[r] = e[r] + parents[best, cols]
        back[r] = best - 1
    return M, back


def find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    """Find the minimum-cost vertical seam.

    Builds ``M`` and ``back`` with :func:`cumulative_map`, starts at the smallest value of the
    last row and walks up the rows following the stored parent offsets.

    Args:
        energy: (H, W) energy map, same rules as :func:`cumulative_map` (any real dtype,
            negative and large values allowed, input not modified).

    Returns:
        (H,) int64, ``seam[r]`` is the column chosen in row ``r``;
        ``|seam[r] - seam[r-1]| <= 1``. Ties resolve to the leftmost column, so the result is
        deterministic (a constant map gives column 0 in every row).
    """
    M, back = cumulative_map(energy)
    n_rows = M.shape[0]
    seam = np.empty(n_rows, dtype=np.int64)
    c = int(np.argmin(M[-1]))  # first minimum = leftmost end column
    seam[-1] = c
    for r in range(n_rows - 1, 0, -1):
        # int(...) matters: adding an int8 to a large column index overflows in NumPy 2.
        c += int(back[r, c])
        seam[r - 1] = c
    return seam


def seam_cost(energy: np.ndarray, seam: np.ndarray) -> float:
    """Return the total energy along a vertical seam."""
    raise NotImplementedError


def is_valid_seam(seam: np.ndarray, width: int) -> bool:
    """Check that a seam stays inside ``[0, width)`` and moves at most one column per row."""
    raise NotImplementedError
