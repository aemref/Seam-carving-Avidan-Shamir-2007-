"""Minimum seam search with dynamic programming (paper Eq. 2-4). Owner: Emre.

The energy may contain negative and very large values (mask weights); it never contains inf.
"""

import numpy as np


def cumulative_map(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the cumulative minimum cost map.

    Args:
        energy: (H, W) float64.

    Returns:
        M: (H, W) float64 cumulative cost.
        back: (H, W) int8 column offset of the chosen parent in the row above: -1 / 0 / +1.
    """
    raise NotImplementedError


def find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    """Find the minimum-cost vertical seam.

    Args:
        energy: (H, W) float64.

    Returns:
        (H,) int64 with ``|seam[r] - seam[r-1]| <= 1``; ties resolve to the leftmost column.
    """
    raise NotImplementedError


def seam_cost(energy: np.ndarray, seam: np.ndarray) -> float:
    """Return the total energy along a vertical seam."""
    raise NotImplementedError


def is_valid_seam(seam: np.ndarray, width: int) -> bool:
    """Check that a seam stays inside ``[0, width)`` and moves at most one column per row."""
    raise NotImplementedError
