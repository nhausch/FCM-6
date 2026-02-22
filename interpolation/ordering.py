"""
Ordering of interpolation nodes (for Newton form stability).
Returns reordered 1D array of nodes (same length as input).
"""

import numpy as np


def order_mesh(x, mode):
    """
    Reorder mesh x according to mode.

    Parameters
    ----------
    x : array-like, shape (n,)
        Node positions.
    mode : str
        - "increasing": ascending order
        - "decreasing": descending order
        - "leja": Leja ordering (greedy max product of distances from selected)

    Returns
    -------
    x_ordered : np.ndarray, shape (n,)
        Reordered nodes (same dtype as x).
    """
    x = np.asarray(x)
    n = x.size

    if mode == "increasing":
        return np.sort(x)
    if mode == "decreasing":
        return np.sort(x)[::-1].copy()
    if mode == "leja":
        return _leja_order(x)
    raise ValueError(f"Unknown mode: {mode}")


def _leja_order(x):
    """Leja ordering: greedy selection maximizing product of distances to already selected."""
    x = np.asarray(x).ravel()
    n = x.size
    if n <= 1:
        return x.copy()
    # First point: argmax |x_i|
    idx = np.argmax(np.abs(x))
    selected = [idx]
    remaining = list(range(n))
    remaining.remove(idx)
    for _ in range(n - 1):
        best_j = None
        best_prod = -1.0
        xs = x[selected]
        for j in remaining:
            prod = np.prod(np.abs(x[j] - xs))
            if prod > best_prod:
                best_prod = prod
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)
    return x[selected].copy()
