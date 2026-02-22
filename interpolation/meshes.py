"""
Mesh generation for polynomial interpolation.
Returns sorted 1D arrays of nodes with requested dtype (float32/float64).
"""

import numpy as np


def uniform_mesh(a, b, n, dtype=np.float64):
    """Uniform nodes on [a, b]: x_k = a + (b-a)*k/(n-1), k=0..n-1. Sorted ascending."""
    x = np.linspace(a, b, n, dtype=dtype)
    return np.sort(x)


def chebyshev_first_kind(a, b, n, dtype=np.float64):
    """
    Chebyshev nodes of the first kind on [a, b].
    On [-1,1]: t_k = cos((2k-1)*pi/(2n)), k=1..n. Then affine map to [a,b].
    Returns sorted ascending.
    """
    k = np.arange(1, n + 1, dtype=dtype)
    t = np.cos((2 * k - 1) * np.pi / (2 * n))
    x = (b - a) / 2 * t + (a + b) / 2
    return np.sort(x).astype(dtype)


def chebyshev_second_kind(a, b, n, dtype=np.float64):
    """
    Chebyshev nodes of the second kind on [a, b] (n points).
    On [-1,1]: t_k = cos(k*pi/(n-1)), k=0..n-1 (n nodes). Then affine map to [a,b].
    Returns sorted ascending.
    """
    if n == 1:
        return np.array([(a + b) / 2], dtype=dtype)
    k = np.arange(0, n, dtype=dtype)
    t = np.cos(k * np.pi / (n - 1))
    x = (b - a) / 2 * t + (a + b) / 2
    return np.sort(x).astype(dtype)
