"""
Newton form of polynomial interpolant.
p(x) = c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ... + c_n(x-x_0)...(x-x_{n-1})
Coefficients are divided differences: c_k = f[x_0,...,x_k].
"""

import numpy as np

from . import ordering


def divided_differences(x, f, dtype=np.float64):
    """
    Compute Newton coefficients (divided differences) for interpolant.

    Parameters
    ----------
    x : array-like, shape (n,)
        Interpolation nodes (sorted; use ordering.order_mesh if needed).
    f : callable
        Function (vectorized).
    dtype : np.dtype

    Returns
    -------
    coeffs : np.ndarray, shape (n,)
        c_k = f[x_0,...,x_k], so p = c_0 + c_1(x-x_0) + ...
    y : np.ndarray, shape (n,)
        f(x).
    """
    x = np.asarray(x, dtype=dtype).ravel()
    n = x.size
    y = np.asarray(f(x), dtype=dtype).ravel()
    if y.size != n:
        raise ValueError("f(x) must return array of same length as x")
    # d[i,j] = f[x_i,...,x_j], j >= i
    d = np.zeros((n, n), dtype=dtype)
    d[:, 0] = y
    for j in range(1, n):
        for i in range(0, n - j):
            d[i, j] = (d[i + 1, j - 1] - d[i, j - 1]) / (x[i + j] - x[i])
    coeffs = d[0, :].copy()
    return coeffs, y


def newton_eval(x_eval, x_nodes, coeffs, dtype=np.float64):
    """
    Evaluate Newton form using Horner-like recurrence.
    p = c_{n-1} + (x - x_{n-1}) * ( c_{n-2} + (x - x_{n-2}) * ( ... ) ).

    Parameters
    ----------
    x_eval : array-like, shape (m,)
    x_nodes : array-like, shape (n,)
        Nodes in same order as used for divided_differences.
    coeffs : array-like, shape (n,)
        From divided_differences.
    dtype : np.dtype

    Returns
    -------
    p : np.ndarray, shape (m,)
    """
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    x_nodes = np.asarray(x_nodes, dtype=dtype).ravel()
    coeffs = np.asarray(coeffs, dtype=dtype).ravel()
    n = coeffs.size
    if x_nodes.size != n:
        raise ValueError("x_nodes and coeffs must have same length")
    m = x_eval.size
    p = np.full(m, coeffs[n - 1], dtype=dtype)
    for i in range(n - 2, -1, -1):
        p = p * (x_eval - x_nodes[i]) + coeffs[i]
    return p


def setup_newton(x, f, dtype=np.float64, order_mode="increasing"):
    """
    Optional helper: order nodes then compute Newton coefficients.
    Useful for testing Leja vs increasing.

    Parameters
    ----------
    x : array-like
        Raw nodes (e.g. from meshes).
    f : callable
    dtype : np.dtype
    order_mode : str
        "increasing", "decreasing", or "leja"

    Returns
    -------
    x_ordered : np.ndarray
    coeffs : np.ndarray
    y : np.ndarray
    """
    x = np.asarray(x, dtype=dtype).ravel()
    x_ordered = ordering.order_mesh(x, order_mode)
    coeffs, y = divided_differences(x_ordered, f, dtype)
    return x_ordered, coeffs, y
