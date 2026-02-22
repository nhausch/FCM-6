import numpy as np

from . import ordering


# Computes the Newton coefficients (divided differences) for the interpolant.
def divided_differences(x, f, dtype=np.float64):
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

# Evaluates Newton form using Horner-like recurrence.
# p = c_{n-1} + (x - x_{n-1}) * ( c_{n-2} + (x - x_{n-2}) * ( ... ) ).
def newton_eval(x_eval, x_nodes, coeffs, dtype=np.float64):
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

# Optional helper: order nodes then compute Newton coefficients.
# Useful for testing Leja vs increasing.
def setup_newton(x, f, dtype=np.float64, order_mode="increasing"):
    x = np.asarray(x, dtype=dtype).ravel()
    x_ordered = ordering.order_mesh(x, order_mode)
    coeffs, y = divided_differences(x_ordered, f, dtype)
    return x_ordered, coeffs, y
