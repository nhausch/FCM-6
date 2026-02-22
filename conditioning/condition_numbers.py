import numpy as np

from interpolation.barycentric_form1 import (
    NODE_TOL,
    barycentric1_condition_numerator,
    barycentric1_eval,
)


# Condition number kappa(x, n, y) = (sum_i |ell_i(x) y_i|) / |p_n(x)|.
# Returns 1D array of length len(x_eval).
# At nodes returns 1. Where |p_n(x)| is zero or tiny, returns np.nan.
def kappa_xy(x_eval, x_nodes, gamma, y, dtype=np.float64):
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    x_nodes = np.asarray(x_nodes, dtype=dtype).ravel()
    gamma = np.asarray(gamma, dtype=dtype).ravel()
    y = np.asarray(y, dtype=dtype).ravel()
    n = x_nodes.size
    m = x_eval.size

    num = barycentric1_condition_numerator(x_eval, x_nodes, gamma, y, dtype)
    p_vals = barycentric1_eval(x_eval, x_nodes, gamma, y, dtype)
    denom = np.abs(p_vals)

    scale = np.max(np.abs(x_nodes)) if n > 0 else dtype(1.0)
    tol = scale * NODE_TOL * max(n, 1)
    out = np.empty(m, dtype=dtype)
    out[:] = np.nan
    for k in range(m):
        d = x_eval[k] - x_nodes
        j_near = np.argmin(np.abs(d))
        if np.abs(d[j_near]) <= tol:
            out[k] = dtype(1.0)
            continue
        if denom[k] <= tol:
            continue
        out[k] = num[k] / denom[k]
    return out

# Lebesgue function kappa(x, n, 1) = sum_i |ell_i(x)|.
# ell_i(x) = (gamma_i/(x-x_i)) / sum_j (gamma_j/(x-x_j)), so
# kappa_x1 = (sum_i |gamma_i/(x-x_i)|) / |sum_j gamma_j/(x-x_j)|.
# At nodes returns 1.0.
# Returns 1D array of length len(x_eval).
def kappa_x1(x_eval, x_nodes, gamma, dtype=np.float64):
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    x_nodes = np.asarray(x_nodes, dtype=dtype).ravel()
    gamma = np.asarray(gamma, dtype=dtype).ravel()
    n = x_nodes.size
    m = x_eval.size
    out = np.empty(m, dtype=dtype)
    scale = np.max(np.abs(x_nodes)) if n > 0 else dtype(1.0)
    tol = scale * NODE_TOL * max(n, 1)

    for k in range(m):
        xk = x_eval[k]
        d = xk - x_nodes
        j_near = np.argmin(np.abs(d))
        if np.abs(d[j_near]) <= tol:
            out[k] = dtype(1.0)
            continue
        denom_sum = np.sum(gamma / d)
        denom_abs = np.abs(denom_sum)
        if denom_abs <= tol:
            out[k] = np.nan
            continue
        num_sum = np.sum(np.abs(gamma / d))
        out[k] = num_sum / denom_abs
    return out
