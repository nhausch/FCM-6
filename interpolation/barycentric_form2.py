import numpy as np

from . import meshes
from .barycentric_form1 import NODE_TOL, compute_gamma


def compute_beta(x, mesh_type, dtype=np.float64):

    # Use the same weights as Form 1 (gamma) so that the interpolant matches
    # regardless of mesh_type. mesh_type is used only by setup_barycentric2
    # to select which mesh to build.
    return compute_gamma(x, dtype)

# Builds the mesh, computes beta and y = f(x) for Barycentric Form 2.
def setup_barycentric2(mesh_type, a, b, n, f, dtype=np.float64):
    if mesh_type == "uniform":
        x = meshes.uniform_mesh(a, b, n, dtype)
    elif mesh_type == "cheb1":
        x = meshes.chebyshev_first_kind(a, b, n, dtype)
    elif mesh_type == "cheb2":
        x = meshes.chebyshev_second_kind(a, b, n, dtype)
    else:
        raise ValueError(f"Unknown mesh_type: {mesh_type}")

    beta = compute_beta(x, mesh_type, dtype)
    y = np.asarray(f(x), dtype=dtype).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError("f(x) must return array of same length as x")
    return x, beta, y


# Evaluates the interpolant at x_eval using Barycentric Form 2.
# p(x) = (sum_i beta_i*y_i/(x-x_i)) / (sum_i beta_i/(x-x_i)).
# Uses exact-node and near-node safeguards.
def barycentric2_eval(x_eval, x_nodes, beta, y, dtype=np.float64):
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    x_nodes = np.asarray(x_nodes, dtype=dtype).ravel()
    beta = np.asarray(beta, dtype=dtype).ravel()
    y = np.asarray(y, dtype=dtype).ravel()
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
            out[k] = y[j_near]
            continue
        denom = np.sum(beta / d)
        num = np.sum(beta * y / d)
        out[k] = num / denom
    return out
