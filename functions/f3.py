"""f3(x) = ℓ_n(x), the nth Lagrange basis function for a given mesh. Mesh-dependent."""

import numpy as np


def make_f3(x_nodes):
    """
    Constructs f3(x) = ℓ_n(x), the nth Lagrange basis function
    associated with the given mesh.

    Parameters
    ----------
    x_nodes : array_like
        Interpolation nodes, length n+1.

    Returns
    -------
    func : callable
        Evaluates ℓ_n(x).
    interval : tuple
        (min(x_nodes), max(x_nodes)).
    roots : ndarray
        Roots of ℓ_n(x) (all nodes except x_n), i.e. x_0, ..., x_{n-1}.
    denom : float
        Normalization constant prod_{j=0}^{n-1} (x_n - x_j).
    """
    x_nodes = np.asarray(x_nodes)
    n = len(x_nodes) - 1
    roots = x_nodes[:n].copy()
    x_n = x_nodes[n]

    denom = np.array(1.0, dtype=x_nodes.dtype)
    for r in roots:
        denom *= (x_n - r)

    def func(x):
        x = np.asarray(x)
        result = np.ones_like(x, dtype=np.result_type(x, roots))
        roots_same = roots.astype(result.dtype, copy=False)
        for r in roots_same:
            result *= (x - r)
        return result / np.asarray(denom, dtype=result.dtype)

    interval = (float(np.min(x_nodes)), float(np.max(x_nodes)))
    return func, interval, roots, denom
