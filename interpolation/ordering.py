import numpy as np


# Reorder mesh x according to mode.
def order_mesh(x, mode):
    x = np.asarray(x)
    n = x.size

    if mode == "increasing":
        return np.sort(x)
    if mode == "decreasing":
        return np.sort(x)[::-1].copy()
    if mode == "leja":
        return _leja_order(x)
    raise ValueError(f"Unknown mode: {mode}")


# Return indices such that x[indices] is the reordered mesh.
def order_mesh_indices(x, mode):
    x = np.asarray(x).ravel()
    n = x.size
    if mode == "increasing":
        return np.argsort(x)
    if mode == "decreasing":
        return np.argsort(x)[::-1]
    if mode == "leja":
        return _leja_order_indices(x)
    raise ValueError(f"Unknown mode: {mode}")

# Leja ordering: greedy selection maximizing product of distances to already selected.
def _leja_order(x):
    idx = _leja_order_indices(x)
    return np.asarray(x).ravel()[idx].copy()


def _leja_order_indices(x):
    x = np.asarray(x).ravel()
    n = x.size
    if n <= 1:
        return np.arange(n)
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
    return np.array(selected)
