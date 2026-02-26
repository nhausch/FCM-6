import numpy as np

from conditioning.statistics import sup_norm


# Sup norm of the difference between approximate and reference values.
# Returns max_i |p_approx[i] - p_ref[i]|. Handles NaN via sup_norm.
def forward_error_sup(p_approx, p_ref):
    p_approx = np.asarray(p_approx).ravel()
    p_ref = np.asarray(p_ref).ravel()
    if p_approx.size != p_ref.size:
        raise ValueError("p_approx and p_ref must have same length")
    return sup_norm(p_approx - p_ref)

# Vector of absolute differences |p_approx[i] - p_ref[i]| for plotting or per-point analysis.
def forward_error_vector(p_approx, p_ref):
    p_approx = np.asarray(p_approx).ravel()
    p_ref = np.asarray(p_ref).ravel()
    if p_approx.size != p_ref.size:
        raise ValueError("p_approx and p_ref must have same length")
    return np.abs(p_approx - p_ref)

# Observed forward error over a simple condition-based bound.
# bound = Lambda_n * eps * max(|y_ref|) if y_ref given (else max(..., 1e-16));
# else bound = max(Lambda_n * eps, 1e-16).
# Returns forward_error_sup_val / bound. Ratio > 1 means error exceeds the bound.
def stability_ratio(forward_error_sup_val, Lambda_n, eps, y_ref=None):
    if y_ref is not None:
        y_ref = np.asarray(y_ref).ravel()
        scale = np.nanmax(np.abs(y_ref)) if y_ref.size > 0 else 1.0
        if scale <= 0:
            scale = 1.0
        bound = Lambda_n * eps * scale
    else:
        bound = max(Lambda_n * eps, 1e-16)
    if bound <= 0:
        bound = 1e-16
    return forward_error_sup_val / bound

# Compute bound, whether error is within it, and ratio.
# Returns dict: bound, forward_error_sup, within_bound, ratio.
def compare_to_bound(forward_error_sup_val, Lambda_n, eps, y_ref=None):
    if y_ref is not None:
        y_ref = np.asarray(y_ref).ravel()
        scale = np.nanmax(np.abs(y_ref)) if y_ref.size > 0 else 1.0
        if scale <= 0:
            scale = 1.0
        bound = Lambda_n * eps * scale
    else:
        bound = max(Lambda_n * eps, 1e-16)
    if bound <= 0:
        bound = 1e-16
    ratio = forward_error_sup_val / bound
    return {
        "bound": bound,
        "forward_error_sup": forward_error_sup_val,
        "within_bound": forward_error_sup_val <= bound,
        "ratio": ratio,
    }


def verify_barycentric2_forward_bound(
    p_hat,
    p_ref,
    k_xy,
    k_x1,
    n,
    eps,
):
    """
    Verifies the Higham forward error bound for Barycentric Form 2:

        |p - p_hat| / |p|
        <= (3n+4) κ(x,n,y) u + (3n+2) κ(x,n,1) u + O(u^2)

    Parameters
    ----------
    p_hat : array
        Computed interpolant (e.g. single precision) evaluated on grid.
    p_ref : array
        Reference interpolant (e.g. double precision) on same grid.
    k_xy : array
        κ(x, n, y) evaluated on grid (double).
    k_x1 : array
        κ(x, n, 1) evaluated on grid (double).
    n : int
        Polynomial degree (number of nodes - 1).
    eps : float
        Unit roundoff of working precision.

    Returns
    -------
    relative_error : ndarray
        Pointwise relative forward error |p_hat - p_ref| / |p_ref|.
    theoretical_bound : ndarray
        Pointwise first-order bound (3n+4)*k_xy*eps + (3n+2)*k_x1*eps.
    stability_ratio : ndarray
        Pointwise ratio relative_error / theoretical_bound.
    max_ratio : float
        Maximum of stability_ratio (scalar).
    """
    p_ref = np.asarray(p_ref, dtype=np.float64).ravel()
    p_hat = np.asarray(p_hat, dtype=np.float64).ravel()
    k_xy = np.asarray(k_xy, dtype=np.float64).ravel()
    k_x1 = np.asarray(k_x1, dtype=np.float64).ravel()

    tol = 10 * eps  # or something reasonable

    valid = (
        (np.abs(p_ref) > tol) &
        np.isfinite(k_xy) &
        np.isfinite(k_x1)
    )

    relative_error = np.full_like(p_ref, np.nan)
    relative_error[valid] = (
        np.abs(p_hat[valid] - p_ref[valid]) /
        np.abs(p_ref[valid])
    )

    theoretical_bound = np.full_like(p_ref, np.nan)
    theoretical_bound[valid] = (
        (3 * n + 4) * k_xy[valid] * eps +
        (3 * n + 2) * k_x1[valid] * eps
    )

    stability_ratio = np.full_like(p_ref, np.nan)
    good = valid & (theoretical_bound > 0)
    stability_ratio[good] = (
        relative_error[good] / theoretical_bound[good]
    )

    max_ratio = np.nanmax(stability_ratio)

    return (
        relative_error,
        theoretical_bound,
        stability_ratio,
        max_ratio,
    )
