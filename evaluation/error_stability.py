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
