import numpy as np

from interpolation import barycentric_form1, barycentric_form2, meshes, newton, ordering
from conditioning import condition_numbers, statistics
from evaluation import error_stability
from utils.precision import get_dtype


def build_newton_interpolants(x_nodes, y_values, x_eval, dtype):
    x_nodes = np.asarray(x_nodes, dtype=dtype).ravel()
    y_values = np.asarray(y_values, dtype=dtype).ravel()
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    out = {}
    max_dd = {}
    for order_name, order_mode in [
        ("Newton_inc", "increasing"),
        ("Newton_dec", "decreasing"),
        ("Newton_Leja", "leja"),
    ]:
        idx = ordering.order_mesh_indices(x_nodes, order_mode)
        x_ord = x_nodes[idx]
        y_ord = y_values[idx]
        coeffs = newton.divided_differences_from_values(x_ord, y_ord, dtype)
        max_dd[order_name] = float(np.max(np.abs(coeffs)))
        p_newt = newton.newton_eval(x_eval, x_ord, coeffs, dtype)
        out[order_name] = p_newt
    return out, max_dd

def compute_forward_errors_and_ratios(interpolants, p_ref, Lambda_n, eps, y_ref=None):
    p_ref = np.asarray(p_ref).ravel()
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

    forward_errors = {}
    stability_ratios = {}
    within_bound = {}
    for name, p_single in interpolants.items():
        p_single = np.asarray(p_single).ravel()
        fe = statistics.sup_norm(p_single.astype(np.float64) - p_ref.astype(np.float64))
        forward_errors[name] = fe
        stability_ratios[name] = fe / bound
        within_bound[name] = fe <= bound
    return forward_errors, stability_ratios, within_bound, bound


def run_experiment(
    f,
    x_nodes,
    *,
    grid_size=100,
    precision="single",
    reference="interpolant",
    interval=None,
    mesh_type=None,
    degree=None,
    label=None,
):
    """
    Run a single interpolation experiment with pre-built nodes.

    reference: "interpolant" = compare to BF1 polynomial in double;
               "exact" = compare to f(x_eval) on the grid.
    interval: optional (a, b) for evaluation grid; if None, use x_nodes.min/max.
    """
    dtype_ref = np.float64
    dtype_approx = get_dtype(precision)
    x_nodes = np.asarray(x_nodes, dtype=dtype_ref).ravel()
    n_nodes = x_nodes.size

    f_values_double = np.asarray(f(x_nodes), dtype=dtype_ref).ravel()
    f_values_single = f_values_double.astype(dtype_approx)
    if f_values_double.size != n_nodes:
        raise ValueError("f(x_nodes) must return length equal to x_nodes")

    if interval is not None:
        a, b = float(interval[0]), float(interval[1])
    else:
        a, b = float(x_nodes.min()), float(x_nodes.max())

    x_eval = np.linspace(a, b, grid_size, dtype=dtype_ref)
    x_eval_approx = x_eval.astype(dtype_approx)

    gamma = barycentric_form1.compute_gamma(x_nodes, dtype_ref)
    if reference == "interpolant":
        p_ref = barycentric_form1.barycentric1_eval(
            x_eval, x_nodes, gamma, f_values_double, dtype_ref
        )
    elif reference == "exact":
        p_ref = np.asarray(f(x_eval), dtype=dtype_ref).ravel()
    else:
        raise ValueError(f"reference must be 'interpolant' or 'exact', got {reference!r}")

    beta, y_single = barycentric_form2.setup_barycentric2_from_values(
        x_nodes, f_values_single, dtype_approx
    )
    p_bf2 = barycentric_form2.barycentric2_eval(
        x_eval_approx, x_nodes.astype(dtype_approx), beta, y_single, dtype_approx
    )
    newton_interpolants, newton_max_dd = build_newton_interpolants(
        x_nodes, f_values_single, x_eval_approx, dtype_approx
    )
    interpolants = {"BF2": p_bf2, **newton_interpolants}

    kappa_xy = condition_numbers.kappa_xy(
        x_eval, x_nodes, gamma, f_values_double, dtype_ref
    )
    kappa_1 = condition_numbers.kappa_x1(x_eval, x_nodes, gamma, dtype_ref)
    Lambda_n = statistics.lebesgue_constant(kappa_1)
    H_n = statistics.Hn(kappa_xy)

    eps = np.finfo(dtype_approx).eps
    forward_errors, stability_ratios, within_bound, bound = compute_forward_errors_and_ratios(
        interpolants, p_ref, Lambda_n, eps, y_ref=f_values_double
    )
    forward_error_vectors = {
        name: np.abs(np.asarray(p_single, dtype=dtype_ref).ravel() - p_ref)
        for name, p_single in interpolants.items()
    }

    n_degree = n_nodes - 1
    bf2_rel, bf2_bound_pt, bf2_ratio_pt, bf2_max_ratio = error_stability.verify_barycentric2_forward_bound(
        p_bf2, p_ref, kappa_xy, kappa_1, n_degree, eps
    )

    return {
        "Lambda_n": Lambda_n,
        "H_n": H_n,
        "forward_errors": forward_errors,
        "forward_error_vectors": forward_error_vectors,
        "stability_ratios": stability_ratios,
        "within_bound": within_bound,
        "bound": bound,
        "newton_max_dd": newton_max_dd,
        "x_eval": x_eval,
        "kappa_1": kappa_1,
        "kappa_xy": kappa_xy,
        "p_ref": p_ref,
        "p_exact": p_ref,
        "interpolants": interpolants,
        "label": label,
        "bf2_forward_bound": {
            "relative_error": bf2_rel,
            "theoretical_bound": bf2_bound_pt,
            "stability_ratio": bf2_ratio_pt,
            "max_ratio": bf2_max_ratio,
        },
    }


# Runs the double loop over mesh_types and degree_range (n list). Return nested results.
# Tasks 2 and 3.
def run_task_sweep(config, f, interval, mesh_types=None):
    a, b = interval
    mesh_types = mesh_types or config.get("mesh_types", ["uniform", "cheb1", "cheb2"])
    n_list = config["degree_range"]
    grid_size = config.get("evaluation_grid_size", 100)
    precision = config.get("precision", "single")
    dtype_ref = np.float64

    results = {}
    for mesh_type in mesh_types:
        results[mesh_type] = {}
        for n in n_list:
            x_nodes = meshes.build_mesh(mesh_type, a, b, n, dtype_ref)
            results[mesh_type][n] = run_experiment(
                f,
                x_nodes,
                grid_size=grid_size,
                precision=precision,
                reference="interpolant",
                interval=(a, b),
                mesh_type=mesh_type,
                degree=n - 1,
            )
    return results
