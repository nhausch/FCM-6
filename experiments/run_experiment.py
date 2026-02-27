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


def compute_exact_interpolant(f, x_nodes, x_eval):
    """Compute exact interpolant p_n^exact(x) in double precision (BF1)."""
    dtype = np.float64
    x_nodes = np.asarray(x_nodes, dtype=dtype).ravel()
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    f_values = np.asarray(f(x_nodes), dtype=dtype).ravel()
    gamma = barycentric_form1.compute_gamma(x_nodes, dtype)
    p_exact = barycentric_form1.barycentric1_eval(
        x_eval, x_nodes, gamma, f_values, dtype
    )
    return p_exact, f_values, gamma


def compute_single_precision_interpolants(f_values_double, x_nodes, x_eval, precision):
    """Build single-precision approximations (BF2 + Newton orderings)."""
    dtype = get_dtype(precision)
    x_nodes_sp = np.asarray(x_nodes, dtype=np.float64).ravel().astype(dtype)
    x_eval_sp = np.asarray(x_eval, dtype=np.float64).ravel().astype(dtype)
    f_values_sp = np.asarray(f_values_double, dtype=dtype).ravel()
    beta, y_sp = barycentric_form2.setup_barycentric2_from_values(
        x_nodes_sp, f_values_sp, dtype
    )
    p_bf2 = barycentric_form2.barycentric2_eval(
        x_eval_sp, x_nodes_sp, beta, y_sp, dtype
    )
    newton_interpolants, newton_max_dd = build_newton_interpolants(
        x_nodes_sp, f_values_sp, x_eval_sp, dtype
    )
    return p_bf2, newton_interpolants, newton_max_dd


def compute_forward_errors(p_exact, approximations):
    """Forward error ||p_hat - p_exact||_infty per method and pointwise vectors."""
    p_exact = np.asarray(p_exact, dtype=np.float64).ravel()
    forward_errors = {}
    forward_error_vectors = {}
    for name, p_hat in approximations.items():
        p_hat = np.asarray(p_hat, dtype=np.float64).ravel()
        err_vec = np.abs(p_hat - p_exact)
        forward_errors[name] = float(np.max(err_vec))
        forward_error_vectors[name] = err_vec
    return forward_errors, forward_error_vectors


def verify_bf2_stability(p_bf2, p_exact, kappa_xy, kappa_1, n, eps):
    """BF2 Higham forward error bound only. Returns dict with same keys as before."""
    bf2_rel, bf2_bound_pt, bf2_ratio_pt, bf2_max_ratio = (
        error_stability.verify_barycentric2_forward_bound(
            p_bf2, p_exact, kappa_xy, kappa_1, n, eps
        )
    )
    return {
        "relative_error": bf2_rel,
        "theoretical_bound": bf2_bound_pt,
        "stability_ratio": bf2_ratio_pt,
        "max_ratio": bf2_max_ratio,
    }


def run_experiment(f, x_nodes, interval, grid_size=100, precision="single"):
    """
    Run a single interpolation experiment. Reference is always the exact
    interpolating polynomial in double precision (BF1). Conditioning,
    forward error (single vs exact interpolant), and stability (BF2 Higham
    only; Newton max DD only) are computed in order.
    """
    dtype_ref = np.float64
    x_nodes = np.asarray(x_nodes, dtype=dtype_ref).ravel()
    a, b = float(interval[0]), float(interval[1])
    x_eval = np.linspace(a, b, grid_size, dtype=dtype_ref)

    p_exact, f_values_double, gamma = compute_exact_interpolant(f, x_nodes, x_eval)

    kappa_xy = condition_numbers.kappa_xy(
        x_eval, x_nodes, gamma, f_values_double, dtype_ref
    )
    kappa_1 = condition_numbers.kappa_x1(x_eval, x_nodes, gamma, dtype_ref)
    Lambda_n = statistics.lebesgue_constant(kappa_1)
    H_n = statistics.Hn(kappa_xy)

    p_bf2, newton_interpolants, newton_max_dd = compute_single_precision_interpolants(
        f_values_double, x_nodes, x_eval, precision
    )
    approximations = {"BF2": p_bf2, **newton_interpolants}

    forward_errors, forward_error_vectors = compute_forward_errors(
        p_exact, approximations
    )

    eps = np.finfo(get_dtype(precision)).eps
    n_degree = x_nodes.size - 1
    bf2_forward_bound = verify_bf2_stability(
        p_bf2, p_exact, kappa_xy, kappa_1, n_degree, eps
    )

    # Approximation error (convergence to true f): |f(x) - p_exact(x)|_infty
    true_values = np.asarray(f(x_eval), dtype=np.float64).ravel()
    approx_error = float(np.max(np.abs(true_values - p_exact)))

    return {
        "Lambda_n": Lambda_n,
        "H_n": H_n,
        "forward_errors": forward_errors,
        "forward_error_vectors": forward_error_vectors,
        "bf2_forward_bound": bf2_forward_bound,
        "newton_max_dd": newton_max_dd,
        "kappa_xy": kappa_xy,
        "kappa_1": kappa_1,
        "p_exact": p_exact,
        "x_eval": x_eval,
        "approx_error": approx_error,
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
                f, x_nodes, (a, b), grid_size=grid_size, precision=precision
            )
    return results
