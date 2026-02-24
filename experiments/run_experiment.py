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
    for order_name, order_mode in [
        ("Newton_inc", "increasing"),
        ("Newton_dec", "decreasing"),
        ("Newton_Leja", "leja"),
    ]:
        idx = ordering.order_mesh_indices(x_nodes, order_mode)
        x_ord = x_nodes[idx]
        y_ord = y_values[idx]
        coeffs = newton.divided_differences_from_values(x_ord, y_ord, dtype)
        p_newt = newton.newton_eval(x_eval, x_ord, coeffs, dtype)
        out[order_name] = p_newt
    return out

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

# Runs a single experiment for a given mesh type / mesh size.
# Tasks 2 and 3.
def run_experiment_with_parameters(mesh_type, n, f, a, b, grid_size, precision="single"):
    dtype_ref = np.float64
    dtype_approx = get_dtype(precision)

    # Evaluation grid: ref in double (for condition numbers); approx in chosen precision.
    x_grid = np.linspace(a, b, grid_size, dtype=dtype_ref)
    x_grid_approx = np.linspace(a, b, grid_size, dtype=dtype_approx)

    # Build the reference mesh and setup Form 1 for condition numbers (double precision).
    x_nodes = meshes.build_mesh(mesh_type, a, b, n, dtype_ref)
    gamma_ref, y_ref = barycentric_form1.setup_barycentric1(
        x_nodes, f, dtype_ref
    )
    p_ref = barycentric_form1.barycentric1_eval(
        x_grid, x_nodes, gamma_ref, y_ref, dtype_ref
    )

    # Compute the condition numbers (double precision).
    k_xy = condition_numbers.kappa_xy(x_grid, x_nodes, gamma_ref, y_ref, dtype_ref)
    k_x1 = condition_numbers.kappa_x1(x_grid, x_nodes, gamma_ref, dtype_ref)
    Lambda_n = statistics.lebesgue_constant(k_x1)
    Hn_val = statistics.Hn(k_xy)

    # Build interpolants: BF2 + Newton (three orderings).
    x_nodes_a = meshes.build_mesh(mesh_type, a, b, n, dtype_approx)
    beta_a, y_a = barycentric_form2.setup_barycentric2(x_nodes_a, f, dtype_approx)
    p_bf2 = barycentric_form2.barycentric2_eval(
        x_grid_approx, x_nodes_a, beta_a, y_a, dtype_approx
    )
    y_ref_approx = np.asarray(y_ref, dtype=dtype_approx)
    newton_interpolants = build_newton_interpolants(
        x_nodes, y_ref_approx, x_grid_approx, dtype_approx
    )
    interpolants = {"BF2": p_bf2, **newton_interpolants}

    p_ref_casted = np.asarray(p_ref, dtype=dtype_approx)
    eps = np.finfo(dtype_approx).eps
    forward_errors, stability_ratios, within_bound, bound = compute_forward_errors_and_ratios(
        interpolants, p_ref_casted, Lambda_n, eps, y_ref=y_ref
    )

    return {
        "kappa_xy_max": Hn_val,
        "Lambda_n": Lambda_n,
        "forward_errors": forward_errors,
        "stability_ratios": stability_ratios,
        "within_bound": within_bound,
        "bound": bound,
    }

# Runs the double loop over mesh_types and degree_range (n list). Return nested results.
# Tasks 2 and 3.
def run_task_sweep(config, f, interval, mesh_types=None):
    a, b = interval
    mesh_types = mesh_types or config.get("mesh_types", ["uniform", "cheb1", "cheb2"])
    n_list = config["degree_range"]
    grid_size = config.get("evaluation_grid_size", 2000)
    precision = config.get("precision", "single")

    results = {}
    for mesh_type in mesh_types:
        results[mesh_type] = {}
        for n in n_list:
            results[mesh_type][n] = run_experiment_with_parameters(
                mesh_type, n, f, a, b, grid_size, precision
            )
    return results

# Runs a single interpolation experiment with pre-built nodes.
# Task 4.
def run_experiment_with_nodes(
    f,
    x_nodes,
    degree,
    mesh_type,
    label=None,
    grid_size=4000,
    precision="single",
):
    dtype_ref = np.float64
    dtype_approx = get_dtype(precision)
    x_nodes = np.asarray(x_nodes, dtype=dtype_ref)
    n_nodes = x_nodes.size

    # Prepare the data.
    f_values_double = np.asarray(f(x_nodes), dtype=dtype_ref).ravel()
    f_values_single = f_values_double.astype(np.float32)
    if f_values_double.size != n_nodes:
        raise ValueError("f(x_nodes) must return length equal to x_nodes")

    # Create the grid.
    a, b = float(x_nodes.min()), float(x_nodes.max())
    x_eval = np.linspace(a, b, grid_size, dtype=dtype_ref)
    x_eval_approx = x_eval.astype(dtype_approx)

    # Construct the reference polynomial (double precision).
    p_exact = np.asarray(f(x_eval), dtype=dtype_ref).ravel()

    # Construct the interpolants (single precision): BF2 + Newton (three orderings).
    beta, y_single = barycentric_form2.setup_barycentric2_from_values(
        x_nodes, f_values_single, dtype_approx
    )
    p_bf2 = barycentric_form2.barycentric2_eval(
        x_eval_approx, x_nodes.astype(dtype_approx), beta, y_single, dtype_approx
    )
    newton_interpolants = build_newton_interpolants(
        x_nodes, f_values_single, x_eval_approx, dtype_approx
    )
    interpolants = {"BF2": p_bf2, **newton_interpolants}

    # Conditioning (double).
    gamma = barycentric_form1.compute_gamma(x_nodes, dtype_ref)
    kappa_xy = condition_numbers.kappa_xy(
        x_eval, x_nodes, gamma, f_values_double, dtype_ref
    )
    kappa_1 = condition_numbers.kappa_x1(x_eval, x_nodes, gamma, dtype_ref)
    Lambda_n = statistics.lebesgue_constant(kappa_1)
    H_n = statistics.Hn(kappa_xy)

    # Forward errors and stability ratios (unified bound: Lambda_n * eps * max|y|)
    eps = np.finfo(dtype_approx).eps
    forward_errors, stability_ratios, within_bound, bound = compute_forward_errors_and_ratios(
        interpolants, p_exact, Lambda_n, eps, y_ref=f_values_double
    )
    forward_error_vectors = {
        name: np.abs(np.asarray(p_single, dtype=dtype_ref).ravel() - p_exact)
        for name, p_single in interpolants.items()
    }

    return {
        "Lambda_n": Lambda_n,
        "H_n": H_n,
        "forward_errors": forward_errors,
        "forward_error_vectors": forward_error_vectors,
        "stability_ratios": stability_ratios,
        "within_bound": within_bound,
        "x_eval": x_eval,
        "kappa_1": kappa_1,
        "kappa_xy": kappa_xy,
        "p_exact": p_exact,
        "interpolants": interpolants,
        "label": label,
    }

