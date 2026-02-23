"""
Single-experiment runner: one (mesh_type, n, f, interval, grid_size, precision).
Returns dict with kappa_xy_max, Lambda_n, forward_error_sup, stability_ratio, within_bound, etc.
Reference and condition numbers in double; approximate interpolant in single when precision is single.
"""

import numpy as np

from interpolation import barycentric_form1, barycentric_form2, meshes
from conditioning import condition_numbers, statistics
from evaluation import error_stability
from utils.precision import get_dtype


def run_experiment(mesh_type, n, f, a, b, grid_size, precision="single"):
    dtype_ref = np.float64
    dtype_approx = get_dtype(precision)

    # Evaluation grid (same for ref and approx; use ref for condition numbers).
    x_grid = np.linspace(a, b, grid_size, dtype=dtype_ref)

    # 1. Build mesh and setup Form 2 (for nodes); Form 1 for ref and condition numbers.
    x_nodes = meshes.build_mesh(mesh_type, a, b, n, dtype_ref)
    x_nodes, beta, y_b2 = barycentric_form2.setup_barycentric2(x_nodes, f, dtype_ref)
    gamma_ref, y_ref = barycentric_form1.setup_barycentric1(
        x_nodes, f, dtype_ref
    )
    p_ref = barycentric_form1.barycentric1_eval(
        x_grid, x_nodes, gamma_ref, y_ref, dtype_ref
    )

    # 2. Approximate interpolant in single (or double if precision is double).
    x_nodes_a = meshes.build_mesh(mesh_type, a, b, n, dtype_approx)
    x_nodes_a, beta_a, y_a = barycentric_form2.setup_barycentric2(x_nodes_a, f, dtype_approx)
    p_approx = barycentric_form2.barycentric2_eval(
        x_grid, x_nodes_a, beta_a, y_a, dtype_approx
    )
    p_ref_same = np.asarray(p_ref, dtype=dtype_approx)

    # 3. Condition numbers (double).
    k_xy = condition_numbers.kappa_xy(x_grid, x_nodes, gamma_ref, y_ref, dtype_ref)
    k_x1 = condition_numbers.kappa_x1(x_grid, x_nodes, gamma_ref, dtype_ref)
    Lambda_n = statistics.lebesgue_constant(k_x1)
    Hn_val = statistics.Hn(k_xy)

    # 4. Forward error and bound comparison (use eps of approximate precision).
    fe_sup = error_stability.forward_error_sup(p_approx, p_ref_same)
    eps = np.finfo(dtype_approx).eps
    cmp_res = error_stability.compare_to_bound(
        fe_sup, Lambda_n, eps, y_ref=y_ref
    )
    ratio = error_stability.stability_ratio(
        fe_sup, Lambda_n, eps, y_ref=y_ref
    )

    return {
        "kappa_xy_max": Hn_val,
        "Lambda_n": Lambda_n,
        "forward_error_sup": fe_sup,
        "stability_ratio": ratio,
        "within_bound": cmp_res["within_bound"],
        "bound": cmp_res["bound"],
    }

# Run the double loop over mesh_types and degree_range (n list). Return nested results.
# config: dict with mesh_types, degree_range (list of n), interval (a,b), evaluation_grid_size, precision.
def run_task_f(config, f, interval, mesh_types=None):
    a, b = interval
    mesh_types = mesh_types or config.get("mesh_types", ["uniform", "cheb1", "cheb2"])
    n_list = config["degree_range"]
    grid_size = config.get("evaluation_grid_size", 2000)
    precision = config.get("precision", "single")

    results = {}
    for mesh_type in mesh_types:
        results[mesh_type] = {}
        for n in n_list:
            results[mesh_type][n] = run_experiment(
                mesh_type, n, f, a, b, grid_size, precision
            )
    return results
