"""Task 5: f4(x) = 1/(1+25*x^2) convergence study. Error vs n for each mesh type (double precision)."""

import numpy as np
import os

from functions import f4
from interpolation import barycentric_form2
from conditioning.statistics import sup_norm


def run(args):
    a, b = args.interval if hasattr(args, "interval") and args.interval else f4.interval
    n_min = 5
    n_max = getattr(args, "n_max", 50)
    grid_size = getattr(args, "evaluation_grid_size", 2000)
    x_grid = np.linspace(a, b, grid_size, dtype=np.float64)
    f_ref = f4.func(x_grid)

    mesh_types = ["uniform", "cheb1", "cheb2"]
    errors_per_mesh = {}
    n_list = list(range(n_min, n_max + 1))

    for mesh_type in mesh_types:
        errors = []
        for n in n_list:
            x_nodes, beta, y = barycentric_form2.setup_barycentric2(
                mesh_type, a, b, n, f4.func, np.float64
            )
            p_n = barycentric_form2.barycentric2_eval(
                x_grid, x_nodes, beta, y, np.float64
            )
            err = sup_norm(p_n - f_ref)
            errors.append(err)
        errors_per_mesh[mesh_type] = (n_list, errors)

    _print_table(n_list, errors_per_mesh)
    if getattr(args, "plot", False):
        output_dir = getattr(args, "output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        try:
            from utils.plotting import plot_convergence
            path = os.path.join(output_dir, "task5_convergence.png")
            plot_convergence(n_list, errors_per_mesh, path)
            print(f"Saved plot to {path}")
        except Exception as e:
            print(f"Plotting failed: {e}")
    return errors_per_mesh


def _print_table(n_list, errors_per_mesh):
    print("\nTask 5 (f4 convergence): sup_norm(p_n - f4) vs n")
    print("-" * 60)
    for mesh_type in errors_per_mesh:
        ns, errs = errors_per_mesh[mesh_type]
        print(f"  {mesh_type}: n_min={min(ns)}, n_max={max(ns)}  err_min={min(errs):.4e}  err_max={max(errs):.4e}")
    print()
