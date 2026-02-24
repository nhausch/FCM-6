"""Task 2: f1(x) = (x-2)^9. Parameter sweep over mesh types and n."""

import os

from functions import f1
from . import run_experiment


def run(args):
    a, b = args.interval if hasattr(args, "interval") and args.interval else f1.interval
    degree_min = getattr(args, "degree_min", 5)
    degree_max = getattr(args, "degree_max", 30)
    n_list = list(range(degree_min, degree_max + 1))
    config = {
        "mesh_types": ["uniform", "cheb1", "cheb2"],
        "degree_range": n_list,
        "interval": (a, b),
        "evaluation_grid_size": getattr(args, "evaluation_grid_size", 2000),
        "precision": getattr(args, "precision", "single"),
    }
    results = run_experiment.run_task_sweep(config, f1.func, (a, b))
    _print_table(results)
    if getattr(args, "plot", False):
        output_dir = getattr(args, "output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        try:
            from utils.plotting import plot_forward_error_vs_degree
            path = os.path.join(output_dir, "task2_forward_error.png")
            plot_forward_error_vs_degree(results, list(results.keys()), path)
            print(f"Saved plot to {path}")
            for method in ["Newton_inc", "Newton_dec", "Newton_Leja"]:
                path_newton = os.path.join(output_dir, f"task2_forward_error_{method}.png")
                plot_forward_error_vs_degree(results, list(results.keys()), path_newton, method=method)
                print(f"Saved plot to {path_newton}")
        except Exception as e:
            print(f"Plotting failed: {e}")
        try:
            import numpy as np
            from interpolation import meshes
            from utils.plotting import plot_relative_error_vs_x
            n_plot = 30
            methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
            for mesh_type in ["uniform", "cheb1"]:
                x_nodes = meshes.build_mesh(mesh_type, a, b, n_plot, np.float64)
                res = run_experiment.run_experiment_with_nodes(
                    f1.func,
                    x_nodes,
                    degree=n_plot - 1,
                    mesh_type=mesh_type,
                    grid_size=config["evaluation_grid_size"],
                    precision=config["precision"],
                )
                path = os.path.join(output_dir, f"task2_relative_error_30pt_{mesh_type}.png")
                plot_relative_error_vs_x(
                    res["x_eval"],
                    res["p_exact"],
                    res["forward_error_vectors"],
                    methods,
                    path,
                    title=f"Relative error in p_n(x), 30 nodes, {mesh_type} (f1)",
                )
                print(f"Saved plot to {path}")
        except Exception as e:
            print(f"Relative-error plot failed: {e}")
    return results

def _print_table(results):
    methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
    print("\nTask 2 (f1): forward_errors / Lambda_n (per method)")
    print("-" * 85)
    for mesh_type in results:
        print(f"  {mesh_type}:")
        for n in sorted(results[mesh_type].keys()):
            r = results[mesh_type][n]
            fe_str = "  ".join(f"fe_{m}={r['forward_errors'][m]:.4e}" for m in methods)
            print(f"    n={n:3d}  Lambda_n={r['Lambda_n']:.4e}  {fe_str}")
    print()
