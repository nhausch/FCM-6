"""Task 3: f2(x; d) = product_{i=1}^d (x - i). Parameter sweep over mesh types and n."""

import os

import numpy as np

from functions.f2 import make_f2
from . import run_experiment


def run(args):
    d = getattr(args, "f2_d", 9)
    func, _, interval, _ = make_f2(d)
    a_default, b_default = interval[0], interval[1]
    a, b = args.interval if hasattr(args, "interval") and args.interval else (a_default, b_default)
    degree_min = getattr(args, "degree_min", 5)
    degree_max = getattr(args, "degree_max", 30)
    n_list = list(range(degree_min, degree_max + 1))
    config = {
        "mesh_types": ["uniform", "cheb1", "cheb2"],
        "degree_range": n_list,
        "interval": (a, b),
        "evaluation_grid_size": getattr(args, "evaluation_grid_size", 100),
        "precision": getattr(args, "precision", "single"),
    }
    results = run_experiment.run_task_sweep(config, func, (a, b))
    _print_table(results)
    _print_bf2_bound_table(results)
    _print_newton_max_dd_table(results)

    if getattr(args, "plot", False):
        output_dir = getattr(args, "output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        try:
            from utils.plotting import plot_forward_error_vs_degree
            path = os.path.join(output_dir, "task3_forward_error.png")
            plot_forward_error_vs_degree(results, list(results.keys()), path)
            print(f"Saved plot to {path}")
            for method in ["Newton_inc", "Newton_dec", "Newton_Leja"]:
                path_newton = os.path.join(output_dir, f"task3_forward_error_{method}.png")
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
                res = run_experiment.run_experiment(
                    func,
                    x_nodes,
                    (a, b),
                    grid_size=config["evaluation_grid_size"],
                    precision=config["precision"],
                )
                path = os.path.join(output_dir, f"task3_relative_error_30pt_{mesh_type}.png")
                plot_relative_error_vs_x(
                    res["x_eval"],
                    res["p_exact"],
                    res["forward_error_vectors"],
                    methods,
                    path,
                    title=f"Relative error in p_n(x), 30 nodes, {mesh_type} (f2)",
                )
                print(f"Saved plot to {path}")
        except Exception as e:
            print(f"Relative-error plot failed: {e}")
    return results


def _print_table(results):
    methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
    print("\nTask 3 (f2): forward_errors / Lambda_n (per method)")
    print("-" * 85)
    for mesh_type in results:
        print(f"  {mesh_type}:")
        for n in sorted(results[mesh_type].keys()):
            r = results[mesh_type][n]
            fe_str = "  ".join(f"fe_{m}={r['forward_errors'][m]:.10f}" for m in methods)
            print(f"    n={n:3d}  Lambda_n={r['Lambda_n']:.10f}  {fe_str}")
    print()


def _print_bf2_bound_table(results):
    w = 14
    print("\nTask 3 (f2): BF2 forward error bound")
    print("-" * 100)
    for mesh_type in results:
        print(f"  {mesh_type}:")
        for n in sorted(results[mesh_type].keys()):
            r = results[mesh_type][n]
            bf2 = r["bf2_forward_bound"]
            bound_max = np.max(np.atleast_1d(bf2["theoretical_bound"]))
            rel_max = np.max(np.atleast_1d(bf2["relative_error"]))
            max_ratio = float(bf2["max_ratio"])
            print(f"    n={n:3d}  bf2_bound_max={bound_max:>{w}.10f}  bf2_rel_max={rel_max:>{w}.10f}  bf2_max_ratio={max_ratio:>{w}.10f}")
    print()


def _print_newton_max_dd_table(results):
    newton_orders = ["Newton_inc", "Newton_dec", "Newton_Leja"]
    w = 14
    print("\nTask 3 (f2): Newton divided differences max |coeff|")
    print("-" * 80)
    for mesh_type in results:
        print(f"  {mesh_type}:")
        for n in sorted(results[mesh_type].keys()):
            r = results[mesh_type][n]
            parts = [f"max_dd_{m}={r['newton_max_dd'][m]:>{w}.10f}" for m in newton_orders]
            print(f"    n={n:3d}  " + "  ".join(parts))
    print()
