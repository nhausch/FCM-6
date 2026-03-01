"""Task 4: f3(x) = ℓ_n(x). Parameter sweep over mesh types and interpolant degree (mirrors Task 3)."""

import os

import numpy as np

from functions import get
from interpolation import meshes
from . import run_experiment


MESH_TYPES = ["uniform", "cheb1", "cheb2"]


def run(args):
    make_f3, a_default, b_default = get(4)
    a, b = args.interval if hasattr(args, "interval") and args.interval else (a_default, b_default)
    n_f3 = getattr(args, "f3_degree", 29)
    degree_min = getattr(args, "degree_min", 5)
    degree_max = getattr(args, "degree_max", 30)
    n_list = list(range(degree_min, degree_max + 1))
    grid_size = getattr(args, "evaluation_grid_size", 100)
    precision = getattr(args, "precision", "single")
    output_dir = getattr(args, "output_dir", "output")

    results = {}
    for mesh_type in MESH_TYPES:
        results[mesh_type] = {}
        x_nodes_f3 = meshes.build_mesh(mesh_type, a, b, n_f3 + 1, np.float64)
        f3, _interval, _roots, _denom = make_f3(x_nodes_f3)
        for n in n_list:
            x_nodes = meshes.build_mesh(mesh_type, a, b, n, np.float64)
            res = run_experiment.run_experiment(
                f3, x_nodes, (a, b), grid_size=grid_size, precision=precision
            )
            results[mesh_type][n] = res

    _print_table(results)
    _print_bf2_bound_table(results)
    _print_newton_max_dd_table(results)

    if getattr(args, "plot", False):
        os.makedirs(output_dir, exist_ok=True)
        config = {
            "evaluation_grid_size": grid_size,
            "precision": precision,
        }
        try:
            from utils.plotting import (
                plot_forward_error_2x2,
                plot_lambda_and_Hn_vs_n,
                plot_relative_error_30pt_grid,
            )
            mesh_list = list(results.keys())
            plot_lambda_and_Hn_vs_n(results, mesh_list, os.path.join(output_dir, "task4_lambda_n_Hn.png"))
            print(f"Saved plot to {os.path.join(output_dir, 'task4_lambda_n_Hn.png')}")
            path_fe = os.path.join(output_dir, "task4_forward_error.png")
            plot_forward_error_2x2(results, mesh_list, path_fe)
            print(f"Saved plot to {path_fe}")
        except Exception as e:
            print(f"Plotting failed: {e}")
        try:
            from utils.plotting import plot_relative_error_30pt_grid
            n_plot = 30
            methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
            entries = []
            for mesh_type in ["uniform", "cheb1"]:
                x_nodes_f3 = meshes.build_mesh(mesh_type, a, b, n_f3 + 1, np.float64)
                f3, _, _, _ = make_f3(x_nodes_f3)
                x_nodes = meshes.build_mesh(mesh_type, a, b, n_plot, np.float64)
                res = run_experiment.run_experiment(
                    f3, x_nodes, (a, b), grid_size=config["evaluation_grid_size"], precision=config["precision"]
                )
                entries.append((
                    res["x_eval"],
                    res["p_exact"],
                    res["forward_error_vectors"],
                    f"Relative error in p_n(x), 30 nodes, {mesh_type} (f3 = ℓ_n)",
                ))
            path_rel = os.path.join(output_dir, "task4_relative_error_30pt.png")
            plot_relative_error_30pt_grid(entries, methods, path_rel)
            print(f"Saved plot to {path_rel}")
        except Exception as e:
            print(f"Relative-error plot failed: {e}")
    return results


def _print_table(results):
    methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
    print("\nTask 4 (f3 = ℓ_n): forward_errors / Lambda_n, H_n (per method)")
    print("-" * 85)
    for mesh_type in results:
        print(f"  {mesh_type}:")
        for n in sorted(results[mesh_type].keys()):
            r = results[mesh_type][n]
            fe_str = "  ".join(f"fe_{m}={r['forward_errors'][m]:.10f}" for m in methods)
            print(f"    n={n:3d}  Lambda_n={r['Lambda_n']:.10f}  H_n={r['H_n']:.10f}  {fe_str}")
    print()


def _print_bf2_bound_table(results):
    w = 14
    print("\nTask 4 (f3 = ℓ_n): BF2 forward error bound")
    print("-" * 120)
    for mesh_type in results:
        print(f"  {mesh_type}:")
        for n in sorted(results[mesh_type].keys()):
            r = results[mesh_type][n]
            bf2 = r["bf2_forward_bound"]
            bound_max = np.max(np.atleast_1d(bf2["theoretical_bound"]))
            rel_max = np.max(np.atleast_1d(bf2["relative_error"]))
            max_ratio = float(bf2["max_ratio"])
            fe_bf2 = r["forward_errors"]["BF2"]
            within_bound = fe_bf2 <= bound_max
            print(f"    n={n:3d}  bf2_bound_max={bound_max:>{w}.10f}  bf2_rel_max={rel_max:>{w}.10f}  bf2_max_ratio={max_ratio:>{w}.10f}  within_bf2_bound={within_bound}")
    print()


def _print_newton_max_dd_table(results):
    newton_orders = ["Newton_inc", "Newton_dec", "Newton_Leja"]
    w = 14
    print("\nTask 4 (f3 = ℓ_n): Newton divided differences max |coeff|")
    print("-" * 80)
    for mesh_type in results:
        print(f"  {mesh_type}:")
        for n in sorted(results[mesh_type].keys()):
            r = results[mesh_type][n]
            parts = [f"max_dd_{m}={r['newton_max_dd'][m]:>{w}.10f}" for m in newton_orders]
            print(f"    n={n:3d}  " + "  ".join(parts))
    print()
