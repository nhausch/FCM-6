"""Task 5: f4(x) = 1/(1+25*x^2) convergence study.

Reuses run_experiment: conditioning (Part 1), forward error/stability (Part 2),
approximation error |f4 - p_n|_infty (Part 3, convergence to true function).
"""

import os

import numpy as np

from functions import f4
from . import run_experiment


def run(args):
    a, b = args.interval if hasattr(args, "interval") and args.interval else f4.interval
    n_min = 5
    n_max = getattr(args, "n_max", 50)
    n_list = list(range(n_min, n_max + 1))
    grid_size = getattr(args, "evaluation_grid_size", 1000)
    config = {
        "mesh_types": ["uniform", "cheb1", "cheb2"],
        "degree_range": n_list,
        "evaluation_grid_size": grid_size,
        "precision": "single",
    }
    results = run_experiment.run_task_sweep(config, f4.func, (a, b))

    _print_part1_conditioning(results)
    _print_part2_stability(results)
    _print_part3_convergence(results)

    if getattr(args, "plot", False):
        output_dir = getattr(args, "output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        mesh_list = list(results.keys())
        methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
        try:
            from utils.plotting import (
                plot_convergence,
                plot_lambda_and_Hn_vs_n,
                plot_forward_error_vs_degree,
                plot_relative_error_vs_x,
                plot_interpolant_vs_function,
            )
            # Plot 5 & 6: Lambda_n and H_n vs n (one figure)
            plot_lambda_and_Hn_vs_n(results, mesh_list, os.path.join(output_dir, "task5_lambda_n_Hn.png"))  # title="Task 5 (f4): Lambda_n and H_n vs n"
            print(f"Saved plot to {os.path.join(output_dir, 'task5_lambda_n_Hn.png')}")
            # Plot 7 & 8: Higham-style relative error vs x for uniform and cheb1 at n=20, n=40
            for mesh_type in ["uniform", "cheb1"]:
                if mesh_type not in results:
                    continue
                for n in [20, 40]:
                    if n not in results[mesh_type]:
                        continue
                    r = results[mesh_type][n]
                    path = os.path.join(output_dir, f"task5_relative_error_{mesh_type}_n{n}.png")
                    plot_relative_error_vs_x(
                        r["x_eval"],
                        r["p_exact"],
                        r["forward_error_vectors"],
                        methods,
                        path,
                        # title=f"Task 5 (f4): relative error vs x, {mesh_type}, n={n}",
                    )
                    print(f"Saved plot to {path}")
            # Plot 9: Forward error vs n (BF2)
            plot_forward_error_vs_degree(results, mesh_list, os.path.join(output_dir, "task5_forward_error.png"), method="BF2")
            print(f"Saved plot to {os.path.join(output_dir, 'task5_forward_error.png')}")
            # Plot 10: Convergence (existing)
            n_list_sorted = sorted(results["uniform"].keys())
            errors_per_mesh = {
                mesh_type: (
                    n_list_sorted,
                    [results[mesh_type][n]["approx_error"] for n in n_list_sorted],
                )
                for mesh_type in results
            }
            path = os.path.join(output_dir, "task5_convergence.png")
            plot_convergence(n_list_sorted, errors_per_mesh, path)
            print(f"Saved plot to {path}")
            # Plot 11: p_n(x) vs f4(x) for uniform and cheb1, n=10, 20, 40
            for mesh_type in ["uniform", "cheb1"]:
                if mesh_type not in results:
                    continue
                ns_plot = [10, 20, 40]
                if not all(n in results[mesh_type] for n in ns_plot):
                    continue
                x_eval = results[mesh_type][ns_plot[0]]["x_eval"]
                f_ref = np.asarray(f4.func(x_eval)).ravel()
                curves = {"f4": f_ref}
                for n in ns_plot:
                    curves[f"n={n}"] = results[mesh_type][n]["p_exact"]
                path = os.path.join(output_dir, f"task5_interpolant_vs_f4_{mesh_type}.png")
                plot_interpolant_vs_function(
                    x_eval, f_ref, curves, path,
                    # title=f"Task 5 (f4): p_n(x) vs f4(x), {mesh_type}",
                )
                print(f"Saved plot to {path}")
        except Exception as e:
            print(f"Plotting failed: {e}")

    return results


def _print_part1_conditioning(results):
    """Part 1: Conditioning (Lambda_n, H_n)."""
    print("\nTask 5 (f4) Part 1 — Conditioning: Lambda_n, H_n")
    print("-" * 70)
    for mesh_type in results:
        ns = sorted(results[mesh_type].keys())
        lambdas = [results[mesh_type][n]["Lambda_n"] for n in ns]
        h_ns = [results[mesh_type][n]["H_n"] for n in ns]
        print(f"  {mesh_type}: n=[{min(ns)}, {max(ns)}]  Lambda_n min={min(lambdas):.6f} max={max(lambdas):.6f}  H_n min={min(h_ns):.6f} max={max(h_ns):.6f}")
    print()


def _print_part2_stability(results):
    """Part 2: Forward error (stability), BF2 bound, Newton max divided difference."""
    methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
    w = 12
    print("\nTask 5 (f4) Part 2 — Stability / accuracy: forward_errors, BF2 bound, Newton max_dd")
    print("-" * 90)
    for mesh_type in results:
        print(f"  {mesh_type}:")
        for n in sorted(results[mesh_type].keys())[:5]:  # first 5 n
            r = results[mesh_type][n]
            fe_bf2 = r["forward_errors"]["BF2"]
            bf2 = r["bf2_forward_bound"]
            max_ratio = float(bf2["max_ratio"])
            dd = r["newton_max_dd"]
            fe_str = "  ".join(f"{m}={r['forward_errors'][m]:{w}.6f}" for m in methods)
            print(f"    n={n:3d}  {fe_str}  bf2_max_ratio={max_ratio:{w}.6f}  max_dd_Leja={dd['Newton_Leja']:{w}.6f}")
        if len(results[mesh_type]) > 5:
            print(f"    ... ({len(results[mesh_type])} total n values)")
    print()


def _print_part3_convergence(results):
    """Part 3: Approximation error |f4 - p_n|_infty (convergence to true function)."""
    print("\nTask 5 (f4) Part 3 — Convergence: approx_error = |f4 - p_n|_infty")
    print("-" * 70)
    for mesh_type in results:
        ns = sorted(results[mesh_type].keys())
        errs = [results[mesh_type][n]["approx_error"] for n in ns]
        print(f"  {mesh_type}: n_min={min(ns)}, n_max={max(ns)}  err_min={min(errs):.10f}  err_max={max(errs):.10f}")
    print()
