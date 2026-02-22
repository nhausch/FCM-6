"""Task 4: f3(x) = exp(x). Parameter sweep over mesh types and n."""

import os

from functions import get
from . import run_experiment


def run(args):
    f, a_default, b_default = get(4)
    a, b = args.interval if hasattr(args, "interval") and args.interval else (a_default, b_default)
    degree_min = getattr(args, "degree_min", 5)
    degree_max = getattr(args, "degree_max", 20)
    n_list = list(range(degree_min, degree_max + 1))
    config = {
        "mesh_types": ["uniform", "cheb1", "cheb2"],
        "degree_range": n_list,
        "interval": (a, b),
        "evaluation_grid_size": getattr(args, "evaluation_grid_size", 2000),
        "precision": getattr(args, "precision", "single"),
    }
    results = run_experiment.run_task_f(config, f, (a, b))
    _print_table(results)
    if getattr(args, "plot", False):
        output_dir = getattr(args, "output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        try:
            from utils.plotting import plot_forward_error_vs_degree
            path = os.path.join(output_dir, "task4_forward_error.png")
            plot_forward_error_vs_degree(results, list(results.keys()), path)
            print(f"Saved plot to {path}")
        except Exception as e:
            print(f"Plotting failed: {e}")
    return results


def _print_table(results):
    print("\nTask 4 (f3): forward_error_sup / Lambda_n")
    print("-" * 60)
    for mesh_type in results:
        print(f"  {mesh_type}:")
        for n in sorted(results[mesh_type].keys()):
            r = results[mesh_type][n]
            print(f"    n={n:3d}  fe_sup={r['forward_error_sup']:.4e}  Lambda_n={r['Lambda_n']:.4f}")
    print()
