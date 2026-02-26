"""Task 4: f3(x) = ℓ_n(x). Parameter sweep over mesh types and degree; canonical and over-interpolation."""

import os

import numpy as np

from functions import get
from interpolation import meshes
from . import run_experiment


# Degrees: at least two > 20, include n=29 (Higham).
N_VALUES = [21, 29, 35]
MESH_TYPES = ["uniform", "cheb1", "cheb2"]
OVER_INTERP_OFFSETS = [5, 10]  # m = n + 5, n + 10


def run(args):
    make_f3, a_default, b_default = get(4)
    a, b = args.interval if hasattr(args, "interval") and args.interval else (a_default, b_default)
    grid_size = getattr(args, "evaluation_grid_size", 4000)
    precision = getattr(args, "precision", "single")
    output_dir = getattr(args, "output_dir", "output")

    results_canonical = {}
    results_over = {}

    for mesh_type in MESH_TYPES:
        results_canonical[mesh_type] = {}
        results_over[mesh_type] = {}
        for n in N_VALUES:
            n_nodes = n + 1
            x_nodes = meshes.build_mesh(mesh_type, a, b, n_nodes, np.float64)
            f3, _interval, _roots, _denom = make_f3(x_nodes)

            # Canonical: interpolate with same n+1 nodes
            res = run_experiment.run_experiment_with_nodes(
                f3,
                x_nodes,
                degree=n,
                mesh_type=mesh_type,
                label="canonical",
                grid_size=grid_size,
                precision=precision,
            )
            results_canonical[mesh_type][n] = res

            # Over-interpolation: same f3, larger mesh
            results_over[mesh_type][n] = {}
            for m in [n + d for d in OVER_INTERP_OFFSETS]:
                m_nodes = m + 1
                x_large = meshes.build_mesh(mesh_type, a, b, m_nodes, np.float64)
                res_over = run_experiment.run_experiment_with_nodes(
                    f3,
                    x_large,
                    degree=m,
                    mesh_type=mesh_type,
                    label=f"overinterp_{m}",
                    grid_size=grid_size,
                    precision=precision,
                )
                results_over[mesh_type][n][m] = res_over

    _print_table(results_canonical, results_over)
    _print_ratios_table(results_canonical, results_over)
    if getattr(args, "plot", False):
        os.makedirs(output_dir, exist_ok=True)
        _plot_n29_higham(results_canonical, output_dir)
        _plot_relative_error_30pt(results_canonical, output_dir)
    return {"canonical": results_canonical, "over": results_over}


def _print_table(results_canonical, results_over):
    print("\nTask 4 (f3 = ℓ_n): canonical — Lambda_n, H_n, forward_errors, stability_ratio (BF2)")
    print("-" * 90)
    for mesh_type in MESH_TYPES:
        print(f"  {mesh_type}:")
        for n in sorted(results_canonical[mesh_type].keys()):
            r = results_canonical[mesh_type][n]
            fe_bf2 = r["forward_errors"]["BF2"]
            ratio_bf2 = r["stability_ratios"]["BF2"]
            print(
                f"    n={n:2d}  Lambda_n={r['Lambda_n']:.10f}  H_n={r['H_n']:.10f}  "
                f"fe_BF2={fe_bf2:.10f}  ratio_BF2={ratio_bf2:.10f}"
            )
            for form in ["Newton_inc", "Newton_dec", "Newton_Leja"]:
                fe = r["forward_errors"][form]
                print(f"           fe_{form}={fe:.10f}")
    print("\nTask 4 (f3): over-interpolation (sample m=n+5)")
    print("-" * 60)
    for mesh_type in MESH_TYPES:
        print(f"  {mesh_type}:")
        for n in sorted(results_over[mesh_type].keys()):
            m = n + 5
            if m not in results_over[mesh_type][n]:
                continue
            r = results_over[mesh_type][n][m]
            fe_bf2 = r["forward_errors"]["BF2"]
            print(f"    n={n:2d} m={m:2d}  fe_BF2={fe_bf2:.10f}  Lambda_n={r['Lambda_n']:.10f}")
    print()


def _print_ratios_table(results_canonical, results_over):
    methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
    w = 14
    print("\nTask 4 (f3 = ℓ_n): stability_ratios (within_bound) — canonical")
    print("-" * 120)
    for mesh_type in MESH_TYPES:
        print(f"  {mesh_type}:")
        for n in sorted(results_canonical[mesh_type].keys()):
            r = results_canonical[mesh_type][n]
            parts = [f"ratio_{m}={r['stability_ratios'][m]:>{w}.10f} ({r['within_bound'][m]})" for m in methods]
            print(f"    n={n:2d}  " + "  ".join(parts))
    print("\nTask 4 (f3 = ℓ_n): stability_ratios (within_bound) — over-interpolation (m=n+5)")
    print("-" * 120)
    for mesh_type in MESH_TYPES:
        print(f"  {mesh_type}:")
        for n in sorted(results_over[mesh_type].keys()):
            m = n + 5
            if m not in results_over[mesh_type][n]:
                continue
            r = results_over[mesh_type][n][m]
            parts = [f"ratio_{meth}={r['stability_ratios'][meth]:>{w}.10f} ({r['within_bound'][meth]})" for meth in methods]
            print(f"    n={n:2d} m={m:2d}  " + "  ".join(parts))
    print()


def _plot_n29_higham(results_canonical, output_dir):
    """Higham-style plots for n=29: log10 error and log10 kappa vs x (uniform and cheb1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mesh_type in ["uniform", "cheb1"]:
        if mesh_type not in results_canonical or 29 not in results_canonical[mesh_type]:
            continue
        r = results_canonical[mesh_type][29]
        x_eval = r["x_eval"]
        kappa_1 = r["kappa_1"]
        kappa_xy = r["kappa_xy"]
        # Avoid log(0); use small floor
        eps = 1e-20
        log_k1 = np.log10(np.maximum(np.abs(kappa_1), eps))
        log_kxy = np.log10(np.maximum(np.abs(kappa_xy), eps))

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        axes[0].plot(x_eval, log_k1, label=r"$\log_{10} \kappa(x,n,1)$")
        axes[0].plot(x_eval, log_kxy, label=r"$\log_{10} \kappa(x,n,y)$")
        axes[0].set_ylabel(r"$\log_{10}$ condition")
        axes[0].legend()
        axes[0].grid(True, which="both", linestyle="--", alpha=0.7)
        axes[0].set_title(f"Task 4 (f3 = ℓ_n), n=29, {mesh_type}")

        err_bf2 = r["forward_error_vectors"]["BF2"]
        log_err = np.log10(np.maximum(err_bf2, eps))
        axes[1].plot(x_eval, log_err, label="BF2")
        for name in ["Newton_inc", "Newton_dec", "Newton_Leja"]:
            ev = r["forward_error_vectors"][name]
            axes[1].plot(x_eval, np.log10(np.maximum(ev, eps)), label=name)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel(r"$\log_{10}$ |p_single - p_exact|")
        axes[1].legend()
        axes[1].grid(True, which="both", linestyle="--", alpha=0.7)
        plt.tight_layout()
        path = os.path.join(output_dir, f"task4_{mesh_type}_n29_higham.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved plot to {path}")


def _plot_relative_error_30pt(results_canonical, output_dir):
    """Relative error vs x (Higham-style) for 30 points: uniform and Chebyshev first kind."""
    from utils.plotting import plot_relative_error_vs_x

    methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
    for mesh_type in ["uniform", "cheb1"]:
        if mesh_type not in results_canonical or 29 not in results_canonical[mesh_type]:
            continue
        r = results_canonical[mesh_type][29]
        x_eval = r["x_eval"]
        p_ref = r["p_exact"]
        absolute_errors_by_method = r["forward_error_vectors"]
        path = os.path.join(output_dir, f"task4_relative_error_30pt_{mesh_type}.png")
        title = f"Relative error in p_n(x), 30 nodes, {mesh_type} (f3 = ℓ_n)"
        plot_relative_error_vs_x(
            x_eval, p_ref, absolute_errors_by_method, methods, path, title=title
        )
        print(f"Saved plot to {path}")
