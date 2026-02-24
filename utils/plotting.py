"""Plotting helpers: forward error vs degree, convergence (error vs n)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_forward_error_vs_degree(results, mesh_types, path, method="BF2"):
    """
    Plot max forward error vs n for each mesh type.
    results[mesh_type][n] may have "forward_error_sup" (legacy) or "forward_errors" (dict per method).
    If forward_errors is present, method selects which to plot (default "BF2").
    """
    plt.figure()
    for mesh_type in mesh_types:
        if mesh_type not in results:
            continue
        n_vals = sorted(results[mesh_type].keys())
        r0 = results[mesh_type][n_vals[0]] if n_vals else {}
        if "forward_errors" in r0:
            err_vals = [results[mesh_type][n]["forward_errors"].get(method, results[mesh_type][n]["forward_errors"]["BF2"]) for n in n_vals]
            plt.semilogy(n_vals, err_vals, "o-", label=mesh_type)
        else:
            err_vals = [results[mesh_type][n]["forward_error_sup"] for n in n_vals]
            plt.semilogy(n_vals, err_vals, "o-", label=mesh_type)
    plt.xlabel("n (number of nodes)")
    plt.ylabel("forward_error")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_convergence(n_list, errors_per_mesh, path):
    """
    Plot error vs n for each mesh type (Task 5). errors_per_mesh[mesh_type] = (n_list, error_list).
    """
    plt.figure()
    for mesh_type, (ns, errs) in errors_per_mesh.items():
        plt.semilogy(ns, errs, "o-", label=mesh_type, markersize=3)
    plt.xlabel("n (number of nodes)")
    plt.ylabel("sup_norm(p_n - f)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
