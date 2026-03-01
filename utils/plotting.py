"""Plotting helpers: forward error vs degree, convergence (error vs n)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_forward_error_vs_degree(results, mesh_types, path=None, method="BF2", ax=None):
    """
    Plot max forward error vs n for each mesh type.
    results[mesh_type][n] may have "forward_error_sup" (legacy) or "forward_errors" (dict per method).
    If forward_errors is present, method selects which to plot (default "BF2").
    If ax is provided, plot on it and do not save/close; otherwise create figure, save to path, close.
    """
    if ax is None and path is None:
        raise ValueError("path is required when ax is None")
    if ax is None:
        plt.figure()
        target = plt.gca()
    else:
        target = ax
    for mesh_type in mesh_types:
        if mesh_type not in results:
            continue
        n_vals = sorted(results[mesh_type].keys())
        r0 = results[mesh_type][n_vals[0]] if n_vals else {}
        if "forward_errors" in r0:
            err_vals = [results[mesh_type][n]["forward_errors"].get(method, results[mesh_type][n]["forward_errors"]["BF2"]) for n in n_vals]
            target.semilogy(n_vals, err_vals, "o-", label=mesh_type)
        else:
            err_vals = [results[mesh_type][n]["forward_error_sup"] for n in n_vals]
            target.semilogy(n_vals, err_vals, "o-", label=mesh_type)
    target.set_xlabel("n")
    target.set_ylabel("max forward_error")
    target.legend()
    target.grid(True, which="both", linestyle="--", alpha=0.7)
    if ax is None:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


def plot_relative_error_vs_x(x_eval, p_ref, absolute_errors_by_method, methods, path=None, title=None, ax=None):
    """
    Plot pointwise relative error vs x (log scale), Higham-style.
    Relative error = |p_computed - p_ref| / max(|p_ref|, 1e-16).
    absolute_errors_by_method: dict method_name -> absolute error array (same length as x_eval).
    If ax is provided, plot on it and do not save/close; path is only used when ax is None.
    """
    if ax is None and path is None:
        raise ValueError("path is required when ax is None")
    import numpy as np
    p_ref = np.asarray(p_ref).ravel()
    tiny = 1e-16
    floor_log = 1e-20
    denom = np.maximum(np.abs(p_ref), tiny)
    if ax is None:
        plt.figure(figsize=(8, 5))
        target = plt.gca()
    else:
        target = ax
    for method in methods:
        if method not in absolute_errors_by_method:
            continue
        abs_err = np.asarray(absolute_errors_by_method[method]).ravel()
        rel_err = np.maximum(abs_err / denom, floor_log)
        target.semilogy(x_eval, rel_err, label=method, alpha=0.8)
    target.set_xlabel("x")
    target.set_ylabel("Relative error")
    # if title:
    #     target.set_title(title)
    target.legend()
    target.grid(True, which="both", linestyle="--", alpha=0.7)
    if ax is None:
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
    plt.xlabel("n")
    plt.ylabel("sup_norm(p_n - f)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_lambda_vs_n(results, mesh_types, path=None, title=None, ax=None):
    """Plot Lebesgue constant Lambda_n vs n for each mesh type (semilog y). If ax given, plot on it only."""
    if ax is None and path is None:
        raise ValueError("path is required when ax is None")
    if ax is None:
        plt.figure()
        target = plt.gca()
    else:
        target = ax
    for mesh_type in mesh_types:
        if mesh_type not in results:
            continue
        n_vals = sorted(results[mesh_type].keys())
        lambda_vals = [results[mesh_type][n]["Lambda_n"] for n in n_vals]
        target.semilogy(n_vals, lambda_vals, "o-", label=mesh_type)
    target.set_xlabel("n")
    target.set_ylabel("Lambda_n")
    # if title:
    #     target.set_title(title)
    target.legend()
    target.grid(True, which="both", linestyle="--", alpha=0.7)
    if ax is None:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


def plot_Hn_vs_n(results, mesh_types, path=None, title=None, ax=None):
    """Plot H_n vs n for each mesh type (semilog y). If ax given, plot on it only."""
    if ax is None and path is None:
        raise ValueError("path is required when ax is None")
    if ax is None:
        plt.figure()
        target = plt.gca()
    else:
        target = ax
    for mesh_type in mesh_types:
        if mesh_type not in results:
            continue
        n_vals = sorted(results[mesh_type].keys())
        h_vals = [results[mesh_type][n]["H_n"] for n in n_vals]
        target.semilogy(n_vals, h_vals, "o-", label=mesh_type)
    target.set_xlabel("n")
    target.set_ylabel("H_n")
    # if title:
    #     target.set_title(title)
    target.legend()
    target.grid(True, which="both", linestyle="--", alpha=0.7)
    if ax is None:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


def plot_interpolant_vs_function(x_eval, f_ref, curves, path, title=None):
    """Plot true function and interpolants p_n(x) vs x (linear scale). curves: dict label -> 1D array."""
    import numpy as np
    x_eval = np.asarray(x_eval).ravel()
    plt.figure(figsize=(8, 5))
    for label, y_vals in curves.items():
        y_vals = np.asarray(y_vals).ravel()
        plt.plot(x_eval, y_vals, label=label, alpha=0.8)
    plt.xlabel("x")
    plt.ylabel("value")
    # if title:
    #     plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


_METHOD_LABELS = {
    "BF2": "Barycentric 2",
    "Newton_inc": "Newton Increasing",
    "Newton_dec": "Newton Decreasing",
    "Newton_Leja": "Newton Leja",
}


def plot_forward_error_2x2(results, mesh_types, path, title=None):
    """One figure with 2x2 subplots: forward error vs n for BF2, Newton_inc, Newton_dec, Newton_Leja."""
    methods = ["BF2", "Newton_inc", "Newton_dec", "Newton_Leja"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, method in enumerate(methods):
        ax = axes.flat[idx]
        plot_forward_error_vs_degree(results, mesh_types, path=None, method=method, ax=ax)
        ax.set_title(_METHOD_LABELS.get(method, method))
    # if title:
    #     fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_lambda_and_Hn_vs_n(results, mesh_types, path, title=None):
    """One figure with 2 rows: Lambda_n vs n (top), H_n vs n (bottom)."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    plot_lambda_vs_n(results, mesh_types, path=None, title=None, ax=axes[0])
    axes[0].set_xlabel("")
    plot_Hn_vs_n(results, mesh_types, path=None, title=None, ax=axes[1])
    # if title:
    #     fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_relative_error_30pt_grid(entries, methods, path, title=None):
    """One figure with 2 rows: relative error vs x for each entry. entries: list of (x_eval, p_ref, absolute_errors_by_method, subplot_title)."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    for ax, (x_eval, p_ref, errs, stitle) in zip(axes, entries):
        plot_relative_error_vs_x(x_eval, p_ref, errs, methods, path=None, title=stitle, ax=ax)
    axes[0].set_xlabel("")
    # if title:
    #     fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
