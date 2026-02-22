"""
Phase 1–3 validation (default). Task mode: python main.py --task 2 [--precision single] [--plot] ...
"""

import argparse
import numpy as np

from interpolation import meshes, barycentric_form1, barycentric_form2, newton, ordering
from conditioning import condition_numbers, statistics
from utils.precision import get_dtype, PrecisionContext
from evaluation import error_stability


def exact_polynomial(x):
    """f(x) = (x - 2)^9 (degree 9, exact in double/single)."""
    return (x - 2.0) ** 9


def run_validation(a=1.0, b=3.0, n=10, grid_size=200, dtype=np.float64):
    """
    Build meshes, setup all forms, evaluate on fine grid, compare to f and to each other.
    Returns dict of max errors (vs f and between forms).
    """
    x_grid = np.linspace(a, b, grid_size, dtype=dtype)
    f_ref = exact_polynomial(x_grid)

    # Uniform mesh
    x_unif = meshes.uniform_mesh(a, b, n, dtype)
    gamma, y_b1 = barycentric_form1.setup_barycentric1(x_unif, exact_polynomial, dtype)
    p_b1_unif = barycentric_form1.barycentric1_eval(x_grid, x_unif, gamma, y_b1, dtype)

    x_b2_unif, beta_unif, y_b2_unif = barycentric_form2.setup_barycentric2(
        "uniform", a, b, n, exact_polynomial, dtype
    )
    p_b2_unif = barycentric_form2.barycentric2_eval(
        x_grid, x_b2_unif, beta_unif, y_b2_unif, dtype
    )

    x_newt_inc = ordering.order_mesh(x_unif, "increasing")
    coeffs_inc, _ = newton.divided_differences(x_newt_inc, exact_polynomial, dtype)
    p_newt_inc = newton.newton_eval(x_grid, x_newt_inc, coeffs_inc, dtype)

    x_newt_leja = ordering.order_mesh(x_unif, "leja")
    coeffs_leja, _ = newton.divided_differences(x_newt_leja, exact_polynomial, dtype)
    p_newt_leja = newton.newton_eval(x_grid, x_newt_leja, coeffs_leja, dtype)

    errors = {
        "vs_f": {
            "bary1_unif": np.max(np.abs(p_b1_unif - f_ref)),
            "bary2_unif": np.max(np.abs(p_b2_unif - f_ref)),
            "newton_inc": np.max(np.abs(p_newt_inc - f_ref)),
            "newt_leja": np.max(np.abs(p_newt_leja - f_ref)),
        },
        "cross_form_unif": {
            "b1_vs_b2": np.max(np.abs(p_b1_unif - p_b2_unif)),
            "b1_vs_newt_inc": np.max(np.abs(p_b1_unif - p_newt_inc)),
            "b1_vs_newt_leja": np.max(np.abs(p_b1_unif - p_newt_leja)),
        },
    }

    # Cheb1 mesh (same checks)
    x_cheb1 = meshes.chebyshev_first_kind(a, b, n, dtype)
    gamma_c, y_c = barycentric_form1.setup_barycentric1(
        x_cheb1, exact_polynomial, dtype
    )
    p_b1_cheb1 = barycentric_form1.barycentric1_eval(
        x_grid, x_cheb1, gamma_c, y_c, dtype
    )
    x_b2_c, beta_c, y_b2_c = barycentric_form2.setup_barycentric2(
        "cheb1", a, b, n, exact_polynomial, dtype
    )
    p_b2_cheb1 = barycentric_form2.barycentric2_eval(
        x_grid, x_b2_c, beta_c, y_b2_c, dtype
    )
    errors["vs_f"]["bary1_cheb1"] = np.max(np.abs(p_b1_cheb1 - f_ref))
    errors["vs_f"]["bary2_cheb1"] = np.max(np.abs(p_b2_cheb1 - f_ref))
    errors["cross_form_cheb1"] = {
        "b1_vs_b2": np.max(np.abs(p_b1_cheb1 - p_b2_cheb1)),
    }

    return errors


def main():
    print("Phase 1 validation: f(x) = (x-2)^9, interval [1,3], grid size 200\n")
    for dtype in (np.float64, np.float32):
        name = "float64" if dtype == np.float64 else "float32"
        eps = np.finfo(dtype).eps
        print(f"--- dtype = {name} (eps ≈ {eps:.2e}) ---")
        for n in (3, 5, 10):
            err = run_validation(a=1.0, b=3.0, n=n, grid_size=200, dtype=dtype)
            print(f"  n = {n}:")
            if n >= 10:
                for k, v in err["vs_f"].items():
                    print(f"    max|p - f| {k}: {v:.4e}")
            for k, v in err["cross_form_unif"].items():
                print(f"    max|p1 - p2| {k}: {v:.4e}")
            for k, v in err["cross_form_cheb1"].items():
                print(f"    max|p1 - p2| {k}: {v:.4e}")
            if n < 10:
                u, c = err["cross_form_unif"], err["cross_form_cheb1"]
                ok = u["b1_vs_b2"] < 10 * eps and c["b1_vs_b2"] < 10 * eps
                print(f"    cross-form ok: {ok}")
        print()
    print("Phase 1 validation done.")


def run_phase2_validation(a=1.0, b=3.0, n=5, grid_size=200, dtype=np.float64):
    """
    Phase 2: compute kappa_xy, kappa_x1 on a grid; Lambda_n, Hn; sanity checks.
    """
    x_grid = np.linspace(a, b, grid_size, dtype=dtype)
    x_nodes = meshes.uniform_mesh(a, b, n, dtype)
    gamma, y = barycentric_form1.setup_barycentric1(x_nodes, exact_polynomial, dtype)

    k_xy = condition_numbers.kappa_xy(x_grid, x_nodes, gamma, y, dtype)
    k_x1 = condition_numbers.kappa_x1(x_grid, x_nodes, gamma, dtype)

    Lambda_n = statistics.lebesgue_constant(k_x1)
    Hn_val = statistics.Hn(k_xy)

    # Sanity: kappa_x1 >= 1 (Lebesgue function bounded below by 1)
    k_x1_fin = k_x1[np.isfinite(k_x1)]
    ok_x1_ge_1 = np.all(k_x1_fin >= 1.0 - 1e-10) if k_x1_fin.size else True
    # At nodes, kappa_x1 = 1 (spot-check: eval at nodes)
    at_nodes = condition_numbers.kappa_x1(x_nodes, x_nodes, gamma, dtype)
    ok_nodes_one = np.allclose(at_nodes, 1.0)
    # kappa_xy positive where finite
    k_xy_fin = k_xy[np.isfinite(k_xy)]
    ok_xy_pos = np.all(k_xy_fin >= 0) if k_xy_fin.size else True
    # Lambda_n finite and >= 1
    ok_Lambda = np.isfinite(Lambda_n) and Lambda_n >= 1.0

    return {
        "Lambda_n": Lambda_n,
        "Hn": Hn_val,
        "kappa_x1_min": np.nanmin(k_x1) if np.any(np.isfinite(k_x1)) else np.nan,
        "kappa_x1_at_nodes": at_nodes,
        "ok_x1_ge_1": ok_x1_ge_1,
        "ok_nodes_one": ok_nodes_one,
        "ok_xy_pos": ok_xy_pos,
        "ok_Lambda": ok_Lambda,
    }


def run_phase3_validation(a=1.0, b=3.0, n=8, grid_size=200):
    """
    Phase 3: precision helpers, single vs double forward error, stability ratio, compare_to_bound.
    Reference = double; approximate = single (same mesh, same f).
    """
    # 1. Precision
    assert get_dtype("single") == np.float32
    assert get_dtype("double") == np.float64
    ctx_single = PrecisionContext("single")
    assert ctx_single.dtype == np.float32
    assert np.isfinite(ctx_single.eps) and ctx_single.eps > 0

    # 2. Build reference (double) and approximate (single) interpolants on same mesh
    dtype_ref = np.float64
    dtype_approx = np.float32
    x_grid_ref = np.linspace(a, b, grid_size, dtype=dtype_ref)
    x_grid_approx = np.linspace(a, b, grid_size, dtype=dtype_approx)

    x_nodes_ref = meshes.uniform_mesh(a, b, n, dtype_ref)
    gamma_ref, y_ref = barycentric_form1.setup_barycentric1(
        x_nodes_ref, exact_polynomial, dtype_ref
    )
    p_ref = barycentric_form1.barycentric1_eval(
        x_grid_ref, x_nodes_ref, gamma_ref, y_ref, dtype_ref
    )

    x_nodes_approx = meshes.uniform_mesh(a, b, n, dtype_approx)
    gamma_approx, y_approx = barycentric_form1.setup_barycentric1(
        x_nodes_approx, exact_polynomial, dtype_approx
    )
    p_approx = barycentric_form1.barycentric1_eval(
        x_grid_approx, x_nodes_approx, gamma_approx, y_approx, dtype_approx
    )

    # Forward error: compare on same grid (cast p_ref to float32 for same-length comparison, or compare in float64)
    p_ref_on_grid = np.asarray(p_ref, dtype=dtype_approx)
    fe_sup = error_stability.forward_error_sup(p_approx, p_ref_on_grid)
    assert np.isfinite(fe_sup) and fe_sup >= 0

    # 3. Lambda_n in double (reference precision for condition numbers)
    k_x1 = condition_numbers.kappa_x1(x_grid_ref, x_nodes_ref, gamma_ref, dtype_ref)
    Lambda_n = statistics.lebesgue_constant(k_x1)
    eps_single = np.finfo(np.float32).eps

    # Stability ratio and compare_to_bound (use single eps for "stability experiment")
    ratio = error_stability.stability_ratio(fe_sup, Lambda_n, eps_single, y_ref=y_ref)
    cmp_res = error_stability.compare_to_bound(
        fe_sup, Lambda_n, eps_single, y_ref=y_ref
    )
    assert np.isfinite(ratio) and ratio > 0
    assert cmp_res["forward_error_sup"] == fe_sup
    assert cmp_res["within_bound"] == (fe_sup <= cmp_res["bound"])
    assert abs(cmp_res["ratio"] - ratio) < 1e-10

    return {
        "forward_error_sup": fe_sup,
        "Lambda_n": Lambda_n,
        "stability_ratio": ratio,
        "within_bound": cmp_res["within_bound"],
        "bound": cmp_res["bound"],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Interpolation experiments and validations.")
    p.add_argument("--task", type=int, default=None, metavar="N",
                   help="Run task 2, 3, 4, or 5. If omitted, run Phase 1–3 validations.")
    p.add_argument("--precision", type=str, default="single", choices=["single", "double"],
                   help="Precision for approximate interpolant (default: single).")
    p.add_argument("--plot", action="store_true", help="Produce and save plots.")
    p.add_argument("--interval", type=float, nargs=2, default=None, metavar=("A", "B"),
                   help="Interval [a, b]. Default from function for chosen task.")
    p.add_argument("--degree-min", type=int, default=5, help="Min n for tasks 2–4 (default 5).")
    p.add_argument("--degree-max", type=int, default=20, help="Max n for tasks 2–4 (default 20).")
    p.add_argument("--n-max", type=int, default=50, help="Max n for Task 5 convergence (default 50).")
    p.add_argument("--output-dir", type=str, default="output", help="Directory for plots (default: output).")
    p.add_argument("--evaluation-grid-size", type=int, default=2000, help="Grid size for evaluation (default 2000).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task is None:
        main()
        print("\n--- Phase 2 validation ---")
        for dtype in (np.float64, np.float32):
            name = "float64" if dtype == np.float64 else "float32"
            res = run_phase2_validation(a=1.0, b=3.0, n=5, grid_size=200, dtype=dtype)
            print(f"dtype={name}: Lambda_n={res['Lambda_n']:.6f}, Hn={res['Hn']:.6f}")
            print(f"  kappa_x1 min={res['kappa_x1_min']:.6f}, at nodes ~1: {res['ok_nodes_one']}")
            print(f"  sanity: x1>=1 {res['ok_x1_ge_1']}, xy>=0 {res['ok_xy_pos']}, Lambda ok {res['ok_Lambda']}")
        print("Phase 2 validation done.")
        print("\n--- Phase 3 validation ---")
        res3 = run_phase3_validation(a=1.0, b=3.0, n=8, grid_size=200)
        print(f"forward_error_sup (single vs double): {res3['forward_error_sup']:.4e}")
        print(f"Lambda_n: {res3['Lambda_n']:.6f}, bound: {res3['bound']:.4e}")
        print(f"stability_ratio: {res3['stability_ratio']:.4f}, within_bound: {res3['within_bound']}")
        print("Phase 3 validation done.")
    else:
        if args.task not in (2, 3, 4, 5):
            raise SystemExit("--task must be 2, 3, 4, or 5.")
        if args.interval is not None:
            args.interval = tuple(args.interval)
        executor = __import__("task_executor", fromlist=["TaskExecutor"]).TaskExecutor()
        executor.run(args.task, args)
