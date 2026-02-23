"""
CLI for interpolation experiments. Run tasks: python main.py --task 2 [--plot] ...
Run validations: python main.py  (or python run_validations.py)
"""

import argparse

from task_executor import TaskExecutor


def parse_args():
    p = argparse.ArgumentParser(
        description="Interpolation experiments. Use --task 2,3,4,5 to run a task; omit to run Phase 1–3 validations."
    )
    p.add_argument(
        "--task",
        type=int,
        default=None,
        metavar="N",
        help="Run task 2, 3, 4, or 5. If omitted, run validations (see run_validations.py).",
    )
    p.add_argument(
        "--precision",
        type=str,
        default="single",
        choices=["single", "double"],
        help="Precision for approximate interpolant (default: single).",
    )
    p.add_argument("--plot", action="store_true", help="Produce and save plots.")
    p.add_argument(
        "--interval",
        type=float,
        nargs=2,
        default=None,
        metavar=("A", "B"),
        help="Interval [a, b]. Default from function for chosen task.",
    )
    p.add_argument(
        "--degree-min",
        type=int,
        default=5,
        help="Min n for tasks 2-4 (default 5).",
    )
    p.add_argument(
        "--degree-max",
        type=int,
        default=20,
        help="Max n for tasks 2-4 (default 20).",
    )
    p.add_argument(
        "--n-max",
        type=int,
        default=50,
        help="Max n for Task 5 convergence (default 50).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for plots (default: output).",
    )
    p.add_argument(
        "--evaluation-grid-size",
        type=int,
        default=2000,
        help="Grid size for evaluation (default 2000).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task is None:
        from run_validations import run_all
        run_all()
    else:
        if args.task not in (2, 3, 4, 5):
            raise SystemExit("--task must be 2, 3, 4, or 5.")
        if args.interval is not None:
            args.interval = tuple(args.interval)
        TaskExecutor().run(args.task, args)
