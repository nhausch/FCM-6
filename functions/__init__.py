"""Test functions for interpolation tasks. get(task_id) returns (func, a, b)."""

from . import f1, f2, f3, f4

_registry = {
    2: (f1.func, f1.interval[0], f1.interval[1]),
    3: (f2.func, f2.interval[0], f2.interval[1]),
    4: (f3.func, f3.interval[0], f3.interval[1]),
    5: (f4.func, f4.interval[0], f4.interval[1]),
}

def get(task_id):
    """Return (f, a, b) for task_id in {2, 3, 4, 5}."""
    if task_id not in _registry:
        raise ValueError(f"Unknown task_id: {task_id}. Use 2, 3, 4, or 5.")
    f, a, b = _registry[task_id]
    return f, a, b

__all__ = ["f1", "f2", "f3", "f4", "get"]
