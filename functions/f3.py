"""f3(x) = exp(x). Smooth function for interpolation tests."""

import numpy as np

def func(x):
    return np.exp(np.asarray(x, dtype=float))

interval = (-1.0, 1.0)
