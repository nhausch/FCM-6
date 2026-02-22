"""f2(x) = 1/(1+x^2). Standard test function (Runge-type)."""

import numpy as np

def func(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + x**2)

interval = (-1.0, 1.0)
