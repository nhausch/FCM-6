"""f4(x) = 1/(1+25*x^2). Runge function for convergence study (uniform diverges, Chebyshev stabilizes)."""

import numpy as np

def func(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + 25.0 * x**2)

interval = (-1.0, 1.0)
