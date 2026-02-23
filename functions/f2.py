"""
Function 2:
    f2(x; d) = product_{i=1}^d (x - i)
"""

import numpy as np

def make_f2(d):

    roots = np.arange(1, d + 1, dtype=np.float64)

    def product_form(x):
        x = np.asarray(x)
        result = np.ones_like(x, dtype=x.dtype)
        for r in roots.astype(x.dtype):
            result *= (x - r)
        return result

    def monomial_coefficients():
        coeffs = np.array([1.0])
        for i in range(1, d + 1):
            coeffs = np.convolve(coeffs, np.array([1.0, -float(i)]))
        return coeffs

    interval = (0.0, float(d + 1))

    return product_form, monomial_coefficients, interval, roots


# Default for task 3: degree d polynomial on (0, d+1).
_default_d = 9
func, _, interval, _ = make_f2(_default_d)
