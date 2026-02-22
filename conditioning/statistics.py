"""
Statistics over evaluation grids: sup-norm, mean, variance, Lebesgue constant, Hn.
Used on condition number vectors (e.g. kappa_x1, kappa_xy over a grid).
"""

import numpy as np


def sup_norm(values):
    """Max_i |v_i|. Ignores NaN."""
    v = np.asarray(values).ravel()
    if v.size == 0:
        return np.nan
    return np.nanmax(np.abs(v))


def mean(values):
    """Sample mean (1/n) * sum_i v_i. Ignores NaN."""
    v = np.asarray(values).ravel()
    if v.size == 0:
        return np.nan
    return np.nanmean(v)


def variance(values, ddof=0):
    """
    Variance over values. Default ddof=0: (1/n) * sum_i (v_i - bar{v})^2.
    Use ddof=1 for sample variance. Ignores NaN.
    """
    v = np.asarray(values).ravel()
    if v.size == 0:
        return np.nan
    return np.nanvar(v, ddof=ddof)


def lebesgue_constant(kappa_x1_values):
    """Lambda_n = max_x kappa(x, n, 1) = sup_norm(kappa_x1_values)."""
    return sup_norm(kappa_x1_values)


def Hn(kappa_xy_values):
    """H_n = max_x kappa(x, n, y) = sup_norm(kappa_xy_values)."""
    return sup_norm(kappa_xy_values)
