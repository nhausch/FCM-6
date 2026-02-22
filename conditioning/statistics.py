import numpy as np


# Max_i |v_i|. Ignores NaN.
def sup_norm(values):
    """"""
    v = np.asarray(values).ravel()
    if v.size == 0:
        return np.nan
    return np.nanmax(np.abs(v))

# Sample mean (1/n) * sum_i v_i. Ignores NaN.
def mean(values):
    v = np.asarray(values).ravel()
    if v.size == 0:
        return np.nan
    return np.nanmean(v)

# Variance over values. Default ddof=0: (1/n) * sum_i (v_i - bar{v})^2.
# Use ddof=1 for sample variance. Ignores NaN.
def variance(values, ddof=0):
    v = np.asarray(values).ravel()
    if v.size == 0:
        return np.nan
    return np.nanvar(v, ddof=ddof)

# Lambda_n = max_x kappa(x, n, 1) = sup_norm(kappa_x1_values).
def lebesgue_constant(kappa_x1_values):
    return sup_norm(kappa_x1_values)

# H_n = max_x kappa(x, n, y) = sup_norm(kappa_xy_values).
def Hn(kappa_xy_values):
    return sup_norm(kappa_xy_values)
