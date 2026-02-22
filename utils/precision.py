import numpy as np


def get_dtype(precision: str) -> np.dtype:
    if precision == "single":
        return np.float32
    return np.float64

class PrecisionContext:
    def __init__(self, precision: str):
        self.dtype = get_dtype(precision)
        self.eps = np.finfo(self.dtype).eps
