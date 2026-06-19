import numpy as np
from scipy.interpolate import interp1d


def resample(t, x, new_dt, kind="linear", fill_value="extrapolate"):
    t = np.asarray(t)
    x = np.asarray(x)

    if len(t) != len(x):
        raise ValueError("t and x must have the same length")
    if new_dt <= 0:
        raise ValueError("new_dt must be positive")

    t_start, t_end = t[0], t[-1]
    n = int(np.floor((t_end - t_start) / new_dt)) + 1
    t_new = t_start + np.arange(n) * new_dt

    interpolator = interp1d(
        t, x, kind=kind, fill_value=fill_value, bounds_error=False, assume_sorted=True
    )
    x_new = interpolator(t_new)
    return t_new, x_new
