"""
# -------------------------------------------------------------------------
# Name:            error_utils.py
# Description:     General-purpose error/metric utility functions.
#                   Project-agnostic: import freely in any project.
#
#                   Functions:
#                     metricError(predicted, true, error_type) -> float
#                       Scalar error metrics between two arrays.
#
#                   Available error_type keys (accessible via METRIC_TYPES):
#                     mse, rmse,
#                     nrmse_range, nrmse_true, nrmse_quantile
#
# Author:          Antoine
# Collaborator:    Bona, Bruno, Edoardo
# Date created:    06/2026
# Project:         wec_modeling_benchmark (general-purpose)
# -------------------------------------------------------------------------
"""

import numpy as np

# Registry of all available metric names — importable for other projects
METRIC_TYPES = ["mse", "rmse", "nrmse_range", "nrmse_true", "nrmse_quantile"]


def metricError(predicted, true, error_type: str = "rmse") -> float:
    """
    Compute a scalar error metric between predicted and true arrays.

    Parameters
    ----------
    predicted  : array-like  — model output
    true       : array-like  — reference values
    error_type : str         — one of METRIC_TYPES

    Returns
    -------
    float — computed metric value
    """
    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)

    if predicted.shape != true.shape:
        raise ValueError("predicted and true must have the same shape.")

    e = predicted - true
    mse = float(np.mean(e**2))
    rmse = float(np.sqrt(mse))

    key = error_type.lower()

    if key == "mse":
        return mse

    elif key == "rmse":
        return rmse

    elif key == "nrmse_range":
        rng = float(np.max(true) - np.min(true))
        if rng == 0:
            raise ValueError("nrmse_range: true value range is zero.")
        return rmse / rng

    elif key == "nrmse_true":
        denom = float(np.sqrt(np.mean(true**2)))
        if denom == 0:
            raise ValueError("nrmse_true: true RMS is zero.")
        return rmse / denom

    elif key == "nrmse_quantile":
        q10, q90 = np.quantile(true, [0.10, 0.90])
        iqr = float(q90 - q10)
        if iqr == 0:
            raise ValueError("nrmse_quantile: IQR is zero.")
        return rmse / iqr

    else:
        raise ValueError(
            f'Unknown error_type "{error_type}". Available: {METRIC_TYPES}'
        )


# %% SIGNAL ANALYSIS FUNCTIONS (general-purpose, reusable across projects)


def cross_correlation(y_sim: np.ndarray, y_ref: np.ndarray, dt: float) -> tuple:
    """
    Normalised cross-correlation of y_sim vs y_ref.

    Returns
    -------
    lag_s : np.ndarray  — lag values in seconds (single vector for both x/xdot)
    corr  : np.ndarray  — normalised correlation values in [-1, 1]

    Usage
    -----
    lag_s, corr = cross_correlation(x_sim, x_ref, dt)
    peak_lag = lag_s[np.argmax(corr)]   # seconds of lead/lag
    peak_corr = np.max(corr)            # shape similarity (1 = perfect)
    """
    from scipy.signal import correlate, correlation_lags

    s = (y_sim - y_sim.mean()) / (y_sim.std() + 1e-12)
    r = (y_ref - y_ref.mean()) / (y_ref.std() + 1e-12)
    corr = correlate(s, r, mode="full") / len(r)
    lags = correlation_lags(len(s), len(r), mode="full")
    return lags * dt, corr


def inst_phase_diff(y_sim: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """
    Instantaneous phase difference in degrees via Hilbert transform.

    Inputs
    ------
    y_sim : predicted/model signal (e.g. x_sim, xdot_sim)
    y_ref : reference/CFD signal  (e.g. x_ref, xdot_ref)

    Returns
    -------
    phase_deg : np.ndarray — phase_sim - phase_ref in degrees (same length as inputs)
                Positive = simulation leads reference.
                Negative = simulation lags reference.
                The rose plot (polar histogram) of this array shows the
                distribution of instantaneous phase tracking error.
    """
    from scipy.signal import hilbert

    phi_sim = np.angle(hilbert(y_sim))
    phi_ref = np.angle(hilbert(y_ref))
    return np.degrees(np.unwrap(phi_sim - phi_ref))
