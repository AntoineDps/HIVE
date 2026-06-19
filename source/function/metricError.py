import numpy as np


def metricError(predicted, true, error_type="mse"):
    """
    Compute various error metrics between predicted and true values.

    Parameters
    ----------
    predicted : array-like
        Predicted values.
    true : array-like
        True values.
    error_type : str
        One of: 'mse', 'rmse', 'nrmse_range', 'nrmse_true', 'nrmse_quantile'.

    Returns
    -------
    float
        Computed error metric.
    """
    predicted = np.asarray(predicted)
    true = np.asarray(true)

    if predicted.shape != true.shape:
        raise ValueError("Predicted and true values must have the same shape.")

    e = predicted - true
    mse = np.mean(e**2)
    rmse = np.sqrt(mse)

    error_type = error_type.lower()

    if error_type == "mse":
        return mse
    elif error_type == "rmse":
        return rmse
    elif error_type == "nrmse_range":
        rng = np.max(true) - np.min(true)
        if rng == 0:
            raise ValueError("Cannot compute NRMSE range: true value range is zero.")
        return rmse / rng
    elif error_type == "nrmse_true":
        denom = np.sqrt(np.mean(true**2))
        if denom == 0:
            raise ValueError("Cannot compute NRMSE true: true value RMS is zero.")
        return rmse / denom
    elif error_type == "nrmse_quantile":
        Q10 = np.quantile(true, 0.10)
        Q90 = np.quantile(true, 0.90)
        IQR = Q90 - Q10
        if IQR == 0:
            raise ValueError("Cannot compute NRMSE quantile: IQR is zero.")
        return rmse / IQR
    else:
        raise ValueError(
            f'Unknown error type "{error_type}". Supported: '
            "mse, rmse, nrmse_range, nrmse_true, nrmse_quantile."
        )
