import numpy as np


def findpeak(x, y):
    """
    Find the peak value and its corresponding x value in the given data.

    Parameters:
    x (array-like): The x values.
    y (array-like): The y values.

    Returns:
    peak_value (float): The maximum y value.
    peak_x (float): The x value corresponding to the maximum y value.
    """
    peak_index = np.argmax(y)
    peak_value = y[peak_index]
    peak_x = x[peak_index]

    return peak_x, peak_value
