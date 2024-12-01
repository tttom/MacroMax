from typing import Sequence
import numpy as np


def round125(values: int | float | Sequence | np.ndarray) -> np.ndarray | float | int:
    """
    Rounds numbers to the nearest 0, 1x10^n, 2x10^n or 5x10^n for display.

    :param values: a number, a list of numbers, or an nd-array
    :return: The rounded values as floating point values.
    """
    values_array = np.asarray(values)
    values_abs = np.maximum(np.abs(values_array), 1e-10)

    scale = np.log10(values_abs)
    scale = np.round(3.0 * scale) / 3.0

    order_of_magnitude = 10.0**np.floor(scale)
    pre_factor = np.round(10.0**np.mod(scale, 1.0))

    rounded_array = np.sign(values_array) * pre_factor * order_of_magnitude
    if not np.isscalar(rounded_array):
        return rounded_array
    else:
        rounded_value = rounded_array.item()
        if isinstance(values, int):
            return int(rounded_value)
        else:
            return rounded_value

