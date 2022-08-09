import numpy as np


def round125(values):
    """
    Rounds numbers to the nearest 0, 1x10^n, 2x10^n or 5x10^n for display.

    :param values: a number, a list of numbers, or an nd-array
    :return: The rounded values as floating point values.
    """
    values = np.array(values)
    values_abs = np.maximum(np.abs(values), 1e-10)

    scale = np.log10(values_abs)
    scale = np.round(3.0 * scale) / 3.0

    order_of_magnitude = 10.0**np.floor(scale)
    pre_factor = np.round(10.0**np.mod(scale, 1.0))

    return np.sign(values) * pre_factor * order_of_magnitude


if __name__ == '__main__':
    print(round125([0, 1, 2, np.pi, 4, 5, 6, 7, 8, 9, 10, 15, 123, 1234]))
