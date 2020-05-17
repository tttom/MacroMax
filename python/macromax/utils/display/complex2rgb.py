import numpy as np
from typing import Union, Sequence

from .hsv import hsv2rgb


def complex2rgb(complex_image: Union[complex, Sequence, np.array], normalization: Union[bool, float, int]=None,
                inverted=False, dtype=np.float):
    """
    Converts a complex image to a RGB image.
    :param complex_image: A 2D array
    :param normalization: An optional multidimensional to indicate the target magnitude of the maximum value
    (1.0 is saturation).
    :param inverted: By default 0 is shown as black and amplitudes of 1 as the brightest hues. Setting this input
    argument to True could be useful for printing on white background.
    :param dtype: The output data type. The value is scaled to the maximum positive numeric range for integers
    (np.iinfo(dtype).max). Floating point numbers are within [0, 1]. (Default: float)
    :return: A real 3d-array with values between 0 and 1.
    """
    # Make sure that this is a numpy 2d-array
    complex_image = np.array(complex_image)
    while complex_image.ndim < 2:
        complex_image = complex_image[..., np.newaxis]

    amplitude = np.abs(complex_image)[..., np.newaxis]
    phase = np.angle(complex_image)[..., np.newaxis]

    if normalization is not None:
        if normalization > 0:
            max_value = np.max(abs(amplitude.ravel()))
            if max_value > 0:
                amplitude *= (normalization / max_value)

    intensity = np.minimum(amplitude, 1.0)

    hue = phase / (2 * np.pi) + 0.5
    saturation = np.ones(hue.shape)

    if inverted:
        saturation = np.minimum(1.0, intensity)
        intensity = np.maximum(0.0, 1.0 - 0.5 * intensity)

    hsv_image = np.concatenate((hue, saturation, intensity), axis=-1)

    rgb_image = hsv2rgb(hsv_image)  # Convert HSV to an RGB image

    if issubclass(dtype, np.integer):
        rgb_image = rgb_image * np.iinfo(dtype).max + 0.5

    return rgb_image.astype(dtype)

