import numpy as np
import scipy.constants as const

from . import log


def calc_ranges(range_lengths, sample_pitches=[], center_offsets=[]):
    """
    calc_ranges(range_lengths, sample_pitches, center_offsets)

    returns uniformly spaced ranges of length range_lengths(idx) with a elements spaced by
    sample_pitches(idx) and centered on center_offsets(idx). The center element
    is defined as the one in the center for an odd number of elements and the
    next one for an even number of elements. If a scalar is specified as sample pitch, it is used for all ranges. The
    default sample const.pitch is 1. If a scalar is specified as center_offsets, it is used for the first dimension
    and 0 is used for higher dimensions. The default center offset is 0.

    :param range_lengths: A list of the number of elements per dimension.
    :param sample_pitches: The distance between sample points (the dimensions of the n-D voxel). Default: all 1.
    :param center_offsets: Optional offset of the central voxel. The central voxel is the one with equal number of
        voxel at either side in every dimension. If the number of voxels is even, it is the voxel that has one voxel
        more preceeding it than after it.
    :return: a tuple of ranges, one for each range_length.
        If range_lengths is scalar, a single range is returned, not a tuple of a range.

    Example:

        xRange = calc_ranges(128, 1e-6)
        xRange, yRange = calc_ranges(np.array([128, 128]), np.array([1, 1])*1e-6)

    """
    is_single_range = np.isscalar(range_lengths)
    # Make sure the vectors are of the same length
    nb_dims = np.max((np.array(range_lengths).size, np.array(sample_pitches).size, np.array(center_offsets).size))

    range_lengths = pad_to_length(range_lengths, nb_dims, 1)
    sample_pitches = extend_to_length(sample_pitches, nb_dims)
    center_offsets = pad_to_length(center_offsets, nb_dims, 0)

    ranges = [co + sp * (np.arange(0, rl) - np.floor(rl / 2)) for co, sp, rl in
              zip(center_offsets, np.array(sample_pitches), range_lengths)]

    if is_single_range:
        return ranges[0]
    else:
        return ranges


def calc_frequency_ranges(*ranges, centered=False):
    """
    Determine equivalent frequency ranges for given time ranges.
    The results are ifftshifted so that the zero frequency is in the first
    vector position, unless centered=True.
    This function always returns a tuple

    Example usage:
        (xf_range) = calc_frequency_ranges(x_range) # or
        xf_range = calc_frequency_ranges(x_range)[0]
        xf_range, yf_range = calc_frequency_ranges(x_range, y_range)
        xf_range, yf_range = calc_frequency_ranges(x_range, y_range, centered=True)

    :param ranges: one or more (spatial) time range vectors
    :param centered: Boolean indicating whether the resulting ranges should have the zero at the center.
        Default False.
    :return: A tuple with one or more (spatial) frequency range vectors
    """
    f_ranges = []
    for rng in ranges:
        rng = np.array(rng)  # convert python range to numpy range
        nb = len(rng)
        if nb > 1:
            dt = np.array(rng[-1] - rng[0]) / (nb - 1)
            f_range = (np.arange(0, nb) - np.floor(nb / 2)) / (nb * dt)
        else:
            f_range = 0.0 * rng

        if not centered:
            f_range = np.fft.ifftshift(f_range)

        f_ranges.append(f_range)

    return f_ranges


def to_dim(x, n, axis=0):
    """
    Adds singleton dimensions to a 1D vector up to dimension n and orients the vector in dimension axis (default 0)

    :param x: the input vector
    :param n: the number of desired dimensions
    :param axis: the target axis (default: 0)

    :return: a n-dimensional array with all-but-one singleton dimension
    """
    x = np.array(x, copy=True)
    indexes = [1]*n
    if x.ndim > 0:
        indexes[axis] = x.shape[0]
    return x.reshape(indexes)


def add_dims(arr, n, total):
    """
    Adds n singleton dimension to the right and completes the number of dimensions at the left to makes sure that the
    total number of dimensions equals 'total'.

    :param arr: The input array.
    :param n: The number of dimensions to add at the right.
    :param total: The total number of dimensions required.

    :return: The reshaped array.
    """
    for idx in range(n):
        arr = [arr]
    arr = np.array(arr)

    def add_trailing_dims(arr, n):
        """
        Adds n singleton dimension to the left

        :param arr: The input array.
        :param n: The number of dimensions to add to the left.

        :return: The reshaped array.
        """
        arr = np.array(arr)
        for idx in range(n):
            arr = np.expand_dims(arr, -1)

        return arr

    arr = add_trailing_dims(arr, total - arr.ndim)

    return arr


def pad_to_length(vec, length, padding_value=0):
    """
    Pads a vector on the right-hand side to a given length.

    :param vec: The to-be-padded vector. This can be a numpy.ndarray, a range, a list, or a scalar.
    :param length: The length of the to-be-returned vector.
    :param padding_value: The numeric padding value (default: 0).

    :return: The padded vector of length 'length' as a numpy ndarray.
    """
    values = np.array(vec).flatten()

    return np.pad(values, (0, length - values.size), 'constant', constant_values=padding_value)


def extend_to_length(vec, length):
    """
    Extends a vector on the right-hand side to a given length using its final value.

    :param vec: The to-be-extended vector. This can be a numpy.ndarray, a range, a list, or a scalar.
    :param length: The length of the to-be-returned vector.

    :return: The extended vector with length 'length' as a numpy ndarray.
    """
    values = np.array(vec).flatten()

    return np.pad(values, (0, length - values.size), 'edge')


def complex2rgb(complex_image, normalization=None, inverted=False):
    """
    Converts a complex image to a RGB image.

    :param complex_image: A 2D array
    :param normalization: An optional scalar to indicate the target magnitude of the maximum value (1.0 is saturation).
    :param inverted: By default 0 is shown as black and amplitudes of 1 as the brightest hues.
        When inverted is True, zeros are shown as white and amplitudes of 1 are shown as black.
    :return: Returns a 3D array representing the red, green, and blue channels of a displayable image.
    """
    amp = np.abs(complex_image)
    ph = np.angle(complex_image)

    if normalization is not None:
        if normalization is True:
            normalization = 1.0
        if normalization > 0:
            max_value = np.max(amp)
            if max_value > 0:
                amp *= (normalization / max_value)
        else:
            log.warning('Negative normalization factor %d ignored.', normalization)

    hue = ph / (2 * const.pi) + 0.5
    if not inverted:
        hsv = np.concatenate(
            (hue[:, :, np.newaxis], np.ones((*hue.shape, 1)), np.minimum(1.0, amp[:, :, np.newaxis])), axis=2)
    else:
        hsv = np.concatenate((hue[:, :, np.newaxis],
                              np.minimum(1.0, amp[:, :, np.newaxis]),
                              np.maximum(0.0, 1.0 - 0.5 * amp[:, :, np.newaxis])), axis=2)

    return hsv2rgb(hsv)  # Convert HSV to an RGB image


def hsv2rgb(hsv):
    """
    Converts a hue, saturation, and values to an RGB image.

    :param hsv: A 3D array with hue, saturation, and values per pixel of a 2D image.

    :return: Returns a 3D array representing the red, green, and blue channels of a displayable image.
    """
    # Convert hsv to an RGB image
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    hue = 6.0 * hue
    I = np.array(hue, dtype=np.int8)
    F = hue - I
    P = val * (1.0 - sat)
    Q = val * (1.0 - sat * F)
    T = val * (1.0 - (sat * (1.0 - F)))

    I %= 6
    red = ((I == 0) | (I == 5)) * val + (I == 1) * Q + ((I == 2) | (I == 3)) * P + (I == 4) * T
    green = (I == 0) * T + ((I == 1) | (I == 2)) * val + (I == 3) * Q + (I >= 4) * P
    blue = (I <= 1) * P + (I == 2) * T + ((I == 3) | (I == 4)) * val + (I == 5) * Q
    rgb = np.concatenate((red[:, :, np.newaxis], green[:, :, np.newaxis], blue[:, :, np.newaxis]), axis=2)

    return rgb


def ranges2extent(*ranges):
    """
    Utility function to determine extent values for imshow from the ranges of positions at which the pixels are
    specified. The extents are half a pixel larger at each end of the range, for each range.

    :param ranges: Monotonically increasing ranges, one per dimension: first vertical, next horizontal.
        Each range may be a numpy.array or a Python range.

    :return: An iterable with the extents of each dimension as 4 numerical values: [left, right, bottom, top]
        (assuming origin='lower' on imshow)
    """
    extent = []
    for idx, rng in enumerate(ranges[::-1]):
        step = rng[1] - rng[0]
        first, last = rng[0] - 0.5 * step, rng[-1] + 0.5 * step
        if idx == 1:
            first, last = last, first
        extent.append(first)
        extent.append(last)

    return np.array(extent)


def word_align(input_array, word_length=32):
    """
    Returns a new array thay is byte-aligned to words of length word_length.
    This may be required for libraries such as pyfftw

    :param input_array: The input array to align.
    :param word_length: The word length to align to. This must be an integer multiple of the dtype size.

    :return: A word-aligned array with the same contents and shape as input_array.
    """
    if (input_array.ctypes.data % word_length) == 0:
        aligned_array = input_array
    else:
        extra = int(np.ceil(word_length / input_array.itemsize))
        buffer = np.empty(input_array.size + extra, dtype=input_array.dtype)
        offset = int((-buffer.ctypes.data % word_length) / input_array.itemsize)
        aligned_array = buffer[offset:(offset + input_array.size)].reshape(input_array.shape)
        np.copyto(aligned_array, input_array)

    assert (aligned_array.ctypes.data % word_length) == 0

    return aligned_array

