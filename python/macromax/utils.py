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
              zip(center_offsets, sample_pitches, range_lengths)]

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
    indexes[axis] = x.shape[0]
    return x.reshape(indexes)


def add_dims(A, n, total):
    """
    Adds n singleton dimension at the right and makes sure that the total number of dimensions is total

    :param A: The input array.
    :param n: The number of dimensions to add at the right.
    :param total: The total number of dimensions required.

    :return: The reshaped array.
    """
    for idx in range(n):
        A = [A]
    A = np.array(A)

    A = add_trailing_dims(A, total - A.ndim)

    return A


def add_trailing_dims(A, n):
    """
    Adds n singleton dimension at the left

    :param A: The input array.
    :param n: The number of dimensions to add to the left.

    :return: The reshaped array.
    """
    A = np.array(A)
    for idx in range(n):
        A = np.expand_dims(A, -1)
    return A


def complex2RGB(complex_image, normalization=None, inverted=False):
    """
    Converts a complex image to a RGB image.

    :param complex_image: A 2D array
    :param normalization: An optional scalar to indicate the target magnitude of the maximum value (1.0 is saturation).
    :param inverted: By default 0 is shown as black and amplitudes of 1 as the brightest hues.
        When inverted is True, zeros are shown as white and amplitudes of 1 are shown as black.
    :return: Returns a 3D array representing the red, green, and blue channels of a displayable image.
    """
    A = np.abs(complex_image)
    P = np.angle(complex_image)

    if normalization is not None:
        if normalization is True:
            normalization = 1.0
        if normalization > 0:
            max_value = np.max(A)
            if max_value > 0:
                A *= (normalization / max_value)
        else:
            log.warn('Negative normalization factor %d ignored.', normalization)

    H = P / (2 * const.pi) + 0.5
    if not inverted:
        HSV = np.concatenate((H[:, :, np.newaxis], np.ones((*H.shape, 1)), np.minimum(1.0, A[:, :, np.newaxis])), axis=2)
    else:
        HSV = np.concatenate((H[:, :, np.newaxis],
                              np.minimum(1.0, A[:, :, np.newaxis]),
                              np.maximum(0.0, 1.0 - 0.5 * A[:, :, np.newaxis])), axis=2)

    return hsv2rgb(HSV)  # Convert HSV to an RGB image


def hsv2rgb(HSV):
    """
    Converts a hue, saturation, and values to an RGB image.

    :param HSV: A 3D array with hue, saturation, and values per pixel of a 2D image.

    :return: Returns a 3D array representing the red, green, and blue channels of a displayable image.
    """
    # Convert HSV to an RGB image
    H = HSV[:, :, 0]
    S = HSV[:, :, 1]
    V = HSV[:, :, 2]

    H = 6.0 * H
    I = np.array(H, dtype=np.int8)
    F = H - I
    P = V * (1.0 - S)
    Q = V * (1.0 - S * F)
    T = V * (1.0 - (S * (1.0 - F)))

    I %= 6
    R = ((I == 0) | (I == 5)) * V + (I == 1) * Q + ((I == 2) | (I == 3)) * P + (I == 4) * T
    G = (I == 0) * T + ((I == 1) | (I == 2)) * V + (I == 3) * Q + (I >= 4) * P
    B = (I <= 1) * P + (I == 2) * T + ((I == 3) | (I == 4)) * V + (I == 5) * Q
    RGB = np.concatenate((R[:, :, np.newaxis], G[:, :, np.newaxis], B[:, :, np.newaxis]), axis=2)

    return RGB


def pad_to_length(V, length, padding_value=0):
    values = np.array(V).flatten()

    return np.pad(values, (0, length - values.size), 'constant', constant_values=padding_value)


def extend_to_length(V, length):
    values = np.array(V).flatten()

    return np.pad(values, (0, length - values.size), 'edge')


def ranges2extent(*ranges):
    """
    Utility function to determine extent values for imshow

    :param ranges: monotonically increasing ranges, one per dimension (vertical, horizontal)

    :return: a 1D array
    """
    extent = []
    for idx, rng in enumerate(ranges[::-1]):
        step = rng[1] - rng[0]
        first, last = rng[0], rng[-1]
        if idx == 1:
            first, last = last, first
        extent.append(first - 0.5 * step)
        extent.append(last + 0.5 * step)

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

