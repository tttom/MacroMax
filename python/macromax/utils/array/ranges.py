import numpy as np


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

