import numpy as np


def word_align(input_array, word_length=32):
    """
    Returns a new array that is byte-aligned to words of length word_length bytes.
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

