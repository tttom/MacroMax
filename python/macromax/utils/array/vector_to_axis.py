import numpy as np


def vector_to_axis(vec, axis=0, ndim=None):
    """
    Adds singleton dimensions to a 1D vector up to dimension n and orients the vector in dimension axis (default 0)
    :param vec: the input vector
    :param axis: the target axis (default: 0)
    :param ndim: the number of desired dimensions. (default: axis)
    :return: an n-dimensional array with all-but-one singleton dimension
    """
    # vec = np.array(vec, copy=True)
    #
    # if ndim is None:
    #     ndim = np.abs(axis)
    #
    # indexes = np.ones(ndim, dtype=int)
    # indexes[axis] = vec.ravel().shape[0]
    #
    # return vec.reshape(indexes)
    if ndim is None:
        ndim = 0
    vec = np.array(vec, copy=True)
    indexes = [1]*ndim
    if vec.ndim > 0:
        indexes[axis] = vec.shape[0]
    return vec.reshape(indexes)
