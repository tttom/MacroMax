import numpy as np

from typing import Union, Sequence, Optional
from numbers import Complex
array_like = Union[Complex, Sequence, np.ndarray]


def add_dims_on_right(arr: array_like, new_axes_on_right: int = 0, ndim: Optional[int] = None):
    """
    A function that returns a view with additional singleton dimensions on the right. This is useful to broadcast and
    expand on the left-hand side with an array of an arbitrary number of dimensions on the right-hand side.

    :param arr: The original array or sequence of numbers that can be converted to an array.
    :param new_axes_on_right: (optional) The number of axes to add on the right hand side.
        Default: ndim - arr.ndim or 0 if the latter is not specified. If negative, singleton dimensions are removed from the
        right. This will fail if those on the right are not singleton dimensions.
    :param ndim: (optional) The total number of axes of the returned view. Default: arr.ndim + new_axes_on_right

    :return: A view with ndim == arr.ndim + new_axes_on_right dimensions.
    """
    arr = np.asarray(arr)
    if ndim is None:
        ndim = arr.ndim + new_axes_on_right
    else:
        new_axes_on_right = ndim - arr.ndim
    if new_axes_on_right > 0:
        return np.expand_dims(arr, tuple(range(arr.ndim, ndim)))
        # return arr.reshape(*arr.shape, *([1] * new_axes_on_right))
    else:
        return arr.reshape(arr.shape[:ndim])
