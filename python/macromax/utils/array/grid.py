from typing import Union, Sequence
import numpy as np

from .vector_to_axis import vector_to_axis
from macromax.utils import ft


class Grid(Sequence):
    """
    A class representing an immutable uniformly-spaced plaid grid.

    See also: MutableGrid
    """
    def __init__(self, shape, step=None, extent=None, first=None, center=None, last=None, include_last=False,
                 ndim: int=None,
                 flat: Union[bool, Sequence, np.ndarray]=False,
                 origin_at_center: Union[bool, Sequence, np.ndarray]=True,
                 center_at_index: Union[bool, Sequence, np.ndarray]=True):
        """
        Construct an immutable Grid object.

        :param shape: An integer vector array with the shape of the sampling grid.
        :param step: A vector array with the spacing of the sampling grid.
        :param extent: The extent of the sampling grid as shape * step
        :param first: A vector array with the first element for each dimension.
        The first element is the smallest element if step is positive, and the largest when step is negative.
        :param center: A vector array with the center element for each dimension. The center position in the grid is
        rounded to the next integer index unless center_at_index is set to False for that partical axis.
        :param last: A vector array with the last element for each dimension. Unless include_last is set to True for
        the associated dimension, all but the last element is returned when calling self[axis].
        :param include_last: A boolean vector array indicating whether the returned vectors, self[axis], should include
        the last element (True) or all-but-the-last (False)
        :param ndim: A scalar integer indicating the number of dimensions of the sampling space.
        :param flat: A boolean vector array indicating whether the returned vectors, self[axis], should be
        flattened (True) or returned as an open grid (False)
        :param origin_at_center: A boolean vector array indicating whether the origin should be fft-shifted (True)
        or be ifftshifted to the front (False) of the returned vectors for self[axis].
        :param center_at_index: A boolean vector array indicating whether the center of the grid should be rounded to an
        integer index for each dimension. If False and the shape has an even number of elements, the next index is used
        as the center, (self.shape / 2).astype(np.int).
        """
        # Figure out what dimension is required
        if ndim is None:
            ndim = 0
            if shape is not None:
                ndim = np.maximum(ndim, np.array(shape).size)
            if step is not None:
                ndim = np.maximum(ndim, np.array(step).size)
            if extent is not None:
                ndim = np.maximum(ndim, np.array(extent).size)
            if first is not None:
                ndim = np.maximum(ndim, np.array(first).size)
            if center is not None:
                ndim = np.maximum(ndim, np.array(center).size)
            if last is not None:
                ndim = np.maximum(ndim, np.array(last).size)
        self.__ndim = ndim

        def is_vector(value):
            return value is not None and not np.isscalar(value)
        self.__multidimensional = is_vector(shape) or is_vector(step) or is_vector(extent) or \
                                  is_vector(first) or is_vector(center) or is_vector(last)

        # Convert all input arguments to vectors of length ndim
        shape, step, extent, first, center, last, flat, origin_at_center, include_last, center_at_index = \
            self.__all_to_ndim(shape, step, extent, first, center, last, flat, origin_at_center, include_last,
                               center_at_index)
        shape = np.round(shape).astype(np.int)  # Make sure that the shape is integer

        if shape is None:
            if step is not None and extent is not None:
                shape = np.round(np.real(self._to_ndim(extent) / self._to_ndim(step))).astype(np.int)
            else:
                shape = self._to_ndim(1)  # Default shape to 1
            shape += include_last
        # At this point the shape is not None
        if step is None:
            if extent is not None:
                nb_steps = shape - include_last
                step = extent / nb_steps
            else:
                step = self._to_ndim(1)  # Default step size to 1
        # At this point the step is not None
        if center is None:
            if last is not None:
                nb_steps = shape - include_last
                first = last - step * nb_steps
            if first is not None:
                half_shape = shape / 2
                half_shape[center_at_index] = np.floor(half_shape[center_at_index])
                if np.all(center_at_index):
                    half_shape = half_shape.astype(np.int)
                center = first + step * half_shape
            else:
                center = self._to_ndim(0)  # Center around 0 by default
        # At this point the center is not None

        self._shape = shape
        self._step = step
        self._center = center
        self._flat = flat
        self._origin_at_center = origin_at_center
        self.__center_at_index = center_at_index

    @staticmethod
    def from_ranges(*ranges: Union[int, float, complex, Sequence, np.ndarray]):
        """
        Converts one or more ranges of numbers to a single Grid object representation.
        The ranges can be specified as separate parameters or as a tuple.

        :param ranges: one or more ranges of uniformly spaced numbers.
        :return: A Grid object that represents the same ranges.
        """
        # Unpack if it is a tuple
        if isinstance(ranges[0], tuple):
            ranges = ranges[0]
        # Convert slices to range vectors. This won't work with infinite slices
        ranges = [(np.arange(rng.start, rng.stop, rng.step) if isinstance(rng, slice) else rng) for rng in ranges]
        ranges = [np.array([rng] if np.isscalar(rng) else rng) for rng in ranges]  # Treat a scalar a singleton vector
        ranges = [(rng.swapaxes(0, axis).reshape(rng.shape[axis], -1)[:, 0] if rng.ndim > 1 else rng)
                  for axis, rng in enumerate(ranges)]
        # Work out some properties about the shape and the size of each dimension
        shape = np.array([rng.size for rng in ranges])
        singleton = shape <= 1
        odd = np.mod(shape, 2) == 1
        # Work our what are the first and last elements, which could be at the center
        first = np.array([rng[0] for rng in ranges])  # first when fftshifted, center+ otherwise
        before_center = np.array([rng[int((rng.size - 1) / 2)] for rng in ranges])  # last when ifftshifted, center+ otherwise
        after_center = np.array([rng[-int(rng.size / 2)] for rng in ranges])  # first when ifftshifted, center- otherwise
        last = np.array([rng[-1] for rng in ranges])  # last when fftshifted, center- otherwise
        # The last value is included!

        # If it is not monotonous it is ifftshifted
        origin_at_center = np.abs(last - first) >= np.abs(before_center - after_center)
        # Figure out what is the step size and the center element
        extent_m1 = origin_at_center * (last - first) + (1 - origin_at_center) * (before_center - after_center)
        step = extent_m1 / (shape - 1 + singleton)  # Note that the step can be a complex number
        center = origin_at_center * (odd * before_center + (1 - odd) * after_center) + (1 - origin_at_center) * first

        return Grid(shape=shape, step=step, center=center, flat=False, origin_at_center=origin_at_center)


    #
    # Grid and array properties
    #

    @property
    def ndim(self) -> int:
        """The number of dimensions of the space this grid spans."""
        return self.__ndim

    @property
    def shape(self) -> np.array:
        """The number of sample points along each axis of the grid."""
        return self._shape

    @property
    def step(self) -> np.ndarray:
        """The sample spacing along each axis of the grid."""
        return self._step

    @property
    def center(self) -> np.ndarray:
        """The central coordinate of the grid."""
        return self._center

    @property
    def center_at_index(self) -> np.array:
        """
        Boolean vector indicating whether the central coordinate is aligned with a grid point when the number
        of points is even along the associated axis. This has no effect when the the number of sample points is odd.
        """
        return self.__center_at_index

    @property
    def flat(self) -> np.array:
        """
        Boolean vector indicating whether self[axis] returns flattened (raveled) vectors (True) or not (False).
        """
        return self._flat

    @property
    def origin_at_center(self) -> np.array:
        """
        Boolean vector indicating whether self[axis] returns ranges that are monotonous (True) or
        ifftshifted so that the central index is the first element of the sequence (False).
        """
        return self._origin_at_center
    
    #
    # Conversion methods
    #

    @property
    def as_flat(self):
        """
        :return: A new Grid object where all the ranges are 1d-vectors (flattened or raveled)
        """
        shape, step, center, center_at_index, origin_at_center = \
            self.shape, self.step, self.center, self.center_at_index, self.origin_at_center
        if not self.multidimensional:
            shape, step, center, center_at_index, origin_at_center = \
                shape[0], step[0], center[0], center_at_index[0], origin_at_center[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=True, origin_at_center=origin_at_center)

    @property
    def as_non_flat(self):
        """
        :return: A new Grid object where all the ranges are 1d-vectors (flattened or raveled)
        """
        shape, step, center, center_at_index, origin_at_center = \
            self.shape, self.step, self.center, self.center_at_index, self.origin_at_center
        if not self.multidimensional:
            shape, step, center, center_at_index, origin_at_center = \
                shape[0], step[0], center[0], center_at_index[0], origin_at_center[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=False, origin_at_center=origin_at_center)

    @property
    def as_origin_at_0(self):
        """
        :return: A new Grid object where all the ranges are ifftshifted so that the origin as at index 0.
        """
        shape, step, center, center_at_index, flat = self.shape, self.step, self.center, self.center_at_index, self.flat
        if not self.multidimensional:
            shape, step, center, center_at_index, flat = shape[0], step[0], center[0], center_at_index[0], flat[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=flat, origin_at_center=False)

    @property
    def as_origin_at_center(self):
        """
        :return: A new Grid object where all the ranges have the origin at the center index, even when the number of
        elements is odd.
        """
        shape, step, center, center_at_index, flat = self.shape, self.step, self.center, self.center_at_index, self.flat
        if not self.multidimensional:
            shape, step, center, center_at_index, flat = shape[0], step[0], center[0], center_at_index[0], flat[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=flat, origin_at_center=True)

    def swapaxes(self, axes: Union[slice, Sequence, np.array]):
        """Reverses the order of the specified axes."""
        axes = np.array(axes).flatten()
        all_axes = np.arange(self.ndim)
        all_axes[axes] = axes[::-1]
        return self.transpose(all_axes)

    def transpose(self, axes: Union[None, slice, Sequence, np.array]=None):
        """Reverses the order of all axes."""
        if axes is None:
            axes = np.arange(self.ndim-1, -1, -1)
        return self.project(axes)

    def project(self, axes_to_keep: Union[int, slice, Sequence, np.array, None]=None,
                axes_to_remove: Union[int, slice, Sequence, np.array, None] = None):
        """
        Removes all but the specified axes and reduces the dimensions to the number of specified axes.
        :param axes_to_keep: The indices of the axes to keep.
        :param axes_to_remove: The indices of the axes to remove. Default: None
        :return: A Grid object with ndim == len(axes) and shape == shape[axes].
        """
        if axes_to_keep is None:
            axes_to_keep = np.arange(self.ndim)
        elif isinstance(axes_to_keep, slice):
            axes_to_keep = np.arange(self.ndim)[axes_to_keep]
        if np.isscalar(axes_to_keep):
            axes_to_keep = [axes_to_keep]
        axes_to_keep = np.array(axes_to_keep)
        if axes_to_remove is None:
            axes_to_remove = []
        elif isinstance(axes_to_remove, slice):
            axes_to_remove = np.arange(self.ndim)[axes_to_remove]
        if np.isscalar(axes_to_remove):
            axes_to_remove = [axes_to_remove]
        axes_to_keep = np.array([_ for _ in axes_to_keep if _ not in axes_to_remove])

        if np.any(axes_to_keep >= self.ndim) or np.any(axes_to_keep < -self.ndim):
            raise IndexError(f"Axis range {axes_to_keep} requested from a Grid of dimension {self.ndim}.")

        return Grid(shape=self.shape[axes_to_keep], step=self.step[axes_to_keep], center=self.center[axes_to_keep], flat=self.flat[axes_to_keep],
                    origin_at_center=self.origin_at_center[axes_to_keep],
                    center_at_index=self.center_at_index[axes_to_keep]
                    )

    #
    # Derived properties
    #

    @property
    def first(self) -> np.ndarray:
        """
        :return: A vector with the first element of each range
        """
        half_shape = self.shape / 2
        half_shape[self.center_at_index] = np.floor(half_shape[self.center_at_index])
        if np.all(np.mod(self.shape[np.logical_not(self.center_at_index)], 2) == 0):
            half_shape = half_shape.astype(np.int)
        return self._center - self.step * half_shape

    @property
    def extent(self) -> np.ndarray:
        """ The spatial extent of the sampling grid. """
        return self.shape * self.step

    #
    # Sequence methods
    #

    @property
    def size(self) -> int:
        """ The total number of sampling points as an integer scalar. """
        return int(np.prod(self.shape))

    @property
    def dtype(self):
        """ The numeric data type for the coordinates. """
        return (self.step[0] + self.center[0]).dtype

    #
    # Frequency grids
    #

    @property
    def f(self):
        """ The equivalent frequency Grid. """
        shape, step, flat = self.shape, 1 / self.extent, self.flat
        if not self.multidimensional:
            shape, step, flat = shape[0], step[0], flat[0]

        return Grid(shape=shape, step=step, flat=flat, origin_at_center=False, center_at_index=True)

    @property
    def k(self):
        """ The equivalent k-space Grid. """
        return self.f * (2 * np.pi)

    #
    # Arithmetic methods
    #
    def __add__(self, term):
        """ Add a (scalar) offset to the Grid coordinates. """
        d = self.__dict__
        new_center = self.center + np.asarray(term)
        if not self.multidimensional:
            new_center = new_center[0]
        d['center'] = new_center
        return Grid(**d)

    def __mul__(self, factor: Union[int, float, complex, Sequence, np.array]):
        """
        Scales all ranges with a factor.
        :param factor: A scalar factor for all dimensions, or a vector of factors, one for each dimension.
        :return: A new scaled Grid object.
        """
        if isinstance(factor, Grid):
            raise TypeError("A Grid object can't be multiplied with a Grid object."
                            + "Use matmul @ to determine the tensor space.")
        d = self.__dict__
        factor = np.asarray(factor)
        new_step = self.step * factor
        new_center = self.center * factor
        if not self.multidimensional:
            new_step = new_step[0]
            new_center = new_center[0]
        d['step'] = new_step
        d['center'] = new_center
        return Grid(**d)

    def __matmul__(self, other):
        """
        Determines the Grid spanning the tensor space, with ndim equal to the sum of both ndims.
        :param other: The Grid with the right-hand dimensions.
        :return: A new Grid with ndim == self.ndim + other.ndim.
        """
        return Grid(shape=(*self.shape, *other.shape), step=(*self.step, *other.step),
                    center=(*self.center, *other.center),
                    flat=(*self.flat, *other.flat),
                    origin_at_center=(*self.origin_at_center, *other.origin_at_center),
                    center_at_index=(*self.center_at_index, *other.center_at_index)
                    )

    def __sub__(self, term: Union[int, float, complex, Sequence, np.ndarray]):
        """ Subtract a (scalar) value from all Grid coordinates. """
        return self + (- term)

    def __truediv__(self, denominator: Union[int, float, complex, Sequence, np.ndarray]):
        """ Divide the grid coordinates by a value.
        :param denominator: The denominator to divide by.
        :returns A new Grid with the divided coordinates.
        """
        return self * (1 / denominator)

    def __neg__(self):
        """ Invert the coordinate values and the direction of the axes. """
        return self.__mul__(-1)

    #
    # iterator methods
    #

    def __len__(self) -> int:
        """
        The number of axes in this sampling grid.
        Or, the number of elements when this object is not multi-dimensional.
        """
        if self.multidimensional:
            return self.ndim
        else:
            return self.shape[0]  # Behave as a single Sequence

    def __getitem__(self, key: Union[int, slice, Sequence]):
        """
        Select one or more axes from a multi-dimensional grid,
        or select elements from a single-dimensional object.
        """
        # if self.multidimensional:
        #     return self.project(key)
        # else:
        #     rng = self.center[0] + self.step[0] * (np.arange(self.shape[0]) - (self.shape[0] / 2).astype(np.int))
        #     if not self.__origin_at_center[0]:
        #         rng = ft.ifftshift(rng)
        #     if not self.flat[0]:
        #         rng = vector_to_axis(rng, axis=0, ndim=self.ndim)
        #todo: finish the above to replace project with indexing?

        scalar_key = np.isscalar(key)
        if self.multidimensional:
            indices = np.arange(self.ndim)[key]
        else:
            indices = np.arange(self.shape[0])[key]
        if np.isscalar(indices):
            indices = (indices, )
        result = []
        for idx in indices:
            axis = idx if self.multidimensional else 0  # Behave as a single Sequence

            try:
                c, st, sh = self.center[axis], self.step[axis], self.shape[axis]
            except IndexError as err:
                raise IndexError(f"Axis range {axis} requested from a Grid of dimension {self.ndim}.")
            rng = c + st * (np.arange(sh) - (sh / 2).astype(np.int))
            if not self._origin_at_center[axis]:
                rng = ft.ifftshift(rng)
            if not self.flat[axis]:
                rng = vector_to_axis(rng, axis=axis, ndim=self.ndim)

            result.append(rng if self.multidimensional else rng[idx])

        if scalar_key:
            result = result[0]  # Unpack again

        return result

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    #
    # General object properties
    #

    @property
    def __dict__(self):
        shape, step, center, flat, center_at_index, origin_at_center = \
            self.shape, self.step, self.center, self.flat, self.center_at_index, self.origin_at_center
        if not self.multidimensional:
            shape, step, center, flat, center_at_index, origin_at_center = \
                shape[0], step[0], center[0], flat[0], center_at_index[0], origin_at_center[0]
        return dict(shape=shape, step=step, center=center, flat=flat,
                    center_at_index=center_at_index, origin_at_center=origin_at_center)

    @property
    def immutable(self):
        """ Return a new immutable Grid object. """
        return Grid(**self.__dict__)

    @property
    def mutable(self):
        """ Return a new MutableGrid object. """
        return MutableGrid(**self.__dict__)

    def __str__(self) -> str:
        arg_desc = ", ".join([f"{k}={str(v)}" for k, v in self.__dict__.items()])
        return f"{type(self).__name__}({arg_desc:s})"

    def __eq__(self, other) -> bool:
        """ Compares two Grid objects. """
        return self.ndim == other.ndim and np.all((self.shape == other.shape) & (self.step == other.step) &
                                                  (self.center == other.center) & (self.flat == other.flat) &
                                                  (self.center_at_index == other.center_at_index) &
                                                  (self.origin_at_center == other.origin_at_center) &
                                                  (self.dtype == other.dtype))

    #
    # Assorted property
    #
    @property
    def multidimensional(self) -> bool:
        """ Single-dimensional grids behave as Sequences, multi-dimensional behave as a Sequence of vectors. """
        return self.__multidimensional

    #
    # Protected and private methods
    #

    def _to_ndim(self, arg) -> np.array:
        """
        Helper method to ensure that all arguments are all numpy vectors of the same length, self.ndim.
        """
        if arg is not None:
            arg = np.array(arg).flatten()
            if np.isscalar(arg) or arg.size == 1:
                arg = np.repeat(arg, repeats=self.ndim)
            elif arg.size != self.ndim:
                raise ValueError(
                    f"All input arguments should be scalar or of length {self.ndim}, not {arg.size} as {arg}.")
        return arg

    def __all_to_ndim(self, *args):
        """
        Helper method to ensures that all arguments are all numpy vectors of the same length, self.ndim.
        """
        return tuple([self._to_ndim(arg) for arg in args])


class MutableGrid(Grid):
    """
    A class representing a mmutable uniformly-spaced plaid grid.

    See also: Grid
    """
    def __init__(self, shape, step=None, extent=None, first=None, center=None, last=None, include_last=False,
                 ndim: int=None,
                 flat: Union[bool, Sequence, np.ndarray]=False,
                 origin_at_center: Union[bool, Sequence, np.ndarray]=True,
                 center_at_index: Union[bool, Sequence, np.ndarray]=True):
        """
        Construct a mutable Grid object.

        :param shape: An integer vector array with the shape of the sampling grid.
        :param step: A vector array with the spacing of the sampling grid.
        :param extent: The extent of the sampling grid as shape * step
        :param first: A vector array with the first element for each dimension.
        The first element is the smallest element if step is positive, and the largest when step is negative.
        :param center: A vector array with the center element for each dimension. The center position in the grid is
        rounded to the next integer index unless center_at_index is set to False for that partical axis.
        :param last: A vector array with the last element for each dimension. Unless include_last is set to True for
        the associated dimension, all but the last element is returned when calling self[axis].
        :param include_last: A boolean vector array indicating whether the returned vectors, self[axis], should include
        the last element (True) or all-but-the-last (False)
        :param ndim: A scalar integer indicating the number of dimensions of the sampling space.
        :param flat: A boolean vector array indicating whether the returned vectors, self[axis], should be
        flattened (True) or returned as an open grid (False)
        :param origin_at_center: A boolean vector array indicating whether the origin should be fft-shifted (True)
        or be ifftshifted to the front (False) of the returned vectors for self[axis].
        :param center_at_index: A boolean vector array indicating whether the center of the grid should be rounded to an
        integer index for each dimension. If False and the shape has an even number of elements, the next index is used
        as the center, (self.shape / 2).astype(np.int).
        """
        super().__init__(shape=shape, step=step, extent=extent, first=first, center=center, last=last,
                         include_last=include_last, ndim=ndim, flat=flat, origin_at_center=origin_at_center,
                         center_at_index=center_at_index)

    @property
    def shape(self) -> np.array:
        return super().shape

    @shape.setter
    def shape(self, new_shape: Union[int, Sequence, np.array]):
        if new_shape is not None:
            self._shape = self._to_ndim(new_shape)

    @property
    def step(self) -> np.ndarray:
        return super().step

    @step.setter
    def step(self, new_step: Union[int, float, Sequence, np.array]):
        self._step = self._to_ndim(new_step)

    @property
    def center(self) -> np.ndarray:
        return super().center

    @center.setter
    def center(self, new_center: Union[int, float, Sequence, np.array]):
        self._center = self._to_ndim(new_center)

    @property
    def flat(self) -> np.array:
        return super().flat

    @flat.setter
    def flat(self, value: Union[bool, Sequence, np.array]):
        self._flat = self._to_ndim(value)

    @property
    def origin_at_center(self) -> np.array:
        return super().origin_at_center

    @origin_at_center.setter
    def origin_at_center(self, value: Union[bool, Sequence, np.array]):
        self._origin_at_center = self._to_ndim(value)

    @property
    def first(self) -> np.ndarray:
        """
        :return: A vector with the first element of each range
        """
        return super().first

    @first.setter
    def first(self, new_first):
        self._center = super().center + self._to_ndim(new_first) - self.first

    def __iadd__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.center += np.asarray(number)

    def __imul__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.step *= np.asarray(number)
        self.center *= np.asarray(number)

    def __isub__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.center -= np.asarray(number)

    def __idiv__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.step /= np.asarray(number)
        self.center /= np.asarray(number)

