from __future__ import annotations

import collections.abc as col
from typing import Union, Sequence
import numpy as np

from macromax.utils.array.vector_to_axis import vector_to_axis
from macromax.utils import ft


class Grid(Sequence):
    def __init__(self, shape, step=None, extent=None, first=None, center=None, last=None, include_last=False,
                 ndim: int=None,
                 flat: Union[bool, Sequence, np.ndarray]=False,
                 origin_at_center: Union[bool, Sequence, np.ndarray]=True,
                 center_at_index: Union[bool, Sequence, np.ndarray]=True):
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
            self.__fix_all(shape, step, extent, first, center, last, flat, origin_at_center, include_last,
                           center_at_index)
        shape = np.round(shape).astype(np.int)  # Make sure that the shape is integer

        if shape is None:
            if step is not None and extent is not None:
                shape = np.round(np.real(self.__fix(extent) / self.__fix(step))).astype(np.int)
            else:
                shape = self.__fix(1)  # Default shape to 1
            shape += include_last
        # At this point the shape is not None
        if step is None:
            if extent is not None:
                nb_steps = shape - include_last
                step = extent / nb_steps
            else:
                step = self.__fix(1)  # Default step size to 1
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
                center = self.__fix(0)  # Center around 0 by default
        # At this point the center is not None

        self.__shape = shape
        self.__step = step
        self.__center = center
        self.__flat = flat
        self.__origin_at_center = origin_at_center
        self.__center_at_index = center_at_index

    @staticmethod
    def from_ranges(*ranges: Union[int, float, complex, Sequence, np.ndarray]) -> Grid:
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
        return self.__ndim

    @property
    def shape(self) -> np.array:
        return self.__shape

    @shape.setter
    def shape(self, new_shape: Union[int, col.Sequence, np.array]):
        if new_shape is not None:
            self.__shape = self.__fix(new_shape)

    @property
    def step(self) -> np.ndarray:
        return self.__step

    @step.setter
    def step(self, new_step: Union[int, float, col.Sequence, np.array]):
        self.__step = self.__fix(new_step)

    @property
    def center(self) -> np.ndarray:
        return self.__center

    @center.setter
    def center(self, new_center: Union[int, float, col.Sequence, np.array]):
        self.__center = self.__fix(new_center)

    @property
    def center_at_index(self) -> np.array:
        return self.__center_at_index

    @property
    def flat(self) -> np.array:
        return self.__flat

    @flat.setter
    def flat(self, value: Union[bool, col.Sequence, np.array]):
        self.__flat = self.__fix(value)

    @property
    def origin_at_center(self) -> np.array:
        return self.__origin_at_center

    @origin_at_center.setter
    def origin_at_center(self, value: Union[bool, col.Sequence, np.array]):
        self.__origin_at_center = self.__fix(value)
    
    #
    # Conversion methods
    #

    @property
    def as_flat(self) -> Grid:
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
    def as_non_flat(self) -> Grid:
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
    def as_origin_at_0(self) -> Grid:
        """
        :return: A new Grid object where all the ranges are ifftshifted so that the origin as at index 0.
        """
        shape, step, center, center_at_index, flat = self.shape, self.step, self.center, self.center_at_index, self.flat
        if not self.multidimensional:
            shape, step, center, center_at_index, flat = shape[0], step[0], center[0], center_at_index[0], flat[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=flat, origin_at_center=False)

    @property
    def as_origin_at_center(self) -> Grid:
        """
        :return: A new Grid object where all the ranges have the origin at the center index, even when the number of
        elements is odd.
        """
        shape, step, center, center_at_index, flat = self.shape, self.step, self.center, self.center_at_index, self.flat
        if not self.multidimensional:
            shape, step, center, center_at_index, flat = shape[0], step[0], center[0], center_at_index[0], flat[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=flat, origin_at_center=True)

    def swapaxes(self, axes: Union[slice, Sequence, np.array]) -> Grid:
        axes = np.array(axes).flatten()
        all_axes = np.arange(self.ndim)
        all_axes[axes] = axes[::-1]
        return self.transpose(all_axes)

    def transpose(self, axes: Union[None, slice, Sequence, np.array]=None) -> Grid:
        if axes is None:
            axes = np.arange(self.ndim-1, -1, -1)
        return self.project(axes)

    def project(self, axes: Union[int, slice, Sequence, np.array]=0) -> Grid:
        if isinstance(axes, slice):
            axes = np.arange(self.ndim)[axes]
        if np.isscalar(axes):
            axes = [axes]
        axes = np.array(axes)

        if np.any(axes >= self.ndim) or np.any(axes < -self.ndim):
            raise IndexError(f"Axis range {axes} requested from a Grid of dimension {self.ndim}.")

        return Grid(shape=self.shape[axes], step=self.step[axes], center=self.center[axes], flat=self.flat[axes],
                    origin_at_center=self.origin_at_center[axes],
                    center_at_index=self.center_at_index[axes]
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
        return self.__center - self.step * half_shape

    @first.setter
    def first(self, new_first):
        self.__center += new_first - self.first

    @property
    def extent(self) -> np.ndarray:
        return self.shape * self.step

    #
    # Sequence methods
    #

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def dtype(self):
        return (self.step[0] + self.center[0]).dtype

    #
    # Frequency grids
    #

    @property
    def f(self) -> Grid:
        shape, step, flat = self.shape, 1 / self.extent, self.flat
        if not self.multidimensional:
            shape, step, flat = shape[0], step[0], flat[0]

        return Grid(shape=shape, step=step, flat=flat, origin_at_center=False, center_at_index=True)

    @property
    def k(self) -> Grid:
        return self.f * (2 * np.pi)

    #
    # Arithmetic methods
    #
    def __add__(self, other) -> Grid:
        result = self.copy()
        result.center = result.center + np.array(other)
        return result

    def __mul__(self, factor: Union[int, float, complex, Sequence, np.array]) -> Grid:
        """
        Scales all ranges with a factor.
        :param factor: A scalar factor for all dimensions, or a vector of factors, one for each dimension.
        :return: A new scaled Grid object.
        """
        if isinstance(factor, Grid):
            raise TypeError("A Grid object can't be multiplied with a Grid object."
                            + "Use matmul @ to determine the tensor space.")
        result = self.copy()
        result.step = result.step * np.array(factor)
        result.center = result.center * np.array(factor)
        return result

    def __matmul__(self, other: Grid) -> Grid:
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

    def __sub__(self, other: Union[int, float, complex, Sequence, np.ndarray]) -> Grid:
        result = self.copy()
        result.center = result.center - np.array(other)
        return result

    def __truediv__(self, other: Union[int, float, complex, Sequence, np.ndarray]) -> Grid:
        result = self.copy()
        result.step = result.step / np.array(other)
        result.center = result.center / np.array(other)
        return result

    def __iadd__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.center += np.array(number)

    def __imul__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.step *= np.array(number)
        self.center *= np.array(number)

    def __isub__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.center -= np.array(number)

    def __idiv__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.step /= np.array(number)
        self.center /= np.array(number)

    def __neg__(self):
        self.__imul__(-1)

    #
    # iterator methods
    #

    def __len__(self) -> int:
        if self.multidimensional:
            return self.ndim
        else:
            return self.shape[0]  # Behave as a single Sequence

    def __getitem__(self, key: Union[int, slice, Sequence]):
        # if self.multidimensional:
        #     return self.project(key)
        # else:
        #     rng = self.center[0] + self.step[0] * (np.arange(self.shape[0]) - (self.shape[0] / 2).astype(np.int))
        #     if not self.__origin_at_center[0]:
        #         rng = ft.ifftshift(rng)
        #     if not self.flat[0]:
        #         rng = vector_to_axis(rng, axis=0, ndim=self.ndim)
        #todo: finish the above to replace project with indexing?

        scalar = np.isscalar(key)
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
            if not self.__origin_at_center[axis]:
                rng = ft.ifftshift(rng)
            if not self.flat[axis]:
                rng = vector_to_axis(rng, axis=self.ndim, ndim=axis)

            result.append(rng if self.multidimensional else rng[idx])

        if scalar:
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

    def copy(self) -> Grid:
        return Grid(**self.__dict__)

    def __str__(self) -> str:
        arg_desc = ", ".join([f"{k}={str(v)}" for k, v in self.__dict__.items()])
        return f"Grid({arg_desc:s})"

    def __eq__(self, other) -> bool:
        return self.ndim == other.ndim and np.all(self.shape == other.shape & self.step == other.step &
                                                  self.center == other.center & self.flat == self.flat &
                                                  self.center_at_index & other.center_at_index & self.origin_at_center &
                                                  other.origin_at_center)

    #
    # Assorted property
    #
    @property
    def multidimensional(self) -> bool:
        return self.__multidimensional

    #
    # Private methods
    #

    def __fix(self, arg) -> np.array:
        """
        Helper method to ensures that all arguments are all numpy vectors of the same length, self.ndim.
        """
        if arg is not None:
            arg = np.array(arg).flatten()
            if np.isscalar(arg) or arg.size == 1:
                arg = np.repeat(arg, repeats=self.ndim)
            elif arg.size != self.ndim:
                raise ValueError(
                    f"All input arguments should be scalar or of length {self.ndim}, not {arg.size} as {arg}.")
        return arg

    def __fix_all(self, *args):
        """
        Helper method to ensures that all arguments are all numpy vectors of the same length, self.ndim.
        """
        return tuple([self.__fix(arg) for arg in args])
