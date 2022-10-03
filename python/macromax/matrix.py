from __future__ import annotations

import numpy as np
from typing import Union, Sequence, Callable, Optional
from numbers import Complex, Real
import scipy.constants as const
from scipy.sparse.linalg import LinearOperator
import logging

from . import Solution
from .utils import ft
from macromax.utils.ft.grid import Grid
from macromax.bound import Bound, PeriodicBound

log = logging.getLogger(__name__)

array_like = Union[Complex, Sequence, np.ndarray, LinearOperator]


class CachingMatrix(object):
    def __init__(self, caching: bool = True):
        """
        A mixin for Matrices that can cache the output

        :param caching: Cache field propagation calculations. By default, the results are cached for multiplications
            with basis vectors. Numerical errors might accumulate for certain superpositions. Setting this property to
            False will ensure that field propagations are always used and the constructor argument array is ignored.
        """
        self.__caching: bool = caching
        self.__cached_columns = dict()  # Stores a column vector for each integer column index that has been calculated.

    @property
    def caching(self) -> bool:
        """
        When set to True, this object uses cached values instead of propagating the field through the scatterer.
        Otherwise, field propagation is used for all matrix operations. This can help avoid the accumulation of
        numerical errors.
        """
        return self.__caching  # If True, stores the matrix columns in the dictionary self.__cached_columns

    @caching.setter
    def caching(self, new_value: bool):
        self.__caching = new_value

    def _cache(self, right: array_like, value_function: Optional[Callable[[array_like], np.ndarray]] = None,
               out: Optional[array_like] = None) -> np.ndarray:
        """
        Retrieve a cached value or calculate it using the specified function as necessary.
        Usage: ```left = self._cache(right, calc_left)```

        :param right: The argument value.
        :param value_function: The function with the argument ```right```, returning a vector of values
        :param out: The optional destination array.
        :return: The result of value_function(right), potentially retrieved from the cache.
        """
        right = np.asarray(right)
        input_shape = right.shape
        right = right.reshape(-1, right.shape[-1] if right.ndim > 1 else 1)  # Make into a 2D array

        zero_columns = []
        for column_idx in range(right.shape[1]):
            right_col = right[:, column_idx]
            non_zeros = right_col != 0
            nb_non_zeros = np.sum(non_zeros)
            if nb_non_zeros > 0:
                basis_index = None
                if self.caching and nb_non_zeros == 1:
                    basis_index = np.where(non_zeros)[0][0]
                if not self.caching or basis_index not in self.__cached_columns:
                    # Calculate this column
                    result_col = value_function(right_col).ravel()
                    if out is None:
                        out = np.empty([result_col.shape[0], right.shape[1]], dtype=result_col.dtype)
                    out[:, column_idx] = result_col
                    # If requested and useful, cache the result
                    if self.caching and basis_index is not None:
                        self.__cached_columns[basis_index] = out[:, column_idx] / right_col[basis_index]  # Make sure to not return a reference to the internal cache!
                else:
                    out[:, column_idx] = self.__cached_columns[basis_index] * right_col[basis_index]  # Make sure to npt return a reference to the internal cache!
            else:
                zero_columns.append(column_idx)
        if out is None:
            # All-zero right-hand side! An alternative method is required to determine the output shape and dtype
            if len(self.__cached_columns) > 0:  # Use a cached column
                result_col = self.__cached_columns.values()[0]
            else:  # Do a calculation anyway, even if we know that the result will be all zeros.
                result_col = value_function(right[:, 0]).ravel()
            out = np.zeros([result_col.shape[0], right.shape[1]], dtype=result_col.dtype)
        else:
            out[:, zero_columns] = 0.0  # The right-hand side was all zeros in these columns

        return out


class Matrix(LinearOperator):
    """
    A class to represent rectangular or square matrices that can be multiplied from the left or right, and pseudo-inverted.
    """
    def __init__(self, array: Optional[array_like] = None, shape: Optional[Sequence[int]] = None, dtype=np.complex128):
        """
        Constructs a matrix from a rectangular numpy.ndarray, array-like object, or a function or method that returns one.

        :param array: A sequence or numpy.ndarray of complex numbers representing the matrix, or a function that returns one.
            If not specified, shape must be specified and _matmul and _rmatmul can be implemented in a subclass.
        :param shape: The optional shape of the matrix, i.e. the number of rows and columns.
        :param dtype: The optional dtype of the matrix elements.
        """
        if shape is None and array is not None:
            shape = np.asarray(array).shape
        super().__init__(shape=shape, dtype=dtype)
        self.__array = array

    def __len__(self) -> int:
        """
        :return: The number of rows in the matrix as an integer.
        """
        return self.shape[0]

    def __getitem__(self, item: int) -> np.ndarray:
        """
        Returns a row of the matrix

        :param item: The integer row index, a slice, an np.newaxis, or a tuple thereof.

        :return: An ```numpy.ndarray``` of shape ```(1, self.shape[1])``` and dtype ```self.dtype``` with the row elements.
        """
        if not isinstance(item, tuple):
            item = (item, slice(None))

        new_dims = []
        slices = []
        dropped_axes = 0
        for _, s in enumerate(item):
            if s is not np.newaxis:
                slices.append(s)
                if np.isscalar(s):
                    dropped_axes += 1
            else:
                new_dims.append(_ - dropped_axes)
        if len(slices) > 2:
            raise TypeError(f'Cannot index {len(slices)} in a 2 dimensional Matrix!')
        if len(slices) > 1:
            right = np.eye(self.shape[0])[:, slices[1]]
            arr = self @ right
            arr = arr[slices[0]]
            arr = np.expand_dims(arr, new_dims)
        else:
            arr = self.__array__()[item]
        return arr

    def __setitem__(self, key, value):
        """
        Update (part of) the matrix.

        :param key: Index or slice.
        :param value: The new value.
        """
        self.__array.__setitem__(key, value)

    def __array__(self) -> np.ndarray:
        """
        :return: The array values represented by this matrix.
        """
        if self.__array is not None:
            return np.asarray(self.__array)
        else:
            # Multiply by eye without forming an extra matrix
            full_array = np.zeros(self.shape[::-1], self.dtype)
            vec = np.zeros(self.shape[1], self.dtype)
            for _ in range(len(vec)):
                vec[_] = 1
                full_array[_] = self @ vec
                vec[_] = 0
            return full_array.T

    def _matvec(self, right: array_like) -> np.ndarray:
        """
        Multiply this matrix with a vector or matrix: S @ right
        This is the lowest level operation on which @ and __array__() depend.

        :param: A vector or matrix to right-multiply with the MxN matrix, specified as either an array of shape (N, ) or (N, P).
            Alternatively, an nd-array can be specified of shape (2, N//2, Q) or (2, N//2, P, Q), where the first axis indicates the side
            (front, back), the second axis indicates the input mode number, and the third axis indicates the polarization (V, H).

        :return: The result vector or matrix of shape as the input parameter except for the dimension with N, which becomes M.
        """
        return self.__array @ right

    def _adjoint(self) -> np.ndarray:
        """
        Return the Hermitian transpose of this matrix.
        """
        return self.__array__().conj().transpose()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.__array__()}, side={self.shape[0]}, dtype={self.dtype})'

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(side={self.shape[0]}, dtype={self.dtype})'

    def inv(self, noise_level: float = 0.0) -> np.ndarray:
        """
        The (Tikhonov regularized) inverse of the scattering matrix.

        :param noise_level: (optional) argument to regularize the inversion of a (near-)singular matrix.

        :return: An nd-array with the inverted matrix so that ```self @ self.inv``` approximates the identity.
        """
        return inv(self, noise_level=noise_level)


class SquareMatrix(Matrix):
    """
    A class to represent square matrices that can be inverted with or without regularization.
    """
    def __init__(self, array: Optional[array_like] = None, side: Optional[int] = None, dtype=np.complex128):
        """
        Constructs a matrix from a square array, array-like object, or a function or method that returns one.

        :param array: A sequence or numpy.ndarray of complex numbers representing the matrix, or a function that returns one.
            If not specified, the identity matrix is assumed.
        :param side: The side of the matrix, i.e. the number of rows or columns.
        :param dtype: The optional dtype of the matrix elements.
        """
        if array is not None:
            array = np.asarray(array)
            if side is None:
                side = array.shape[0]
            if array.ndim < 2 or array.shape[1] != side:
                raise ValueError(f'array should be {side}x{side} square but it has shape {array.shape}.')

        super().__init__(array=array, shape=(side, side), dtype=dtype)

    @property
    def side(self) -> int:
        return self.shape[0]


class LiteralScatteringMatrix(SquareMatrix):
    """
    A class to represent scattering matrices constructed from an array of complex numbers.
    """
    def __init__(self, array: Optional[array_like] = None, side: Optional[int] = None, dtype=np.complex128):
        """
        Constructs a scattering matrix from a square array, array-like object, or a function or method that returns one.

        :param array: A sequence or numpy.ndarray of complex numbers representing the matrix, or a function that returns one.
            If not specified, the identity matrix is assumed.
        :param side: The side of the matrix, i.e. the number of rows or columns.
        :param dtype: The optional dtype of the matrix elements.
        """
        super().__init__(array=array, side=side, dtype=dtype)

    def transfer(self, noise_level: float = 0.0) -> np.ndarray:
        """
        Calculates the transfer matrix, relating one side of the scatterer to the other side (top, bottom). Each side
        can have incoming and outgoing waves. This is in contrast to the scattering matrix, ```self.__array__```, which
        relates incoming waves from both sides to outgoing waves from both sides. One can be calculated from the other
        using the ```matrix.convert()``` function, though this calculation may be ill-conditioned (sensitive to noise).
        Therefore the optional argument ```noise_level``` should be used to indicate the root-mean-square expectation
        value of the measurement error. This avoids divisions by near-zero values and obtains a best estimate using
        Tikhonov regularization.

        :param noise_level: (optional) argument to regularize the inversion of a (near) singular backwards transmission matrix.

        :return: An nd-array with the transfer matrix relating top-to-bottom instead of in-to-out. This can be converted
            back into a scattering matrix using the ```matrix.convert()``` function.

        The first half of the vector inputs and outputs to the scattering and transfer matrices represent fields
        propagating forward along the positive propagation axis (0) and the second half represents fields propagating
        backward along the negative direction.

        Notation:
            p: positive propagation direction along propagation axis 0
            n: negative propagation direction along propagation axis 0
            i: inwards propagating (from source on either side)
            o: outwards propagating (backscattered or transmitted)

        Scattering matrix equation (in -> out):
            [po] = [A, B] [pi]
            [no]   [C, D] [ni]

        Transfer matrix equation (top -> bottom):
            [po] = [A - B inv(D) C,  B inv(D)] [pi]
            [ni] = [  -   inv(D) C     inv(D)] [no],
        where inv(D) is the (regularized) inverse of D.
        """
        return convert(self, noise_level=noise_level)

    @property
    def forward_transmission(self) -> ForwardTransmissionMatrix:
        """
        Select the forward-transmitted quarter of the scattering matrix.
        It indicates how the light coming from negative infinity is transmitted to positive infinity.

        :return: The forward-transmission matrix of shape ```self.shape // 2```.
        """
        return ForwardTransmissionMatrix(self)

    @property
    def front_reflection(self) -> FrontReflectionMatrix:
        """
        Select the quarter of the scattering matrix corresponding to the light that is reflected of the front.
        It indicates how the light coming from negative infinity is back reflected to negative infinity.

        :return: The front-reflection matrix of shape ```self.shape // 2```.
        """
        return FrontReflectionMatrix(self)

    @property
    def back_reflection(self) -> BackReflectionMatrix:
        """
        Select the quarter of the scattering matrix corresponding to the light that is reflected of the back.
        It indicates how the light coming from positive infinity is back reflected to positive infinity.

        :return: The back-reflection matrix of shape ```self.shape // 2```.
        """
        return BackReflectionMatrix(self)

    @property
    def backward_transmission(self) -> BackwardTransmissionMatrix:
        """
        Select the backward-transmitted quarter of the scattering matrix.
        It indicates how the light coming from positive infinity is transmitted to negative infinity.

        :return: The backward-transmission matrix of shape ```self.shape // 2```.
        """
        return BackwardTransmissionMatrix(self)


class ScatteringMatrix(LiteralScatteringMatrix):
    """A class representing scattering matrices."""
    def __init__(self, grid: Union[Grid, Sequence, np.ndarray], vectorial: Optional[bool] = True,
                 wavenumber: Optional[Real] = None, angular_frequency: Optional[Real] = None, vacuum_wavelength: Optional[Real] = None,
                 epsilon: Optional[array_like] = None, xi: Optional[array_like] = 0.0, zeta: Optional[array_like] = 0.0, mu: Optional[array_like] = 1.0,
                 refractive_index: Optional[array_like] = None,
                 bound: Bound = None, dtype=None,
                 callback: Callable = lambda s: s.iteration < 1e4 and s.residue > 1e-6,
                 caching: bool = True, array: Optional[array_like] = None):
        """
        Construct a scattering matrix object for a medium specified by a refractive index distribution or the
        corresponding epsilon, xi, zeta, and mu distributions. Each electromagnetic field distribution entering the
        material is scattered into a certain electromagnetic field distributions propagating away from it from both
        sides. The complex matrix relates the amplitude and phase of all N propagating input modes to all N propagating
        output modes. No scattering, as in vacuum, is indicated by the NxN identity matrix. There are N/2 input and
        output modes on either side of the scattering material. Mode i and i+N/2 correspond to plane wave traveling in
        opposing directions. The mode directions are taken in raster-scan order and only propagating modes are included.
        When polarization is considered, the modes come in pairs corresponding to two orthogonal linear polarizations.

        The modes are encoded as a vector of length N = 2xMxP for 2 sides, M angles, and P polarizations.

        - First the N/2 modes propagating along the positive x-axis are considered, then those propagating in the reverse direction.
        - In each direction, M different angles (k-vectors) can be considered. We choose propagating modes on a
            uniformly-spaced plaid grid that includes the origin (corresponding to the k-vector along the x-axis). Modes
            not propagating along the x-axis, i.e. in the y-z-plane are not considered. The angles are ordered in
            raster-scan order from negative k_y to positive k_y (slow) and from negative k_z to positive k_z (fast).
            The grid axes dimensions correspond to x(0), y(1), z(2).
        - When polarization is considered, each angle has a pair of modes, one for each polarization. The first mode has
            the polarization oriented along the rotated y'-axis and the second mode along the rotated z'-axis. To avoid
            ambiguity for normal incidence, the Cartesian-coordinate system is rotated along the shortest possible path,
            i.e. along the axis that is normal to the original x-axis and the mode's k-vector. All rotations are around the
            origin of the coordinate system, incurring no phase shift there.

        Vectors can be converted to field distributions on the complete grid using the methods:
        - ```srcvec2freespace```: super-position of free-space plane waves in the whole volume (fast)
        - ```srcvec2source```: super-position of free-space plane waves at the source planes at the front and back (fast)
        - ```source2detfield```: calculate the field in the whole volume using the ```solver.Solution``` object (slow)
        - ```detfield2detvec```: vector corresponding to the detected field at the detection planes (fast). The fields
            at those planes should only contain the outward propagating waves. Hence, inwards propagating waves should
            be subtracted before using this method!
        - ```srcvec2detfield```: calculate the field in the whole volume and convert it to a detection vector (slow)
        The latter is used in the matrix multiplication method: ```matmul```, @

        :param grid: A Grid object or a Sequence of vectors with uniformly increasing values that indicate the positions
            in a plaid grid of sample points for the material and solution. In the one-dimensional case, a simple increasing
            Sequence of uniformly-spaced numbers may be provided as an alternative. The length of the ranges determines the
            data_shape, to which the source_distribution, epsilon, xi, zeta, mu, and initial_field must broadcast when
            specified as ndarrays.
        :param vectorial: a boolean indicating if the source and solution are 3-vectors-fields (True) or scalar fields (False).
        :param wavenumber: the wavenumber in vacuum = 2 pi / vacuum_wavelength.
            The wavelength in the same units as used for the other inputs/outputs.
        :param angular_frequency: alternative argument to the wavenumber = angular_frequency / c
        :param vacuum_wavelength: alternative argument to the wavenumber = 2 pi / vacuum_wavelength
        :param epsilon: an array or function that returns the (tensor) epsilon that represents the permittivity at
            the points indicated by the grid specified as its input arguments.
        :param xi: an array or function that returns the (tensor) xi for bi-(an)isotropy at the
            points indicated by the grid specified as its input arguments.
        :param zeta: an array or function that returns the (tensor) zeta for bi-(an)isotropy at the
            points indicated by the grid specified as its input arguments.
        :param mu: an array or function that returns the (tensor) permeability at the
            points indicated by the grid specified as its input arguments.
        :param refractive_index: an array or function that returns the (tensor) refractive_index = np.sqrt(permittivity)
            at the points indicated by the `grid` input argument.
        :param bound: An object representing the boundary of the calculation volume. Default: None, PeriodicBound(grid)
        :param dtype: optional numpy datatype for the internal operations and results. This must be a complex number
            type as numpy.complex128 or np.complex64.
        :param callback: optional function that will be called with as argument this solver.
            This function can be used to check and display progress. It must return a boolean value of True to
            indicate that further iterations are required.
        :param caching: Cache field propagation calculations. By default, the results are cached for multiplications
            with basis vectors. Numerical errors might accumulate for certain superpositions. Setting this property to
            False will ensure that field propagations are always used and the constructor argument array is ignored.
        :param array: Optional in case the matrix values have been calculated before and stored. If not specified, the
            matrix is calculated from the material properties. I specified, this must be a sequence or numpy.ndarray of
            complex numbers representing the matrix, or a function that returns one.

        :return: The Solution object that has the E and H fields, as well as iteration information.
        """
        self.__grid = grid
        self.__vectorial = vectorial
        self.__callback = callback

        # Define the solver that will be recycled for every input vector
        self.__solution = Solution(grid=self.grid, vectorial=self.vectorial,
                                   wavenumber=wavenumber, angular_frequency=angular_frequency, vacuum_wavelength=vacuum_wavelength,
                                   epsilon=epsilon, xi=xi, zeta=zeta, mu=mu, refractive_index=refractive_index, bound=bound,
                                   dtype=dtype)

        # Determine the independent input and output modes that are propagating in raster-scan order
        propagating = np.ones(1, dtype=bool)
        k_abs = self.__solution.wavenumber  # The maximum absolute value of the remaining k-vector components for these to be propagating. Start with k0.
        k_grid = self.grid.k.as_origin_at_center
        tol = np.sqrt(np.finfo(dtype).eps)
        for k in k_grid[range(1, self.grid.ndim)]:  # fftshifted k-space coordinate
            propagating = propagating & (np.abs(k) < k_abs - tol)  # only modes propagating into the material
            k_abs = np.sqrt(np.maximum(k_abs**2 - k**2, 0.0))
        # self.__mode_direction_indices lists the indices in the transverse k-space for each mode in the order they
        # appear in the vector, only considering forward propagating and a single polarization.
        shifted_indices = np.where(propagating.ravel())[0]  # indices in fftshifted version
        if self.grid.ndim > 1:
            subs = (np.asarray(np.unravel_index(shifted_indices, self.grid.shape[1:]))
                    - (self.grid.shape[1:, np.newaxis]//2)) % self.grid.shape[1:, np.newaxis]
            self.__mode_direction_indices = np.ravel_multi_index(subs, self.grid.shape[1:])  # indices in ifftshifted version so that k-space is raster scanned from negative to positive
        else:
            self.__mode_direction_indices = shifted_indices  # just a 0

        # Determine the indices of the source and detection planes on both sides of the scatterer
        if bound is None:
            bound = PeriodicBound(grid)
        front_source_plane = int(bound.thickness[0, 0] / self.grid.step[0]) + 1
        back_source_plane = self.grid.shape[0] - int(bound.thickness[0, -1] / self.grid.step[0]) - 1
        self.__source_planes = (front_source_plane, back_source_plane)
        # Detectors placed closer to scattering sample
        front_detector_plane = int((bound.thickness[0, 0] + self.__solution.wavelength) / grid.step[0]) + 1
        back_detector_plane = self.grid.shape[0] - int((bound.thickness[0, -1] + self.__solution.wavelength) / grid.step[0]) - 1
        nb_detection_layers = 1  # int(self.__solution.wavelength / self.grid.step[0] + 0.5)  # TODO: Why doesn't this improve accuracy?
        self.__detector_volumes_indices = [_ + ft.ifftshift(np.arange(nb_detection_layers) - nb_detection_layers // 2)
                                           for _ in (back_detector_plane, front_detector_plane)]  # swap order so the empty space results in the identity

        self.__caching: bool = caching
        self.__cached_columns = dict()  # Stores a column vector for each integer column index that has been calculated.
        nb_sides = 2
        nb_independent_polarizations = 1 + self.vectorial
        matrix_side = nb_sides * nb_independent_polarizations * self.__mode_direction_indices.size
        if array is not None:
            if np.any(matrix_side != np.array(array.shape)):
                raise ValueError(f"Number of modes {matrix_side} is different from the number of rows or columns of the specified matrix {array.shape}!")
            # Store as a dictionary of cached columns
            for column_index, column in enumerate(array.transpose()):
                self.__cached_columns[column_index] = column

        super().__init__(side=matrix_side, dtype=self.__solution.dtype)

    @property
    def grid(self) -> Grid:
        """The calculation grid."""
        return self.__grid

    @property
    def vectorial(self) -> bool:
        """Boolean to indicates whether calculations happen on polarized (True) or scalar (False) fields."""
        return self.__vectorial

    @property
    def caching(self) -> bool:
        """
        When set to True, this object uses cached values instead of propagating the field through the scatterer.
        Otherwise, field propagation is used for all matrix operations. This can help avoid the accumulation of
        numerical errors.
        """
        return self.__caching  # If True, stores the matrix columns in the dictionary self.__cached_columns

    @caching.setter
    def caching(self, new_value: bool):
        self.__caching = new_value

    def __norm(self, vec: array_like, axis: int = 0) -> np.ndarray:
        """Normalizes all vectors in an array, avoiding division by 0."""
        vec_norm = np.linalg.norm(vec, axis=axis)
        return vec / (vec_norm + (vec_norm == 0))

    def __dot(self, a: array_like, b: array_like) -> np.ndarray:
        """Dot product along the left-most dimension."""
        return np.einsum('i...,i...->...', a, b)

    def srcvec2freespace(self, input_vector: array_like) -> np.ndarray:
        """
        Convert an input source vector to a superposition of plane waves at the origin and spreading over the whole volume.
        The input vector specifies the propagating modes in the far-field (the inverse Fourier transform of the fields
        at the sample origin). Incident waves at an angle will result in higher amplitudes to compensate for the reduction
        in propagation along the propagation axis through the entrance plane.

        Used in ```ScatteringMatrix.srcvec2source``` to calculate the source field distribution before entering the scatterer.

        Used in ```ScatteringMatrix.__matmul__``` to distinguish incoming from back-scattered light.

        :param input_vector: A source vector or array of shape [2, M, P], where the first axis indicates the side (front, back),
            the second axis indicates the propagation mode (direction, top-bottom-left-right), and the final axis indicates
            the polarization (1 for scalar, 2 for polarized: V-H).

        :return: An nd-array with the field on the calculation grid. Its shape is (1, *self.grid.shape) for scalar
        calculations and (3, *self.grid.shape) for vectorial calculations with polarization.
        """
        # Convert vector to an nd-array
        nb_sides = 2
        input_vector = np.array(input_vector, dtype=self.dtype).reshape([nb_sides, self.__mode_direction_indices.size,
                                                                         1 + self.vectorial])  # TODO: standardize on [side, pol, transverse] instead?
        input_vector = np.moveaxis(input_vector, -1, 1)  # New order: [side, polarization, transverse_mode]

        # Adjust magnitude to get unity E-fields
        # input_vector *= np.prod(self.grid.shape[1:])
        # Define the output shape
        out = np.zeros([1 + 2 * self.vectorial, *self.grid.shape], dtype=self.dtype)

        # Determine the 3D k-vectors in a 2D slice and normalize
        k_vector_dir = np.zeros([3, *self.grid.shape[1:]])
        for _ in range(1, self.grid.ndim):
            k_vector_dir[_] = self.grid.k[_] / self.__solution.wavenumber
        k_vector_dir = k_vector_dir.reshape([k_vector_dir.shape[0], -1])[:, self.__mode_direction_indices]  # select only the propagating modes
        k_vector_dir[0] = np.sqrt(np.maximum(1 - sum(k_vector_dir[1:]**2), 0.0))  # forward propagating
        if self.vectorial:
            # The axial direction
            axial_dir = np.array([1, 0, 0])[:, np.newaxis]
            # The radial direction, orthonormal to the k-vector and pointing along the negative propagation axis
            radial_dir = self.__norm(self.__dot(k_vector_dir, axial_dir) * k_vector_dir - axial_dir)
            radial_mode_dir = self.__norm(k_vector_dir[1:])  # The mode-space (2D) radial direction

        # Compensate for angle of incidence. The entrance aperture is effectively smaller for waves that enter at an angle,
        # so they should be brighter to get the same field values.
        input_vector /= np.sqrt(k_vector_dir[0])  # TODO: Why square root?

        # Add the fields emanating from the front (forward to positive) and back (backwards to negative)
        for input_vector_single_side, direction in zip(input_vector, [1, -1]):
            if not np.allclose(input_vector_single_side, 0.0):  # skip sides that don't have a source
                if direction < 0:  # This conjugation is undone in the final step when the time-dimension is flipped.
                    input_vector_single_side = input_vector_single_side.conj()
                if self.vectorial:  # Rotate polarization as k-vector with respect to dim 0 (propagation)
                    radial_mode_norm = self.__dot(input_vector_single_side, radial_mode_dir)  # The radial component
                    # Rotate the polarisation from the 2D transverse into 3D space following each k-vector
                    field_ift_per_mode = radial_mode_norm * radial_dir  # The rotated radial component, and add the azimuthal next:
                    field_ift_per_mode[1:] += input_vector_single_side - radial_mode_norm * radial_mode_dir  # The azimuthal component does not rotate
                else:  # Scalar: nothing to be done
                    field_ift_per_mode = input_vector_single_side  # [polarization, transverse-mode]

                # Shift in Inverse Fourier space so that we can use a regular fft for arbitrary origin positions.
                field_ift_per_mode *= np.exp(1j * self.__dot(self.__solution.wavenumber * k_vector_dir[1:self.grid.ndim], self.grid.first[1:, np.newaxis]))

                # Move the polarization one axis further left and extend along dimension 0 (propagation) by Fourier transform
                field_ift_per_mode = field_ift_per_mode[:, np.newaxis, :] \
                                     * np.exp(1j * self.__solution.wavenumber * k_vector_dir[0] * self.grid[0].ravel()[:, np.newaxis])  # adds phase offset with propagation distance for each individual k-vector

                # Assign the propagating modes to the grid
                field_ift = np.zeros([1 + 2 * self.vectorial, self.grid.shape[0], np.prod(self.grid.shape[1:])], dtype=self.dtype)
                field_ift[:, :, self.__mode_direction_indices[::-1]] = field_ift_per_mode  # The second dimension will be replicated below
                field_ift = field_ift.reshape([1 + 2 * self.vectorial, *self.grid.shape])

                if self.grid.ndim > 1:
                    # Fourier transform along higher dimensions (at origin of dim 0)
                    field = ft.fftn(field_ift, axes=1 + np.arange(1, self.grid.ndim))
                else:
                    field = field_ift  # Nothing to be done
                if direction < 0:
                    field = field.conj()  # flip left-right & up-down in real space through Fourier transform

                # Combine solutions for each side
                out += field

        return out

    def srcvec2source(self, input_vector: array_like, out: np.ndarray = None) -> np.ndarray:
        """
        Converts a source vector into an (N+1)D-array with the source field at the front and back of the scatterer.
        The source field is such that it produces E-fields of unit intensity for unit vector inputs.

        Used in ```self.vector2field()``` and ```self.__matmul__()```.

        :param input_vector: A source vector with ```self.shape[1]``` elements.
            One value per side, per independent polarization (2), and per mode (inwards propagating k-vectors only).

        :param out: (optional) numpy array to store the result.

        :return: The field distribution as an array of shape [nb_pol, *self.grid], where nb_pol = 3 for a vectorial
            calculation and 1 for a scalar calculation.
        """
        # Reshape input for convenience
        nb_sides = 2
        source_vector = np.array(input_vector, dtype=self.dtype).reshape([nb_sides, self.__mode_direction_indices.size, 1 + self.vectorial])  # TODO: standardize on [side, pol, transverse] instead?
        # Determine the current density to get unity E-field plane waves
        current_density_amplitude = 1 / (const.c * const.mu_0 * self.grid.step[0] / 2)  # [ A m^-2 ], to get unit E-field
        source_amplitude = (-1.0j * self.__solution.angular_frequency * const.mu_0) * current_density_amplitude  # a scalar
        # If the source is at an angle to the propagation-direction, we need to ensure that the current density is appropriate
        k_transverse_norm = np.sqrt(sum(self.grid.k[_]**2 for _ in range(1, self.grid.ndim)))[0] \
            if self.grid.ndim > 1 else np.zeros(1)
        k_transverse_sin = k_transverse_norm.ravel()[self.__mode_direction_indices] / self.__solution.wavenumber  # a vector with only the propagating modes
        k_transverse_cos = np.sqrt(np.maximum(1.0 - k_transverse_sin**2, 0.0))  # cosine of the plane wave with the x-y-plane.
        source_vector *= (k_transverse_cos * source_amplitude)[:, np.newaxis]  # scale the input vector to get current values
        # Allocate result space if needed
        if out is None:
            out = np.zeros((1 + 2 * self.vectorial, *self.grid.shape), dtype=self.dtype)
        # Place the fields at the correct planes in the calculation grid
        for side_idx in range(nb_sides):
            vector_mask = (np.arange(nb_sides) == side_idx)[:, np.newaxis, np.newaxis]  # mask the other side
            out[:, self.__source_planes[side_idx]] = \
                self.srcvec2freespace(source_vector * vector_mask)[:, self.__source_planes[side_idx]]  # TODO: only calculate the slice of interest  # TODO: standardize on [side, pol, transverse] instead?

        return out

    def source2detfield(self, source: array_like, out: np.ndarray = None) -> np.ndarray:
        """
        Calculates the (N+1)D-input-field distribution throughout the scatterer for a given source field distribution.

        :param source: The source field distribution in the whole space.
        :param out: (optional) numpy array to store the result (shape: self.grid.shape, dtype: self.dtype).

        :return: The field distribution as an array of shape [nb_pol, *self.grid], where nb_pol = 3 for a vectorial
            calculation and 1 for a scalar calculation.
        """
        self.__solution.source_distribution = source
        self.__solution.E = 0  # reset every calculation
        self.__solution.solve(self.__callback)

        if out is None:
            out = self.__solution.E
        else:
            out[:] = self.__solution.E

        return out

    def srcvec2detfield(self, input_vector: array_like, out: np.ndarray = None) -> np.ndarray:
        """
        Calculates the (N+1)D-input-field distribution throughout the scatterer for a given input source vector.

        Used in ```self.__matmul__()``` and external code.

        :param input_vector: A source vector with ```self.shape[1]``` elements.
        :param out: (optional) numpy array to store the result.

        :return: The field distribution as an array of shape [nb_pol, *self.grid], where nb_pol = 3 for a vectorial
            calculation and 1 for a scalar calculation.
        """
        return self.source2detfield(self.srcvec2source(input_vector), out=out)

    def detfield2detvec(self, field: array_like) -> np.ndarray:
        """
        Converts the (N+1)D-output-field defined at the front and back detection planes of the scatterer to a detection
        vector that describes the resulting far-field distribution (the inverse Fourier transform of the fields at the
        sample origin). The fields at those planes should only contain the outward propagating waves. Hence, inwards
        propagating waves should be subtracted before using this method!

        This is used in ```self.__matmul__()```.

        :param field: The detected field in all space (of which only the detection space is used).

        :return: Detection vector.
        """
        field = np.asarray(field)
        detector_plane_fields_ift = np.empty(shape=(field.shape[0], 2, *self.grid.shape[1:]), dtype=self.dtype)
        for direction_idx, detector_volume_indices in enumerate(self.__detector_volumes_indices):
            # Select the front and back plane
            detection_volume_field = field[:, detector_volume_indices]
            # Conjugate the backward propagating field so that the inverse Fourier transform flips the direction, the value conjugation is undone later.
            if direction_idx > 0:
                detection_volume_field = detection_volume_field.conj()
            field_ft = ft.ifftn(detection_volume_field, axes=1 + np.arange(self.grid.ndim))  # [polarization, propagation-axis, *transverse-axes]
            # f_z = ft.ifftshift(np.arange(detector_volume_indices.size) - detector_volume_indices.size // 2)
            # outward_propagating = f_z < 0
            # # outward_propagating = f_z * (direction_idx * 2 - 1) > 0  # 0: forward (detection at back), 1: backward (detection at front)
            # field_ft = np.sum(field_ft[:, outward_propagating], axis=1)  # Average along propagation axis
            # todo: Divide by fraction of sampled sinc inside detector volume. The plane waves do not necessarily have an integer period in the propagation direction, so its Fourier representation could have non-zero values in both directions, which would be cropped by the above.
            field_ft = np.sum(field_ft, axis=1)  # Average along propagation axis
            detector_plane_fields_ift[:, direction_idx] = field_ft

        detector_plane_distances = self.grid[0].ravel()[[_[0] for _ in self.__detector_volumes_indices]]  # Determine the position of the central detection plane in each set (listed first)
        # Convert to transverse k-space at the detector planes, later re-reference to the origin.
        # Pick out the propagating modes in a 3D array of the form [polarization, side, mode_index]
        field_at_modes = detector_plane_fields_ift.reshape([*detector_plane_fields_ift.shape[:2], -1])[:, :, self.__mode_direction_indices[::-1]]

        # Determine the 3D k-vectors in a 2D slice and normalize
        k_vector_dir = np.zeros([3, *self.grid.shape[1:]])
        for _ in range(1, self.grid.ndim):
            k_vector_dir[_] = self.grid.k[_] / self.__solution.wavenumber
        k_vector_dir = k_vector_dir.reshape([k_vector_dir.shape[0], 1, -1])[:, :, self.__mode_direction_indices]  # select only the propagating modes
        k_vector_dir[0] = np.sqrt(np.maximum(1 - sum(k_vector_dir[1:]**2), 0.0))  # forward propagating only
        if self.vectorial:
            # Ensure that the polarization is transverse
            field_at_modes -= self.__dot(field_at_modes, k_vector_dir) * k_vector_dir
            # The axial direction
            axial_dir = np.array([1, 0, 0])[:, np.newaxis, np.newaxis]
            # The radial direction, orthonormal to the k-vector and pointing along the negative propagation axis
            radial_dir = self.__norm(self.__dot(k_vector_dir, axial_dir) * k_vector_dir - axial_dir)
            radial_mode_dir = self.__norm(k_vector_dir[1:])  # The mode-space (2D) radial direction
            radial_mode_norm = self.__dot(field_at_modes, radial_dir)
            azimuthal_mode = field_at_modes[1:] - radial_mode_norm * radial_dir[1:]
            output_vector = azimuthal_mode + radial_mode_norm * radial_mode_dir
        else:
            output_vector = field_at_modes

        # Inverse shift in k-space because we used a regular ifft for arbitrary origin positions.
        output_vector *= np.exp(-1j * self.__solution.wavenumber * self.__dot(k_vector_dir[1:self.grid.ndim], self.grid.first[1:, np.newaxis]))

        # Propagate plane waves backwards from the detection plane to the origin
        output_vector *= np.exp(-1j * self.__solution.wavenumber * k_vector_dir[0] * detector_plane_distances[:, np.newaxis])

        # Compensate for emission angle. The exit aperture is effectively smaller for waves exiting at an angle.
        output_vector *= np.sqrt(k_vector_dir[0])  # TODO: Why square root?

        # Move polarization from the left to the right-hand side
        output_vector = np.moveaxis(output_vector, 0, -1)

        # Undo the conjugation of the values so that only the directions are flipped
        output_vector[1] = output_vector[1].conj()

        return output_vector.ravel()

    def detvec2srcvec(self, vec: array_like) -> np.ndarray:
        """
        Convert forward propagating detection vector into a backward propagating (time-reversed) source vector.

        :param vec: detection vector obtained from solution or scattering matrix multiplication.

        :return: Time-reversed vector, that can be used as a source vector.
        """
        # upward and downward directions are stored in the first and second half of the vector, so these must be swapped.
        # Phase conjugation is needed to reverse time.
        # TODO: Check if the sign needs to flip for vectorial calculations
        return ft.fftshift(vec.conj())

    def _matvec(self, right: array_like) -> np.ndarray:
        """
        Multiply this matrix with a vector or matrix: S @ right
        This is the lowest level operation on which @ and __array__() depend.

        :param: A vector or matrix to right-multiply with the MxM matrix, specified as either an array of shape (M, ) or (M, N).
        Alternatively, an nd-array can be specified of shape (2, M, P) or (2, M, P, N), where the first axis indicates the side
        (front, back), the second axis indicates the mode number, and the third axis indicates the polarization (V, H).

        :return: The result vector or matrix of the same shape as the input parameter.
        """
        right = np.asarray(right)
        input_shape = right.shape
        right = right.reshape(-1, right.shape[-1] if right.ndim > 1 else 1)  # Make into a 2D array
        if self.shape[1] != right.shape[0]:
            raise TypeError(f'Incorrect number of modes specified for this {self.shape} scattering matrix.')

        result = np.empty([self.shape[0], right.shape[1]], dtype=self.dtype)

        for column_idx in range(right.shape[1]):
            right_hand_side = right[:, column_idx]
            non_zeros = right_hand_side != 0
            nb_non_zeros = np.sum(non_zeros)
            if nb_non_zeros > 0:
                basis_index = None
                if self.caching and nb_non_zeros == 1:
                    basis_index = np.where(non_zeros)[0][0]
                if not self.caching or basis_index not in self.__cached_columns:
                    # Determine the forward and backward free-space fields
                    freespace_solutions = []
                    for side in range(2):
                        masked_rhs = right_hand_side.reshape(2, -1) * (np.arange(2) != side)[:, np.newaxis]  # reverse order!
                        freespace_solution = self.srcvec2freespace(masked_rhs.ravel())
                        freespace_solutions.append(freespace_solution[:, self.__detector_volumes_indices[side]])  # only store the incident wave so it can later be subtracted

                    # Calculate the field for the scatterer
                    outward_fld = self.srcvec2detfield(right_hand_side)  # recycle the field array for each column
                    # Remove the incident field
                    for detection_volume_indices, freespace_detection_field in zip(self.__detector_volumes_indices, freespace_solutions):
                        outward_fld[:, detection_volume_indices] -= freespace_detection_field  # field2vector only cares about the 2 detector volumes, everything else is ignored
                    # Convert back to a vector and store one column of the result at a time
                    result[:, column_idx] = self.detfield2detvec(outward_fld)
                    # If requested and useful, cache the result
                    if self.caching and basis_index is not None:
                        self.__cached_columns[basis_index] = result[:, column_idx] / right_hand_side[basis_index]  # Make sure not to return a reference to the internal cache!
                else:
                    result[:, column_idx] = self.__cached_columns[basis_index] * right_hand_side[basis_index]  # Make sure not to return a reference to the internal cache!
            else:
                result[:, column_idx] = 0.0  # The right hand side was all zeros

        return result.reshape(input_shape)  # The scattering matrix is square, so we can use the input_shape as output.

    def __setitem__(self, key, value):
        """
        Updating this matrix is not possible. Use a ```Matrix``` object instead.

        :param key: Index or slice.
        :param value: The new value.
        """
        raise KeyError('The values of this ScatteringMatrix are calculated from the scattering and cannot be changed directly. Use a Matrix object instead.')

    def __array__(self, out: array_like = None):
        """Lazily calculates the scattering matrix as a regular :class:`numpy.ndarray`"""
        # Allocate space for result if needed
        if out is None:
            out = np.empty(self.shape, dtype=self.dtype)
        # Calculate the matrix values one column at a time (and optionally, cache the result).
        input_vector = np.zeros(self.shape[1], dtype=self.dtype)
        for _ in range(self.shape[1]):  # Multiply by np.eye(*self.shape) without constructing the identity matrix
            log.info(f'Calculating column {_}/{self.shape[1]}...')
            input_vector[_] = 1
            out[_] = self @ input_vector
            input_vector[_] = 0
        out = out.transpose()
        return out


#
# Classes and functions to manipulate the scattering matrix and split it into transmission and reflection matrices
#
class QuarterMatrix(SquareMatrix):
    """A base class representing a quarter of a scattering matrix."""
    def __init__(self, matrix: SquareMatrix, backwards_output: bool, backwards_input: bool):
        """
        Construct an object refering to a quarter of a scattering matrix. Any Scattering matrix should have an even
        number of rows and columns.

        :param matrix: The underlying scattering matrix.
        :param backwards_output: When True, select the bottom half of the matrix, i.e. the quadrants corresponding to
            the output modes exiting the front side of the scatterer (back-to-front propagation).
        :param backwards_input: When True, select the right half of the matrix, i.e. the quadrants corresponding to
            the input modes entering from the back side of the scatterer (back-to-front propagation).
        """
        if matrix.side % 2 > 0:
            raise TypeError(f'A QuarterMatrix object can only be constructed using a Matrix with an even number of rows and columns. The specified matrix has dimensions {matrix.shape}.')
        side = matrix.side // 2
        super().__init__(side=side, dtype=matrix.dtype)
        self.__matrix = matrix
        self.__output_slice = slice(side if backwards_output else None, side if not backwards_output else None, None)
        self.__input_slice = slice(side if backwards_input else None, side if not backwards_input else None, None)

    @property
    def full_matrix(self) -> SquareMatrix:
        """The underlying (scattering) matrix of size `2 * self.shape`."""
        return self.__matrix

    def _matvec(self, vec: array_like) -> np.ndarray:
        """Multiply this quarter of the matrix with a vector and return a vector of the same shape."""
        in_scattering_vector = np.zeros(self.shape[1] * 2, dtype=vec.dtype)
        in_scattering_vector[self.__input_slice] = vec.ravel()
        out_scattering_vector = self.full_matrix.matvec(in_scattering_vector)
        return np.reshape(out_scattering_vector[self.__output_slice], vec.shape)


class ForwardTransmissionMatrix(QuarterMatrix):
    """
    The forward transmission matrix of the specified scattering matrix.
    It indicates how the light coming from negative infinity is transmitted to positive infinity.
    """
    def __init__(self, scattering_matrix: SquareMatrix):
        super().__init__(scattering_matrix, False, False)


class FrontReflectionMatrix(QuarterMatrix):
    """
    The forward reflection matrix of the specified scattering matrix.
    It indicates how the light coming from positive infinity is back reflected to positive infinity.
    """
    def __init__(self, scattering_matrix: SquareMatrix):
        super().__init__(scattering_matrix, True, False)


class BackReflectionMatrix(QuarterMatrix):
    """
    The backward reflection matrix of the specified scattering matrix.
    It indicates how the light coming from negative infinity is back reflected to negative infinity.
    """
    def __init__(self, scattering_matrix: SquareMatrix):
        super().__init__(scattering_matrix, False, True)


class BackwardTransmissionMatrix(QuarterMatrix):
    """
    The backward transmission matrix of the specified scattering matrix.
    It indicates how the light coming from positive infinity is transmitted to negative infinity.
    """
    def __init__(self, scattering_matrix: SquareMatrix):
        super().__init__(scattering_matrix, True, True)


class DepositionMatrix(Matrix, CachingMatrix):
    """A rectangular matrix that relates free-space input vectors to an arbitrary subset of fields in the interior."""
    def __init__(self, scattering_matrix: ScatteringMatrix,
                 input_operator: Optional[LinearOperator] = None,
                 output_operator: Union[LinearOperator, Callable[[array_like], np.ndarray], None] = None,
                 caching: bool = True):
        """
        Creates a matrix based on the internally scattered fields of a ScatteringMatrix.

        :param scattering_matrix: The base ScatteringMatrix
        :param input_operator: The optional field projector, a LinearOperator object that multiplies any input vector
            to produce a linear combination of source fields. Default: the srcvec2source method of the ScatteringMatrix.
        :param output_operator: The optional detection field projector, a LinearOperator object that multiplies the
            raveled full-field distribution and returns a projection vector or a function that does the same.
            Default: the detfield2detvec method of the ScatteringMatrix.
        :param caching: Cache field propagation calculations. By default, the results are cached for multiplications
            with basis vectors. Numerical errors might accumulate for certain superpositions. Setting this property to
            False will ensure that field propagations are always used and the constructor argument array is ignored.
        """
        if input_operator is None:
            input_operator = LinearOperator(shape=((1 + 2 * scattering_matrix.vectorial) * scattering_matrix.grid.size, scattering_matrix.shape[1]),
                                            matvec=lambda _: scattering_matrix.srcvec2source(_).ravel(),
                                            dtype=scattering_matrix.dtype
                                            )
        if output_operator is None:
            output_operator = LinearOperator(shape=(scattering_matrix.shape[0], (1 + 2 * scattering_matrix.vectorial) * scattering_matrix.grid.size),
                                            matvec=lambda _: scattering_matrix.detfield2detvec(
                                                _.reshape([1 + 2 * scattering_matrix.vectorial, *scattering_matrix.grid.shape])
                                            ),
                                            dtype=scattering_matrix.dtype
                                            )
        elif isinstance(output_operator, Callable):
            input_size = (1 + 2 * scattering_matrix.vectorial) * scattering_matrix.grid.size
            output_shape = output_operator(np.zeros(input_size)).shape
            if len(output_shape) > 1:
                output_operator = lambda _: output_operator(_).ravel()
            output_operator = LinearOperator(shape=(np.prod(output_shape, dtype=
            int), input_size),
                                             matvec=output_operator,
                                             dtype=scattering_matrix.dtype
                                             )
        self.__scattering_matrix = scattering_matrix
        self.__input_op = input_operator
        self.__output_op = output_operator

        Matrix.__init__(self, shape=(self.__output_op.shape[0], self.__input_op.shape[1]), dtype=scattering_matrix.dtype)
        CachingMatrix.__init__(self, caching=caching)

    def _matvec(self, right: array_like) -> np.ndarray:
        """
        Multiply this matrix with a vector or matrix: S @ right
        This is the lowest level operation on which @ and __array__() depend.

        :param: A vector or matrix to right-multiply with the MxN matrix, specified as either an array of shape (N, ) or (N, P).

        :return: The result vector or matrix of shape as the input parameter except for the dimension with N, which becomes M.
        """
        output_shape = np.array(np.asarray(right).shape)
        output_shape[0] = self.shape[0]
        out = np.empty(output_shape, dtype=self.dtype)
        if output_shape.size == 1:
            right = right[:, np.newaxis]
            out = out[:, np.newaxis]

        def calc_op_product(_: np.ndarray) -> np.ndarray:
            source = (self.__input_op @ _).reshape([1 + 2 * self.__scattering_matrix.vectorial, *self.__scattering_matrix.grid.shape])
            return self.__output_op @ self.__scattering_matrix.source2detfield(source).ravel()

        out = self._cache(right=right, value_function=calc_op_product, out=out)

        return out.reshape(output_shape)


#
# Functions for general use
#

def inv(mat: array_like, noise_level: float = 0.0) -> np.ndarray:
    """
    ScatteringMatrix inversion with optional Tikhonov regularization.
    """
    if noise_level > 0.0:  # regularize
        u, sigma, vh = np.linalg.svd(mat)
        sigma_inv = sigma.conj() / (np.abs(sigma)**2 + noise_level**2)
        return ((u * sigma_inv) @ vh).transpose().conj()
    else:  # can only handle non-singular matrices!
        return np.linalg.inv(mat)


def convert(s: array_like, noise_level: float = 0.0) -> np.ndarray:
    """
    Converts a scattering matrix into a transfer matrix and vice-versa.

    Notation:
        p: positive propagation direction along propagation axis 0,
        n: negative propagation direction along propagation axis 0,
        i: inwards propagating (from source on either side),
        o: outwards propagating (backscattered or transmitted).

    Scattering matrix equation (in -> out):
        [po] = [A, B] [pi],
        [no]   [C, D] [ni].

    Transfer matrix equation (top -> bottom):
        [po] = [A - B inv(D) C,  B inv(D)] [pi],
        [ni] = [  -   inv(D) C     inv(D)] [no],
    where inv(D) is the (regularized) inverse of D.

    The first half of the vector inputs and outputs to the scattering and transfer matrices represent fields
    propagating forward along the positive propagation axis (0) and the second half represents fields propagating
    backward along the negative direction.

    :param s: The scattering (transfer) matrix as a 2D np.ndarray of shape [N, N]. Alternatively, each dimension can
        be split in two halves for downward and upward propagating fields. In that case the shape would be [2, N//2, 2, N//2].

    :param noise_level: The noise level of the measured matrix elements (singular values).
        If greater than 0, (Tikhonov) regularization will be used.

    :return: The transfer (scattering) matrix of the same shape as the scattering matrix.
    """
    s = np.asarray(s)
    orig_shape = s.shape  # [N, N]
    if s.ndim == 2:
        new_shape = sum(([2, _ // 2] for _ in orig_shape), [])
        s = s.reshape(new_shape)  # (2, N//2, 2, N//2):

    t = np.empty_like(s)
    t[1, :, 1] = inv(s[1, :, 1], noise_level)
    t[1, :, 0] = -t[1, :, 1] @ s[1, :, 0]
    t[0, :, 1] = s[0, :, 1] @ t[1, :, 1]
    t[0, :, 0] = s[0, :, 0] + s[0, :, 1] @ t[1, :, 0]  # s[0, :, 0] - s[0, :, 1] @ t[1, :, 1] @ s[1, :, 0]

    if len(orig_shape) == 2:
        t = t.reshape(orig_shape)

    return t
