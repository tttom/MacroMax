import numpy as np
from numpy.lib import scimath as sm
import scipy.constants as const
from typing import Callable, Union, Sequence
from numbers import Complex
from abc import ABC, abstractmethod

from . import log

from .utils.array import vector_to_axis, Grid


__all__ = ['ParallelOps', 'get_parallel_ops_implementation', 'array_like', 'tensor_type']


tensor_type = np.ndarray
array_like = Union[Complex, Sequence, tensor_type]


class ParallelOps(ABC):
    """
    A class that provides methods to work with arrays of matrices or block-diagonal matrices, represented as ndarrays,
    where the first two dimensions are those of the matrix, and the final dimensions are the coordinates over
    which the operations are parallelized and the Fourier transforms are applied.
    """
    def __init__(self, nb_dims: int, grid: Grid, dtype=np.complex128):
        """
        Construct object to handle parallel operations on square matrices of nb_rows x nb_rows elements.
        The matrices refer to points in space on a uniform plaid grid.

        :param nb_dims: The number of rows and columns in each matrix. 1 for scalar operations, 3 for polarization
        :param grid: The grid that defines the position of the matrices.
        :param dtype: (optional) The datatype to use for operations.
        """
        self.__nb_rows = nb_dims
        self.__grid = grid
        self.__dtype = dtype

        self.__cutoff = 2
        self.__total_dims = 2 + self.grid.ndim
        # Define lazily-instantiated working arrays for ft, ift, and convolvution
        self.__array_ft_input = None  # vector array for Fourier Transform input and Inverse Fourier Transform output
        self.__array_ft_output = None  # vector array for Inverse Fourier Transform input and Fourier Transform output

        # Cache some arrays
        self.__longitudinal_projection = None  # scalar array for the calculation of longitudinal_projection_ft
        self.__k = None  # list of vectors returned by self.k
        self.__k2 = None  # scalar array with k-squared

        self.__eye = None

    @property
    def vector_length(self) -> int:
        """
        :return: The shape of the square matrix that transforms a single vector in the set.
        This is a pair of identical integer numbers.
        """
        return self.__nb_rows

    @property
    def ft_axes(self) -> tuple:
        return tuple(range(self.__cutoff, self.__total_dims))  # Don't Fourier transform the matrix dimensions

    @property
    def grid(self) -> Grid:
        """
        :return: A Grid object representing the sample points in the spatial dimensions.
        """
        return self.__grid

    @property
    def vectorial(self) -> bool:
        """
        :return: A boolean indicating if this object represents a vector space (as opposed to a scalar space).
        """
        return self.vector_length > 1

    @property
    def dtype(self):
        return self.__dtype

    def astype(self, arr: array_like, dtype=None) -> tensor_type:
        """
        :param
        :return: torch.Tensor type
        """
        if dtype is None:
            dtype = self.dtype
        return np.asarray(arr, dtype)

    @property
    def eps(self) -> float:
        return np.finfo(self.dtype).eps

    @abstractmethod
    def allocate_array(self, shape: array_like = None, dtype = None, fill_value: Complex = None) -> tensor_type:
        """Allocates a new vector array of shape grid.shape and word-aligned for efficient calculations."""
        pass

    def align_array(self, arr: array_like) -> tensor_type:
        """Copy array to a new array that is word-aligned."""
        alligned_arr = self.copy(arr)
        return alligned_arr

    def copy(self, arr: array_like) -> tensor_type:
        """Makes an independent copy of an ndarray."""
        return arr.copy()

    def ravel(self, arr: array_like) -> tensor_type:
        """Returns a flattened view of the array."""
        return arr.ravel()

    def first(self, arr: array_like) -> Complex:
        """Returns the first element of the flattened array."""
        return self.ravel(arr)[0]

    @staticmethod
    def expand_dims(arr: array_like, axis: int) -> tensor_type:
        """Inserts a new singleton axis at the indicated position, thus increasing ndim by 1."""
        return np.expand_dims(arr, axis=axis)

    @property
    def eye(self) -> tensor_type:
        """
        Returns an identity tensor that can be multiplied using singleton expansion. This can be useful for scalar
        additions or subtractions.

        :return: an array with the number of dimensions matching that of the ParallelOperation's data set.
        """
        if self.__eye is None:
            nb_rows = self.vector_length
            self.__eye = self.astype(
                np.eye(nb_rows, dtype=int).reshape((nb_rows, nb_rows, *np.ones(len(self.grid.shape), dtype=int))))
        return self.__eye

    def any(self, arr):
        return np.any(arr)

    def amax(self, arr):
        return np.amax(arr)

    def allclose(self, arr: array_like, other: array_like = 0.0) -> bool:
        """Returns True if all elements in arr are close to other."""
        return np.allclose(arr, other)

    @abstractmethod
    def ft(self, arr: array_like) -> tensor_type:
        """
        Calculates the discrete Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\log(N))``.

        :param arr: An ndarray representing a vector field.
        :return: An ndarray holding the Fourier transform of the vector field E.
        """
        pass

    @abstractmethod
    def ift(self, arr: array_like) -> tensor_type:
        """
        Calculates the inverse Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\log(N))``.
        The scaling is so that ``E == self.ift(self.ft(E))``

        :param arr: An ndarray representing a Fourier-transformed vector field.
        :return: An ndarray holding the inverse Fourier transform of the vector field E.
        """
        pass

    @property
    def array_ft_input(self) -> np.ndarray:
        if self.__array_ft_input is None:
            self.__array_ft_input = self.allocate_array()
        return self.__array_ft_input

    @property
    def array_ft_output(self) -> np.ndarray:
        if self.__array_ft_output is None:
            self.__array_ft_output = self.allocate_array()
        return self.__array_ft_output

    @property
    def array_ift_input(self) -> np.ndarray:
        return self.array_ft_output

    @property
    def array_ift_output(self) -> np.ndarray:
        return self.array_ft_input

    def conj(self, arr) -> tensor_type:
        return np.conj(arr)

    def abs(self, arr) -> tensor_type:
        return np.abs(arr)

    def convolve(self, operation_ft: Callable[[array_like], array_like], arr: array_like) -> tensor_type:
        """
        Cyclic FFT convolution which overwrites its input argument.
        :param operation_ft: The function that acts on the Fourier-transformed input.
        :param arr: The to-be-convolved argument array.
        :returns the convolved input array.
        """
        arr_ft = self.ft(arr)
        arr_ft = operation_ft(arr_ft)
        arr = self.ift(arr_ft)
        return arr

    def is_scalar(self, arr: array_like) -> bool:
        """
        Tests if A represents a scalar field (as opposed to a vector field).

        :param arr: The ndarray to be tested.
        :return: A boolean indicating whether A represents a scalar field (True) or not (False).
        """
        return np.isscalar(arr) or arr.ndim == 0 or (arr.shape[0] == 1 and (arr.ndim == 1 or arr.shape[1] == 1))

    def is_vector(self, arr: array_like) -> bool:
        """
        Tests if A represents a vector field.

        :param arr: The ndarray to be tested.
        :return: A boolean indicating whether A represents a vector field (True) or not (False).
        """
        return arr.ndim == self.__total_dims - 1

    def is_matrix(self, arr: array_like) -> bool:
        """
        Checks if an ndarray is a matrix as defined by this parallel_ops_column object.

        :param arr: The matrix to be tested.
        :return: boolean value, indicating if A is a matrix.
        """
        return arr.ndim == self.__total_dims

    def swapaxes(self, arr: array_like, ax_from: int, ax_to: int) -> tensor_type:
        """Transpose (permute) two axes of an ndarray."""
        return arr.swapaxes(ax_from, ax_to)

    def to_matrix_field(self, arr: array_like) -> np.ndarray:
        """
        Converts the input to an array of the full number of dimensions: len(self.matrix_shape) + len(self.grid.shape).
        For electric fields in 3-space, self.matrix_shape == (N, N) == (3, 3)
        The first (left-most) dimensions of the output are either
        - 1x1: The identity matrix for a scalar field, as sound waves or isotropic permittivity.
        - Nx1: A vector for a vector field, as the electric field.
        - NxN: A matrix for a matrix field, as anisotropic permittivity

        None is interpreted as 0.
        Singleton dimensions are added on the left so that all dimensions are present.
        Inputs with 1xN are transposed (not conjugate) to Nx1 vectors.

        :param arr: The input can be scalar, which assumes that its value is assumed to be repeated for all space.
            The value can be a one-dimensional vector, in which case the vector is assumed to be repeated for all space.
        :return: An array with ndim == len(self.matrix_shape) + len(self.grid.shape) and with each non-singleton
        dimension matching those of the nb_rows and data_shape.
        """
        if arr is None:
            arr = 0
        arr = self.astype(arr)
        while len(arr.shape) < self.grid.ndim + 2:
            arr = self.expand_dims(arr, 0)  # Make sure that the number of dimensions is correct (total_dims)
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            arr = self.swapaxes(arr, 0, 1)
        return arr

    def adjoint(self, mat: array_like) -> tensor_type:
        """
        Transposes the elements of individual matrices with complex conjugation.

        :param mat: The ndarray with the matrices in the first two dimensions.
        :return: An ndarray with the complex conjugate transposed matrices.
        """
        return np.conj(mat.swapaxes(0, 1))

    def subtract(self, left_term: array_like, right_term: array_like) -> np.ndarray:
        """

        Point-wise difference of A and B.

        :param left_term: The left matrix array, must start with dimensions n x m
        :param right_term: The right matrix array, must have matching or singleton dimensions to those
            of A. In case of missing dimensions, singletons are assumed.
        :return: The point-wise difference of both sets of matrices. Singleton dimensions are expanded.
        """
        if not self.is_scalar(left_term) or not self.is_scalar(right_term):
            if self.is_scalar(left_term):
                left_term = self.eye * self.to_matrix_field(left_term)
            if self.is_scalar(right_term):
                right_term = self.eye * self.to_matrix_field(right_term)
        return left_term - right_term

    def mul(self, left_factor: array_like, right_factor: array_like, out: np.ndarray = None) -> np.ndarray:
        """
        Point-wise matrix multiplication of A and B. Overwrites right_factor!

        :param left_factor: The left matrix array, must start with dimensions n x m
        :param right_factor: The right matrix array, must have matching or singleton dimensions to those
            of A, bar the first two dimensions. In case of missing dimensions, singletons are assumed.
            The first dimensions must be m x p. Where the m matches that of the left hand matrix
            unless both m and p are 1 or both n and m are 1, in which case the scaled identity is assumed.
        :param out: (optional) The destination array for the results.

        :return: An array of matrix products with all but the first two dimensions broadcast as needed.
        """
        if self.is_scalar(left_factor) or self.is_scalar(right_factor):
            if out is not None:
                result = out
                result *= left_factor  # Scalars are assumed to be proportional to the identity matrix
            else:
                result = left_factor * right_factor
        else:
            result = np.einsum('ij...,jk...->ik...', left_factor, right_factor, out=out)

        return result

    @abstractmethod
    def ldivide(self, denominator: array_like, numerator: array_like = 1.0) -> tensor_type:
        """
        Parallel matrix left division, A^{-1}B, on the final two dimensions of A and B
        result_lm = A_kl \ B_km

        A and B must have have all but the final dimension identical or singletons.
        B defaults to the identity matrix.

        :param denominator: The set of denominator matrices.
        :param numerator: The set of numerator matrices.
        :return: The set of divided matrices.
        """
        pass

    def inv(self, mat: array_like) -> tensor_type:
        """
        Inverts the set of input matrices M.

        :param mat: The set of input matrices.
        :return: The set of inverted matrices.
        """
        return self.ldivide(mat, 1.0)

    def curl(self, field_array: array_like) -> tensor_type:
        """
        Calculates the curl of a vector E with the final dimension the vector dimension.
        The input argument may be overwritten!

        :param field_array: The set of input matrices.
        :return: The curl of E.
        """
        return self.convolve(self.curl_ft, field_array)

    def curl_ft(self, field_array_ft: array_like) -> tensor_type:
        """
            Calculates the Fourier transform of the curl of a Fourier transformed E with the final dimension the
            vector dimension.
            The final dimension of the output will always be of length 3; however, the input length may be shorter,
            in which case the missing values are assumed to be zero.
            The first dimension of the input array corresponds to the first element in the final dimension,
            if it exists, the second dimension corresponds to the second element etc.

            :param field_array_ft: The input vector array of dimensions ``[vector_length, 1, *data_shape]``.
            :return: The Fourier transform of the curl of F.
        """
        # Calculate the curl
        curl_field_array_ft = self.allocate_array(shape=[self.vector_length, *field_array_ft.shape[1:]], dtype=self.dtype)
        # as the cross product without representing the first factor in full
        for dim_idx in range(self.vector_length):
            other_dims = (dim_idx + np.array([-1, 1])) % self.vector_length
            if other_dims[0] < field_array_ft.shape[0] and other_dims[1] < len(self.k):
                curl_field_array_ft[dim_idx] = self.k[other_dims[1]] * field_array_ft[other_dims[0]]
            else:
                self.ravel(curl_field_array_ft[dim_idx])[:] = 0.0
            if other_dims[1] < field_array_ft.shape[0] and other_dims[0] < len(self.k):
                curl_field_array_ft[dim_idx] -= self.k[other_dims[0]] * field_array_ft[other_dims[1]]

        return 1j * curl_field_array_ft

    @staticmethod
    def cross(A: array_like, B: array_like) -> tensor_type:
        """
        Calculates the cross product of vector arrays A and B.

        :param A: A vector array of dimensions ``[vector_length, 1, *data_shape]``
        :param B:  A vector array of dimensions ``[vector_length, 1, *data_shape]``
        :return: A vector array of dimensions ``[vector_length, 1, *data_shape]`` containing the cross product A x B
            in the first dimension and the other dimensions remain on the same axes.
        """
        vector_length = A.shape[0]
        result = np.zeros(A.shape, dtype=A.dtype)
        for dim_idx in range(vector_length):
            other_dims = (dim_idx + np.array([-1, 1])) % vector_length
            if other_dims[1] < A.shape[0] and other_dims[0] < B.shape[0]:
                result[dim_idx] = A[other_dims[1]] * B[other_dims[0]]
            if other_dims[0] < A.shape[0] and other_dims[1] < B.shape[0]:
                result[dim_idx] -= A[other_dims[0]] * B[other_dims[1]]
        return result

    @staticmethod
    def outer(A: array_like, B: array_like) -> tensor_type:
        """
        Calculates the Dyadic product of vector arrays A and B.

        :param A: A vector array of dimensions ``[vector_length, 1, *data_shape]``
        :param B:  A vector array of dimensions ``[vector_length, 1, *data_shape]``
        :return: A matrix array of dimensions ``[vector_length, vector_length, *data_shape]``
            containing the dyadic product :math:`A \otimes B` in the first two dimensions and
            the other dimensions remain on the same axes.
        """
        return A * B.conjugate()[np.newaxis, :, 0, ...]  # transpose the first two axes of B

    def div(self, field_array: array_like) -> tensor_type:
        """
        Calculate the divergence of the input vector or tensor field E.
        The input argument may be overwritten!

        :param field_array: The input array representing vector or tensor field. The input is of the shape ``[m, n, x, y, z, ...]``
        :return: The divergence of the vector or tensor field in the shape ``[n, 1, x, y, z]``.
        """
        return self.convolve(self.div_ft, field_array)

    def div_ft(self, field_array_ft: array_like) -> tensor_type:
        """
        Calculated the Fourier transform of the divergence of the pre-Fourier transformed input E_F.

        :param field_array_ft: The input array representing the field pre-Fourier-transformed in all spatial dimensions.
            The input is of the shape ``[m, n, x, y, z, ...]``
        :return: The Fourier transform of the divergence of the field in the shape ``[n, 1, x, y, z]``.
        """
        div_field_array_ft = self.allocate_array(shape=field_array_ft.shape[1:], dtype=self.dtype, fill_value=0)
        for dim_idx in range(np.minimum(len(self.k), field_array_ft.ndim-2)):
            div_field_array_ft += 1j * self.k[dim_idx] * field_array_ft[dim_idx, :, ...]

        return self.expand_dims(div_field_array_ft, 1)

    def transversal_projection(self, field_array: array_like) -> tensor_type:
        """
        Projects vector arrays onto their transverse component.
        The input argument may be overwritten!

        :param field_array: The input vector E array.
        :return: The transversal projection.
        """
        return self.convolve(self.transversal_projection_ft, field_array)

    def longitudinal_projection(self, field_array: array_like) -> tensor_type:
        """
        Projects vector arrays onto their longitudinal component.
        The input argument may be overwritten!

        :param field_array: The input vector E array.
        :return: The longitudinal projection.
        """
        return self.convolve(self.longitudinal_projection_ft, field_array)

    def transversal_projection_ft(self, field_array_ft: array_like) -> np.ndarray:
        """
        Projects the Fourier transform of a vector E array onto its transversal component.

        :param field_array_ft: The Fourier transform of the input vector E array.
        :return: The Fourier transform of the transversal projection.
        """
        transversal_ft = self.array_ift_input
        transversal_ft[:] = field_array_ft.astype(dtype=self.dtype)  # Get it in place for an Inverse Fourier Transform
        transversal_ft -= self.longitudinal_projection_ft(field_array_ft)
        return transversal_ft

    def longitudinal_projection_ft(self, field_array_ft: array_like) -> np.ndarray:
        """
        Projects the Fourier transform of a vector array onto its longitudinal component.
        Overwrites self.array_ft_input!

        :param field_array_ft: The Fourier transform of the input vector E array.
        :return: The Fourier transform of the longitudinal projection.
        """
        data_shape = field_array_ft.shape
        nb_input_vector_dims = data_shape[0]
        data_shape = data_shape[2:]
        nb_data_dims = np.minimum(len(data_shape), len(self.grid.step))
        nb_output_dims = np.maximum(nb_data_dims, nb_input_vector_dims)

        zero_k2 = tuple(np.zeros(1, dtype=int) for _ in range(nb_data_dims))

        # Store the DC components for later use
        field_dc = [field_array_ft[out_dim_idx, 0][zero_k2] for out_dim_idx in range(nb_data_dims)]

        # Pre-alocate a working array
        if self.__longitudinal_projection is None:
            self.__longitudinal_projection = self.allocate_array(shape=data_shape)

        # (K x K) . xFt == K x (K . xFt)
        self.__longitudinal_projection[:] = self.k[0] * field_array_ft[0, 0]  # overwrite with new data
        # Project on k vectors
        for in_dim_idx in range(1, nb_data_dims):
            self.__longitudinal_projection += self.k[in_dim_idx] * field_array_ft[in_dim_idx, 0]

        # Divide by K**2 but handle division by zero separately
        self.ravel(self.__longitudinal_projection)[1:] /= self.ravel(self.k2)[1:]  # skip the / 0 at the origin

        result = self.array_ft_input  # reuse input array for result  #self.__longitudinal_projection
        for out_dim_idx in range(nb_data_dims):  # Save work by not storing the complete tensor
            # stretch each k vector to be as long as the projection
            result[out_dim_idx, 0] = self.k[out_dim_idx] * self.__longitudinal_projection
            result[out_dim_idx, 0][zero_k2] = field_dc[out_dim_idx]  # undefined origin => project the DC as longitudinal
        for out_dim_idx in range(nb_data_dims, nb_output_dims):  # Make sure to add zeros to fit the number of dimensions
            result[out_dim_idx, 0] = 0

        return result  # const.piLF <- (K x K)/K**2

    @property
    def k(self) -> Sequence[tensor_type]:
        """
        :return: A list of the k-vector components along each axis.
        """
        dtype = np.float32 if self.dtype == np.complex64 else np.float64
        if self.__k is None:
            self.__k = [self.copy(self.astype(_)) for _ in self.grid.k]
        return self.__k

    @property
    def k2(self) -> tensor_type:
        """
        Helper def for calculation of the Fourier transform of the Green def

        :return: :math:`|k|^2` for the specified sample grid and output shape
        """
        if self.__k2 is None:
            k2 = sum(_**2 for _ in self.k)
            self.__k2 = self.align_array(k2)
        return self.__k2

    def mat3_eig(self, A: array_like) -> tensor_type:
        """
        Calculates the eigenvalues of the 3x3 matrices represented by A and returns a new array of 3-vectors,
        one for each matrix in A and of the same dimensions, baring the second dimension. When the first two
        dimensions are 3x1 or 1x3, a diagonal matrix is assumed. When the first two dimensions are singletons (1x1),
        a constant diagonal matrix is assumed and only one eigenvalue is returned.
        Returns an array of one dimension less: 3 x data_shape.
        With the exception of the first dimension, the shape is maintained.

        :param A: The set of 3x3 input matrices for which the eigenvalues are requested.
                  This must be an ndarray with the first two dimensions of size 3.
        :return: The set of eigenvalue-triples contained in an ndarray with its first dimension of size 3,
                 and the remaining dimensions equal to all but the first two input dimensions.
        """
        #
        # TODO: Check if this could be implemented faster / more accurately with an iterative algorithm,
        # e.g. Use a single Given's rotation or Householder reflection to make Hermitian tridiagonal + power iteration.
        #
        matrix_shape = np.array(A.shape[:-self.grid.ndim])
        data_shape = np.array(A.shape[-self.grid.ndim:])

        if matrix_shape.size > 2:
            raise ValueError(f'The matrix dimensions should be at most 2, not {matrix_shape.size}.')
        while matrix_shape.size < 2:  # pad dimensions to 2
            matrix_shape = matrix_shape[np.newaxis, ...]
        if np.all(matrix_shape == 3):
            # C = np.zeros([4, *data_shape], dtype=A.dtype)
            C = self.allocate_array([4, *data_shape])
            # A 3x3 matrix in the first two dimensions
            C[0] = A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) \
                   - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) \
                   + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
            C[1] = - A[0, 0] * (A[1, 1] + A[2, 2]) + A[0, 1] * A[1, 0] \
                   + A[0, 2] * A[2, 0] - A[1, 1] * A[2, 2] + A[1, 2] * A[2, 1]
            C[2] = A[0, 0] + A[1, 1] + A[2, 2]
            C[3] = -1

            result = self.calc_roots_of_low_order_polynomial(C)
        else:
            if matrix_shape[0] == 1:  # Maybe a scalar or diagonal-as-column
                result = self.copy(A)
                result = result[0, ...]
            elif matrix_shape[1] == 1:  # Maybe a scalar or diagonal-as-column
                result = self.copy(A)
                result = result[:, 0, ...]
            else:
                raise ValueError(f'The vector dimensions of the input array should be of length 1 or 3, not {matrix_shape}.')

        return result

    def calc_roots_of_low_order_polynomial(self, C: array_like) -> tensor_type:
        """
        Calculates the (complex) roots of polynomials up to order 3 in parallel.
        The coefficients of the polynomials are in the first dimension of C and repeated in the following dimensions,
        one for each polynomial to determine the roots of. The coefficients are
        input for low to high order in each column. In other words, the polynomial is:
        ``np.sum(C * (x**range(C.size))) == 0``

        :param C: The coefficients of the polynomial, per polynomial.
        :return: The zeros in the complex plane, per polynomial.
        """
        nb_terms = C.shape[0]
        output_shape = np.array(C.shape)
        output_shape[0] -= 1

        # A few helper functions
        def sign(arr):  # signum without the zero
            return 2 * (arr >= 0) - 1

        def is_significant(arr):
            return self.astype(np.abs(arr) > np.sqrt(self.eps), dtype=bool)

        def add_roots(coeffs):
            """
            Checks if this is a lower-order polynomial in disguise and adds zero roots to match the order of the array

            :param coeffs: The polynomial coefficient array
            :return: A, nb_roots_added The non-degenerate polynomial coefficient array, A, and the number of roots added
            """
            # Although the coefficients are integers, the calculations may be real and complex.
            coeffs = self.astype(coeffs)
            coeffs = coeffs[..., np.newaxis]  # add a singleton dimension to avoid problems with vectors
            roots_added = np.zeros(coeffs[0].shape, dtype=np.uint8)
            for outer_dim_idx in range(coeffs.shape[0]):
                lower_order = np.where(coeffs[-1] == 0)
                for dim_idx in np.arange(coeffs.shape[0]-1, 0, -1):
                    coeffs[dim_idx][lower_order] = coeffs[dim_idx-1][lower_order]
                coeffs[0][lower_order] = 0
                roots_added[lower_order] += 1
            coeffs[-1][lower_order] = 1  # all zeros

            return coeffs[..., 0], roots_added[..., 0]

        def remove_roots(roots, roots_added):
            """
            Removes dummy roots at ``x == 0`` and replaces them with nan at the end of the vector of roots.
            The roots are sorted with np.sort (real first, imaginary second).

            :param roots: The array of all calculated roots.
            :param roots_added: The number of dummy roots added to the problem.
            :return: The array with the dummy roots placed at the end and replaced by nan.
            """
            roots = roots[..., np.newaxis]  # add a singleton dimension to avoid problems with vectors
            roots_added = roots_added[..., np.newaxis]  # add a singleton dimension to avoid problems with vectors
            for dim_idx in np.arange(roots.shape[0]-1, -1, -1):
                is_zero = ~is_significant(roots[dim_idx])  # not
                remove_zero = self.astype(is_zero & (roots_added > 0), dtype=bool)
                roots[dim_idx][remove_zero] = np.nan
                roots_added[remove_zero] -= 1
            roots = np.sort(roots, axis=0)
            return roots[..., 0]

        if nb_terms < 2:
            X = np.empty((0, *C[1:]), dtype=C.dtype)
        elif nb_terms == 2:  # linear
            # C[0] + C[1]*X == 0
            C, nb_roots_added = add_roots(C)
            X = -C[np.newaxis, 0] / C[np.newaxis, 1]
            X = remove_roots(X, nb_roots_added)
        elif nb_terms == 3:  # quadratic
            # C[0] + C[1]*X + C[2]*X**2 == 0
            C, nb_roots_added = add_roots(C)
            d = C[1] ** 2 - 4.0 * C[2] * C[0]
            sqrt_d = sm.sqrt(d)
            q = -0.5 * (C[1] + sign((np.conj(C[1]) * sqrt_d).real) * sqrt_d)
            X = np.array((q / C[2], C[0] / (q + (1 - is_significant(C[0])) * (1 - is_significant(q)))))
            X = remove_roots(X, nb_roots_added)
        elif nb_terms == 4:  # cubic
            C, nb_roots_added = add_roots(C)

            a = C[2] / C[3] / 3
            b = C[1] / C[3]
            c = C[0] / C[3]

            Q = a ** 2 - b / 3
            R = a ** 3 + (- a * b + c) / 2
            del b, c

            S2 = R ** 2 - Q ** 3
            complex_root = self.astype(True, dtype=bool)  #TODO: fix numerical issues with real roots.  is_significant(R.imag) | is_significant(Q.imag) | (S2 >= 0)
            not_around_origin = is_significant(Q)  # avoiding division by zero in case the roots are centered around zero
            RQm32 = sm.power(R, 1/3) / (not_around_origin * sm.sqrt(Q) + (~not_around_origin))
            AB_all_real_roots = -self.astype(Q)**0.5 * (RQm32 + 1j * sm.sqrt(1.0 - RQm32**2))
            S = sm.sqrt(S2)  # S**2 = R**2 - Q**3 => Q**3 == R**2 - S**2 = (R-S)*(R+S)
            A_complex_root = -sm.power(R + sign(np.real(np.conj(R) * S)) * S, 1/3)
            # Combine the different calculations for A
            A = complex_root * A_complex_root + (~complex_root) * AB_all_real_roots
            # choose sign of sqrt so that real(conj(R) * sqrt(R**2 - Q**3)) >= 0
            A_significant = is_significant(A)
            # if A insignificant, then R + np.sign(np.real(np.conj(R) * S)) * S == 0
            #  and can be divided out of Q^3 = R^2 - S^2 = (R-S)*(R+S)
            B_complex_roots = Q / (A + (~A_significant))  # avoiding division by zero
            B_complex_roots_origin = - sm.power(R - np.sign(np.real(np.conj(R) * S)) * S, 1/3)

            # Combine the different calculations for B
            B = complex_root * (A_significant * B_complex_roots + (~A_significant) * B_complex_roots_origin) \
                + (~complex_root) * (-AB_all_real_roots)
            complex_triangle = vector_to_axis(np.exp(2j * const.pi * np.array([-1, 0, 1]) / 3), 0, output_shape.size)
            X = A[np.newaxis, ...] * complex_triangle + B[np.newaxis, ...] * np.conj(complex_triangle)
            X -= a[np.newaxis, ...]

            X = remove_roots(X, nb_roots_added)
        else:
            message = 'Orders above 3 not implemented!'
            log.critical(message)
            raise Exception(message)

        return X

    @staticmethod
    def evaluate_polynomial(C: array_like, X: array_like) -> tensor_type:
        """
        Evaluates the polynomial P at X for testing.

        :param C: The coefficients of the polynomial, for each polynomial.
        :param X: The argument of the polynomial, per polynomial.
        :return: The values of the polynomials for the arguments X.
        """
        results = 0
        for idx in np.arange(C.shape[0])[-1::-1]:
            results = results * X + C[idx]

        return results

    def norm(self, arr: array_like) -> float:
        """Returns the l2-norm of a vectorized array."""
        return np.linalg.norm(arr.ravel())


def get_parallel_ops_implementation(nb_pol_dims: int, grid: Grid, dtype) -> ParallelOps:
    try:
        import torch
        gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        log.info(f'PyTorch version {torch.__version__} ' + ('with' if gpu_available else 'but no') + 'GPU detected.')
        if not gpu_available:
            raise ImportError('PyTorch installed by no CUDA GPU found!')
        try:
            import torch.fft
            fftn_available = True
        except ImportError:
            fftn_available = False
        if not fftn_available:
            raise ImportError('Function torch.fft.fftn not available in this version of PyTorch, please upgrade to version 1.7.0.')

        from .parallel_ops_torch import ParallelOpsTorch
        log.info('Detected PyTorch, using ParallelOpsTorch...')
        return ParallelOpsTorch(nb_pol_dims, grid, dtype)
    except ImportError:
        log.info('PyTorch module not detected, falling back to ParallelOpsNumpy.')
        from .parallel_ops_numpy import ParallelOpsNumpy
        return ParallelOpsNumpy(nb_pol_dims, grid, dtype)
