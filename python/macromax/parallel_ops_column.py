import numpy as np
from numpy.lib import scimath as sm
import scipy.constants as const

from . import log

from .utils import ft
from .utils.array import vector_to_axis, word_align, Grid

try:
    import multiprocessing
    nb_threads = multiprocessing.cpu_count()
    log.debug('Detected %d CPUs, using up to %d threads.' % (nb_threads, nb_threads))
except ModuleNotFoundError:
    nb_threads = 1
    log.info('Module multiprocessing not found, assuming single-threaded environment.')


class ParallelOperations:
    """
    A class that provides methods to work with arrays of matrices or block-diagonal matrices, represented as ndarrays,
    where the first two dimensions are those of the matrix, and the final dimensions are the coordinates over
    which the operations are parallelized and the Fourier transforms are applied.
    """
    def __init__(self, nb_dims, grid: Grid, dtype=np.complex128):
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

        self.__cutoff = len(self.matrix_shape)
        self.__total_dims = len(self.matrix_shape) + len(self.grid.shape)

        self.__eye = np.eye(nb_dims, dtype=self.dtype).reshape((nb_dims, nb_dims, *np.ones(len(self.grid.shape), dtype=int)))

        ft_axes = range(self.__cutoff, self.__total_dims)  # Don't Fourier transform the matrix dimensions
        try:
            import pyfftw

            log.debug('Module pyfftw imported, using FFTW.')

            ftflags = ('FFTW_DESTROY_INPUT', 'FFTW_ESTIMATE', )
            # ftflags = ('FFTW_DESTROY_INPUT', 'FFTW_PATIENT', )
            # ftflags = ('FFTW_DESTROY_INPUT', 'FFTW_MEASURE', )
            # ftflags = ('FFTW_DESTROY_INPUT', 'FFTW_EXHAUSTIVE', ) # very slow, little gain in general
            log.debug("Number of threads available for FFTW: %d" % nb_threads)

            self.__empty_word_aligned = \
                lambda shape, dtype: pyfftw.empty_aligned(shape=shape, dtype=dtype, n=pyfftw.simd_alignment)
            self.__zeros_word_aligned = \
                lambda shape, dtype: pyfftw.zeros_aligned(shape=shape, dtype=dtype, n=pyfftw.simd_alignment)
            self.__word_align = lambda a: word_align(a, word_length=pyfftw.simd_alignment)

            log.debug('Allocating FFTW''s operating memory.')
            fftw_vec_array = self.__empty_word_aligned([self.matrix_shape[0], 1, *self.grid.shape], dtype=self.dtype)

            log.debug('Initializing FFTW''s forward Fourier transform.')
            fft_vec_object = pyfftw.FFTW(fftw_vec_array, fftw_vec_array, axes=ft_axes, flags=ftflags,
                                         direction='FFTW_FORWARD', planning_timelimit=None, threads=nb_threads)
            log.debug('Initializing FFTW''s backward Fourier transform.')
            ifft_vec_object = pyfftw.FFTW(fftw_vec_array, fftw_vec_array, axes=ft_axes, flags=ftflags,
                                          direction='FFTW_BACKWARD', planning_timelimit=None, threads=nb_threads)
            log.debug('FFTW''s wisdoms generated.')

            # Redefine the default method
            def fftw(E):
                if np.any(np.array(E.shape[2:]) > 1):
                    assert(E.ndim-2 == self.grid.shape.size and np.all(E.shape[2:] == self.grid.shape))
                    if np.all(E.shape == fft_vec_object.input_shape):
                        result_array = self.__empty_word_aligned(E.shape, dtype=self.dtype)
                        fft_vec_object(E, result_array)
                    else:
                        log.debug('Fourier Transform: Array shape not standard, falling back to default interface.')
                        result_array = ft.fftn(E, axes=ft_axes)
                    return result_array
                else:
                    return E.copy()

            def fftw_inplace(E):
                if np.any(np.array(E.shape[2:]) > 1):
                    strides_not_identical = np.any(E.strides != fftw_vec_array.strides)
                    if strides_not_identical:
                        log.debug('In-place Fourier Transform: strides not identical.')
                        E = self.__word_align(E.copy())
                    if not pyfftw.is_n_byte_aligned(E, pyfftw.simd_alignment):
                        log.debug('In-place Fourier Transform: Input/Output array not %d-byte word aligned, aligning now.'
                                  % pyfftw.simd_alignment)
                        E = self.__word_align(E)
                    assert(E.ndim-2 == self.grid.shape.size and np.all(E.shape[2:] == self.grid.shape))
                    if np.all(E.shape == fft_vec_object.input_shape) \
                            and pyfftw.is_n_byte_aligned(E, pyfftw.simd_alignment):
                        fft_vec_object(E, E)  # E should be in SIMD-word-aligned memory zone
                    else:
                        log.debug('Fourier Transform: Array shape not standard, falling back to default interface.')
                        E = ft.fftn(E, axes=ft_axes)
                return E

            # Redefine the default method
            def ifftw(E):
                if np.any(np.array(E.shape[2:]) > 1):
                    assert(E.ndim-2 == self.grid.shape.size and np.all(E.shape[2:] == self.grid.shape))
                    if np.all(E.shape == ifft_vec_object.input_shape):
                        result_array = self.__empty_word_aligned(E.shape, dtype=self.dtype)
                        ifft_vec_object(E, result_array)
                    else:
                        log.debug('Inverse Fourier Transform: Array shape not standard, falling back to default interface.')
                        result_array = ft.ifftn(E, axes=ft_axes)
                    return result_array
                else:
                    return E.copy()

            def ifftw_inplace(E):
                strides_not_identical = np.any(E.strides != fftw_vec_array.strides)
                if strides_not_identical:
                    log.debug('In-place Fourier Transform: strides not identical.')
                if not pyfftw.is_n_byte_aligned(E, pyfftw.simd_alignment) or strides_not_identical:
                    log.debug('In-place Inverse Fourier Transform: Input/Output array not %d-byte word aligned, aligning now.'
                              % pyfftw.simd_alignment)
                    E = self.__word_align(E)

                if np.any(np.array(E.shape[2:]) > 1):
                    assert(E.ndim-2 == self.grid.shape.size and np.all(E.shape[2:] == self.grid.shape))
                    if np.all(E.shape == ifft_vec_object.input_shape):
                        ifft_vec_object(E, E)  # E should be in a SIMD-word-aligned memory zone
                    else:
                        log.debug('Inverse Fourier Transform: Array shape not standard, falling back to default interface.')
                        E = ft.ifftn(E, axes=ft_axes)

                return E

            self.__ft = fftw
            self.__ift = ifftw
        except ModuleNotFoundError:
            log.debug('Module pyfftw not imported, using stock FFT.')

            self.__ft = lambda E: ft.fftn(E, axes=ft_axes)
            self.__ift = lambda E: ft.ifftn(E, axes=ft_axes)
            self.__empty_word_aligned = lambda shape, dtype: np.empty(shape=shape, dtype=dtype)
            self.__zeros_word_aligned = lambda shape, dtype: np.zeros(shape=shape, dtype=dtype)
            self.__word_align = lambda a: a

    @property
    def dtype(self):
        return self.__dtype

    @property
    def eye(self):
        """
        Returns an identity tensor that can be multiplied using singleton expansion. This can be useful for scalar
        additions or subtractions.

        :return: an array with the number of dimensions matching that of the ParallelOperation's data set.
        """
        return self.__eye

    @property
    def matrix_shape(self):
        """
        :return: The shape of the square matrix that transforms a single vector in the set.
        This is a pair of identical integer numbers.
        """
        return np.array([self.__nb_rows, self.__nb_rows])

    @property
    def grid(self):
        """
        :return: A Grid object representing the sample points in the spatial dimensions.
        """
        return self.__grid

    @property
    def vectorial(self):
        """
        :return: A boolean indicating if this object represents a vector space (as opposed to a scalar space).
        """
        return self.matrix_shape[0] > 1

    @staticmethod
    def check_alignment(array, message='NOT ALIGNED'):
        """
        Diagnostic method for testing the word-alignment of an array to enable efficient operations.

        :param array: The ndarray to be tested.
        :param message: Optional message to display in case of failure.
        """
        aligned = pyfftw.is_n_byte_aligned(array, pyfftw.simd_alignment)
        if not aligned and message is not None:
            log.debug(message)

        return aligned

    def ft(self, E):
        """
        Calculates the discrete Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\log(N))``.

        :param E: An ndarray representing a vector field.
        :return: An ndarray holding the Fourier transform of the vector field E.
        """
        return self.__ft(E)  # Defined in __init__ depending on imports

    def ift(self, E):
        """
        Calculates the inverse Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\log(N))``.
        The scaling is so that ``E == self.ift(self.ft(E))``

        :param E: An ndarray representing a Fourier-transformed vector field.
        :return: An ndarray holding the inverse Fourier transform of the vector field E.
        """
        return self.__ift(E)  # Defined in __init__ depending on imports

    def is_scalar(self, A):
        """
        Tests if A represents a scalar field (as opposed to a vector field).

        :param A: The ndarray to be tested.
        :return: A boolean indicating whether A represents a scalar field (True) or not (False).
        """
        return np.isscalar(A) or A.ndim == 0 or (A.shape[0] == 1 and (A.ndim == 1 or A.shape[1] == 1))

    def is_vector(self, A):
        """
        Tests if A represents a vector field.

        :param A: The ndarray to be tested.
        :return: A boolean indicating whether A represents a vector field (True) or not (False).
        """
        return A.ndim == self.__total_dims-1

    def is_matrix(self, A):
        """
        Checks if an ndarray is a matrix as defined by this parallel_ops_column object.

        :param A: The matrix to be tested.
        :return: boolean value, indicating if A is a matrix.
        """
        return A.ndim == self.__total_dims

    def __fix_matrix_dims(self, M):
        """
        Converts everything to an array with self.__total_dims dimensions.

        :param M: The input can be None or scalar, which assumes that its value is assumed to be repeated for all space.
            The value can be a one-dimensional vector, in which case the vector is assumed to be repeated for all space.
        :return: An array with the correct number of dimensions and with each non-singleton dimension matching those pf
            the nb_dims and data_shape.
        """
        if M is None:
            M = 0
        if np.isscalar(M):
            M = np.array(M)
        if M.ndim == self.__total_dims:
            return M  # an input already with the correct number of dimensions
        elif M.ndim == self.__total_dims - 1 and np.any((np.array(M.shape[:2]) != self.matrix_shape) & (np.array(M.shape[:2]) != 1)):
            return M[:, np.newaxis, ...]  # a vector E input
        # complete the dimensions to the right as required
        orig_shape = M.shape
        new_shape = np.ones(self.__total_dims, dtype=np.int)
        new_shape[:len(orig_shape)] = orig_shape
        return M.reshape(new_shape)

    def to_simple_matrix(self, M):
        """
        Converts input to an array of the correct dimensions, though permitting singleton dimensions for all but the two
        matrix dimensions which must either be both full or both singletons. In the latter case, the identity matrix is
        assumed. None is assumed to refer to an all 0 array.

        :param M: The input None, scalar, or array.
        :return: An array with the correct dimensions
        """
        M = self.__fix_matrix_dims(M)

        return M

    def to_simple_vector(self, vec):
        """
        Converts input to an array of the correct dimensions, though permitting singleton dimensions for all but the two
        matrix dimensions which must either be both full or both singletons. In the latter case, the identity matrix is
        assumed. None is assumed to refer to an all 0 array.

        :param vec: The input None, scalar, or array.
        :return: An array with the correct dimensions
        """
        vec = self.__fix_matrix_dims(vec)

        # Since we don't have an explicit distinction between scalar vectors and matrices, and the (unit) scalar already
        # represents the identity matrix, we need to represent scalar vectors in full
        if vec.shape[0] == 1 and self.vectorial:
            vec = vec.repeat(self.matrix_shape[0], 0)

        return vec

    def to_full_matrix(self, M):
        """
        Converts input to an array of the correct dimensions, though permitting singleton dimensions for all but the two
        matrix dimensions on the left, which must equal the matrix_shape property. If these dimensions are both 1,
        the identity matrix is assumed.

        :param M: The input None, scalar, or array.
        :return: An array with the correct dimensions
        """
        M = self.__fix_matrix_dims(M)
        if np.all(np.array(M.shape[:2]) == 1):
            M = self.eye * M
        elif (M.shape[0] != self.matrix_shape[0]) | ((M.shape[1] != 1) & (M.shape[1] != self.matrix_shape[1])):
            message = 'A matrix with dimensions %dx%d expected, one with %dx%d found.' %\
                      (*self.matrix_shape, *M.shape[:2])
            log.critical(message)
            raise Exception(message)
        return M

    def transpose(self, M):
        """
        Transposes the elements of individual matrices without complex conjugation.

        :param M: The ndarray with the matrices in the first two dimensions.
        :return: An ndarray with the transposed matrices.
        """
        new_order = np.arange(self.__total_dims)
        new_order[:2] = new_order[[1, 0]]
        return M.transpose(new_order)

    def conjugate_transpose(self, M):
        """
        Transposes the elements of individual matrices with complex conjugation.

        :param M: The ndarray with the matrices in the first two dimensions.
        :return: An ndarray with the complex conjugate transposed matrices.
        """
        return np.conj(self.transpose(M))

    def add(self, A, B):
        """
        Point-wise addition of A and B.

        :param A: The left matrix array, must start with dimensions n x m
        :param B: The right matrix array, must have matching or singleton dimensions to those
            of A. In case of missing dimensions, singletons are assumed.
        :return: The point-wise addition of both sets of matrices. Singleton dimensions are expanded.
        """
        if not self.is_scalar(A) or not self.is_scalar(B):
            A = self.to_full_matrix(A)
            B = self.to_full_matrix(B)
        return A + B

    def subtract(self, A, B):
        """

        Point-wise difference of A and B.

        :param A: The left matrix array, must start with dimensions n x m
        :param B: The right matrix array, must have matching or singleton dimensions to those
            of A. In case of missing dimensions, singletons are assumed.
        :return: The point-wise difference of both sets of matrices. Singleton dimensions are expanded.
        """
        return self.add(A, -B)

    def mul(self, A, B):
        """
        Point-wise matrix multiplication of A and B.

        :param A: The left matrix array, must start with dimensions n x m
        :param B: The right matrix array, must have matching or singleton dimensions to those
            of A, bar the first two dimensions. In case of missing dimensions, singletons are assumed.
            The first dimensions must be m x p. Where the m matches that of the left hand matrix
            unless both m and p are 1 or both n and m are 1, in which case the scaled identity is assumed.

        :return: An array of matrix products with all but the first two dimensions broadcast as needed.
        """
        if self.is_scalar(A) or self.is_scalar(B):
            return A * B  # Scalars are assumed to be proportional to the identity matrix
        else:
            return np.einsum('ij...,jk...->ik...', A, B, optimize=False)

    def ldivide(self, A, B=1.0):
        """
        Parallel matrix left division, A^{-1}B, on the final two dimensions of A and B
        result_lm = A_kl \ B_km

        A and B must have have all but the final dimension identical or singletons.
        B defaults to the identity matrix.

        :param A: The set of denominator matrices.
        :param B: The set of numerator matrices.
        :return: The set of divided matrices.
        """
        A = self.__fix_matrix_dims(A)  # convert scalar to array if needed
        B = self.__fix_matrix_dims(B)  # convert scalar to array if needed

        shape_A = np.array(A.shape[:2])
        if self.is_scalar(A):
            return B / A
        else:
            new_order = np.roll(np.arange(self.__total_dims), -2)
            A = A.transpose(new_order)
            if self.is_scalar(B):
                if shape_A[0] == shape_A[1]:
                    Y = B * np.linalg.inv(A)
                else:
                    Y = B * np.linalg.pinv(A)
            else:
                B = B.transpose(new_order)
                if shape_A[0] == shape_A[1]:
                    Y = np.linalg.solve(A, B)
                else:
                    Y = np.linalg.lstsq(A, B)[0]
            old_order = np.roll(np.arange(self.__total_dims), 2)
            return Y.transpose(old_order)

    def inv(self, M):
        """
        Inverts the set of input matrices M.

        :param M: The set of input matrices.
        :return: The set of inverted matrices.
        """
        return self.ldivide(M, 1.0)

    def curl(self, E):
        """
        Calculates the curl of a vector E with the final dimension the vector dimension.

        :param E: The set of input matrices.
        :return: The curl of E.
        """
        F = self.ft(E)
        F = self.curl_ft(F)
        return self.ift(F)

    def curl_ft(self, F):
        """
            Calculates the Fourier transform of the curl of a Fourier transformed E with the final dimension the
            vector dimension.
            The final dimension of the output will always be of length 3; however, the input length may be shorter,
            in which case the missing values are assumed to be zero.
            The first dimension of the input array corresponds to the first element in the final dimension,
            if it exists, the second dimension corresponds to the second element etc.

            :param F: The input vector array of dimensions ``[vector_length, 1, *data_shape]``.
            :return: The Fourier transform of the curl of F.
        """
        vector_length = self.matrix_shape[0]

        # Pre-calculate the k-values along each dimension and pre-multiply by i
        k_ranges = self.__get_k_ranges(input_shape=F.shape[2:], imaginary=True)

        # Calculate the curl
        curl_F = self.__empty_word_aligned([vector_length, *F.shape[1:]], dtype=(F[0]+1.0j).dtype)
        # as the cross product without representing the first factor in full
        for dim_idx in range(vector_length):
            other_dims = (dim_idx + np.array([-1, 1])) % vector_length
            if other_dims[0] < F.shape[0] and other_dims[1] < len(k_ranges):
                curl_F[dim_idx] = k_ranges[other_dims[1]][np.newaxis, ...] * F[other_dims[0]]
            else:
                curl_F[dim_idx].ravel()[:] = 0.0
            if other_dims[1] < F.shape[0] and other_dims[0] < len(k_ranges):
                curl_F[dim_idx] -= k_ranges[other_dims[0]][np.newaxis, ...] * F[other_dims[1]]

        return curl_F

    @staticmethod
    def cross(A, B):
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
    def outer(A, B):
        """
        Calculates the Dyadic product of vector arrays A and B.

        :param A: A vector array of dimensions ``[vector_length, 1, *data_shape]``
        :param B:  A vector array of dimensions ``[vector_length, 1, *data_shape]``
        :return: A matrix array of dimensions ``[vector_length, vector_length, *data_shape]``
            containing the dyadic product :math:`A \otimes B` in the first two dimensions and
            the other dimensions remain on the same axes.
        """
        return A * np.conj(B)[np.newaxis, :, 0, ...]  # transpose the first two axes of B

    def div(self, E):
        """
        Calculate the divergence of the input vector or tensor field E.

        :param E: The input array representing vector or tensor field. The input is of the shape ``[m, n, x, y, z, ...]``
        :return: The divergence of the vector or tensor field in the shape ``[n, 1, x, y, z]``.
        """
        E_F = self.ft(E)
        div_E_F = self.div_ft(E_F)
        result = self.ift(div_E_F)

        return result

    def div_ft(self, E_F):
        """
        Calculated the Fourier transform of the divergence of the pre-Fourier transformed input E_F.

        :param E_F: The input array representing the field pre-Fourier transformed in all spatial dimensions.
            The input is of the shape ``[m, n, x, y, z, ...]``
        :return: The Fourier transform of the divergence of the field in the shape ``[n, 1, x, y, z]``.
        """
        # Pre-calculate the k-values along each dimension and pre-multiply by i
        input_shape = E_F.shape[2:]
        k_ranges = self.__get_k_ranges(input_shape=input_shape, imaginary=True)

        div_F = self.__zeros_word_aligned([E_F.shape[1], 1, *input_shape], dtype=(E_F.flatten()[0] + 1j).dtype)
        for dim_idx in range(np.minimum(len(k_ranges), len(input_shape))):
            div_F += k_ranges[dim_idx][np.newaxis, np.newaxis, ...] * E_F[dim_idx, :, np.newaxis, ...]

        return div_F

    # TODO: Needed?
    def integral(self, E):
        """
        Calculate the integral of the input vector or tensor field E.

        :param E: The input array representing vector or tensor field. The input is of the shape ``[m, n, x, y, z, ...]``
        :return: The integral of the vector or tensor field in the shape ``[n, 1, x, y, z]``.
        """
        E_F = self.ft(E)
        integral_E_F = self.integral_ft(E_F)
        result = self.ift(integral_E_F)

        return result

    def integral_ft(self, E_F):
        """
        Calculated the Fourier transform of the integral of the pre-Fourier transformed input E_F.

        :param E_F: The input array representing the field pre-Fourier transformed in all spatial dimensions.
            The input is of the shape ``[m, n, x, y, z, ...]``
        :return: The Fourier transform of the integral of the field in the shape ``[n, 1, x, y, z]``.
        """
        # Pre-calculate the k-values along each dimension and pre-multiply by i
        input_shape = E_F.shape[2:]
        k_ranges = self.__get_k_ranges(input_shape=input_shape, imaginary=True)
        inv_k_ranges = [1.0 / (rng + (rng == 0)) for rng in k_ranges]

        integral_F = self.__zeros_word_aligned([3, 1, *input_shape], dtype=(E_F.flatten()[0] + 1j).dtype)
        for dim_idx in range(np.minimum(len(k_ranges), len(input_shape))):
            integral_F[dim_idx] = inv_k_ranges[dim_idx][np.newaxis, np.newaxis, ...] * E_F[dim_idx, :, np.newaxis, ...]

        return integral_F

    def transversal_projection(self, E):
        """
        Projects vector E arrays onto their transverse component.

        :param E: The input vector E array.
        :return: The transversal projection.
        """
        return self.ift(self.transversal_projection_ft(self.ft(E)))

    def longitudinal_projection(self, E):
        """
        Projects vector E arrays onto their longitudinal component

        :param E: The input vector E array.
        :return: The longitudinal projection.
        """
        return self.ift(self.longitudinal_projection_ft(self.ft(E)))

    def transversal_projection_ft(self, F):
        """
        Projects the Fourier transform of a vector E array onto its transversal component.

        :param E: The Fourier transform of the input vector E array.
        :return: The Fourier transform of the transversal projection.
        """
        return F - self.longitudinal_projection_ft(F)

    def longitudinal_projection_ft(self, F):
        """
        Projects the Fourier transform of a vector E array onto its longitudinal component.

        :param E: The Fourier transform of the input vector E array.
        :return: The Fourier transform of the longitudinal projection.
        """
        F = F[:, 0, ...]
        data_shape = F.shape
        nb_input_vector_dims = data_shape[0]
        data_shape = data_shape[1:]
        nb_data_dims = np.min((len(data_shape), len(self.grid.step)))
        nb_output_dims = np.max((nb_data_dims, nb_input_vector_dims))

        # Pre-calculate the k-values along each dimension
        k_ranges = self.__get_k_ranges(input_shape=data_shape)

        # (K x K) . xFt == K x (K . xFt)
        projection = np.zeros(data_shape, F.dtype)
        K2 = np.zeros(data_shape, F.dtype)
        # Project on k vectors
        for in_dim_idx in range(nb_data_dims):
            projection += k_ranges[in_dim_idx] * F[in_dim_idx]
            K2 += k_ranges[in_dim_idx] ** 2

        # Divide by K**2 but handle division by zero separately
        projection /= (K2 + (K2 == 0))  # avoid the NaN at the origin => replace with zero
        zeroK2 = np.where(K2 == 0)
        del K2

        result = np.zeros([nb_output_dims, *data_shape], dtype=F.dtype)
        for out_dim_idx in range(nb_data_dims):  # Saving memory by not storing the complete tensor
            # stretch each k vector to be as long as the projection
            result[out_dim_idx] = k_ranges[out_dim_idx] * projection
            result[out_dim_idx][zeroK2] = F[out_dim_idx][zeroK2]  # undefined origin => project the DC as longitudinal

        return result[:, np.newaxis, ...]  # const.piLF <- (K x K)/K**2

    def calc_k2(self, data_shape=None):
        """
        Helper def for calculation of the Fourier transform of the Green def

        :param data_shape: The requested output shape.
        :return: :math:`|k|^2` for the specified sample grid and output shape
        """
        if data_shape is None:
            data_shape = self.grid.shape
        k2 = 0.0
        k_grid = Grid(data_shape, self.grid.step).k
        for axis in range(k_grid.ndim):
            k2 = k2 + k_grid[axis]**2

        return k2[np.newaxis, np.newaxis, ...]

    def mat3_eig(self, A):
        """
        Calculates the eigenvalues of the 3x3 matrices represented by A and
        returns a new array of 3-vectors, one for each matrix in A and of the
        same dimensions, baring the second dimension. When the first two
        dimensions are 3x1 or 1x3, a diagonal matrix is assumed. When the first
        two dimensions are singletons (1x1), a constant diagonal matrix is assumed
        and only one eigenvalue is returned.
        All dimensions are maintained in size except for dimension 2
        Returns an array or one dimension less. 3 x data_shape

        :param A: The set of 3x3 input matrices for which the eigenvalues are requested.
                  This must be an ndarray with the first two dimensions of size 3.
        :return: The set of eigenvalue-triples contained in an ndarray with its first dimension of size 3,
                 and the remaining dimensions equal to all but the first two input dimensions.
        """
        #
        # TODO: Check if this could be implemented faster / more accurately with an iterative algorithm,
        # e.g. Given's rotation to make Hermitian tridiagonal + power iteration.
        #
        data_shape = np.array(A.shape[2:])
        matrix_shape = np.array(A.shape[:2])
        if np.all(matrix_shape == 3):
            C = np.zeros([4, *data_shape], dtype=A.dtype)
            # A 3x3 matrix in the first two dimensions
            C[0] = A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) \
                   - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) \
                   + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
            C[1] = - A[0, 0] * (A[1, 1] + A[2, 2]) + A[0, 1] * A[1, 0] \
                   + A[0, 2] * A[2, 0] - A[1, 1] * A[2, 2] + A[1, 2] * A[2, 1]
            C[2] = A[0, 0] + A[1, 1] + A[2, 2]
            C[3] = -1

            D = self.calc_roots_of_low_order_polynomial(C)
        else:
            if np.all((matrix_shape == 1) | (matrix_shape == 3)):
                # Maybe a scalar or diagonal-as-column
                D = A.copy()
            else:
                message = 'The final two dimensions of the input array should be either of length 1 or of length 3.'
                log.critical(message)
                raise Exception(message)

        return D


    def __get_k_ranges(self, input_shape=None, imaginary=False):
        """
        Prepare a set of vectors, one for each dimension, with the k-values along the respective dimension.
        The values are monotonously increasing before an iffshift so that the DC component is in the first entry of the
        each vector.

        :param input_shape: (optional) An alternative shape for the sample grid (without affecting the sample pitch).
        :param imaginary: (optional) multiply all values by the imaginary constant i.
        :return: A vector of 1D-vectors with the k-values along each dimension.
        """
        if input_shape is None:
            input_shape = self.grid.shape[2:]

        nb_data_dims = self.__total_dims - 2
        k_ranges = []  # already including the imaginary constant in the ranges
        for dim_idx in range(np.minimum(self.grid.ndim, nb_data_dims)):
            rl = input_shape[dim_idx]
            k_range = 2.0 * const.pi / (self.grid.step[dim_idx]*rl) * np.fft.ifftshift(np.arange(rl)-np.floor(rl/2))
            if imaginary:
                k_range = 1.0j * k_range  # convert type from real to complex
            k_range = vector_to_axis(k_range, dim_idx, nb_data_dims)
            k_ranges.append(k_range)

        return k_ranges

    def calc_roots_of_low_order_polynomial(self, C):
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
            return np.abs(arr) > np.sqrt(np.finfo(arr.dtype).eps)

        def add_roots(coeffs):
            """
            Checks if this is a lower-order polynomial in disguise and adds zero roots to match the order of the array

            :param coeffs: The polynomial coefficient array
            :return: A, nb_roots_added The non-degenerate polynomial coefficient array, A, and the number of roots added
            """
            # Although the coefficients are integers, the calculations may be real and complex.
            coeffs = np.array(coeffs, dtype=self.dtype)
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
                is_zero = np.bitwise_not(is_significant(roots[dim_idx]))
                remove_zero = np.where(is_zero & (roots_added > 0))
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
            complex_root = True  #TODO: fix numerical issues with real roots.  is_significant(R.imag) | is_significant(Q.imag) | (S2 >= 0)
            not_around_origin = is_significant(Q)  # avoiding division by zero in case the roots are centered around zero
            RQm32 = sm.power(R, 1/3) / (not_around_origin * sm.sqrt(Q) + (1 - not_around_origin))
            AB_all_real_roots = -sm.sqrt(Q) * (RQm32 + 1j * sm.sqrt(1.0 - RQm32**2))
            S = sm.sqrt(S2)  # S**2 = R**2 - Q**3 => Q**3 == R**2 - S**2 = (R-S)*(R+S)
            A_complex_root = -sm.power(R + sign(np.real(np.conj(R) * S)) * S, 1/3)
            # Combine the different calculations for A
            A = complex_root * A_complex_root + (1 - complex_root) * AB_all_real_roots
            # choose sign of sqrt so that real(conj(R) * sqrt(R**2 - Q**3)) >= 0
            A_significant = is_significant(A)
            # if A insignificant, then R + np.sign(np.real(np.conj(R) * S)) * S == 0
            #  and can be divided out of Q^3 = R^2 - S^2 = (R-S)*(R+S)
            B_complex_roots = Q / (A + (1 - A_significant))  # avoiding division by zero
            B_complex_roots_origin = - sm.power(R - np.sign(np.real(np.conj(R) * S)) * S, 1/3)

            # Combine the different calculations for B
            B = complex_root * (A_significant * B_complex_roots + (1 - A_significant) * B_complex_roots_origin) \
                + (1 - complex_root) * (-AB_all_real_roots)
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
    def evaluate_polynomial(C, X):
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
