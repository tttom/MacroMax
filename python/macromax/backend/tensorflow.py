"""The module providing the TensorFlow back-end implementation."""
import numpy as np
from typing import Union, Callable
from numbers import Complex
import tensorflow as tf
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()
import os

from macromax.utils.array import Grid
from .. import log
from .__init__ import BackEnd, array_like

tensor_type = tf.Tensor
array_like = Union[array_like, tensor_type]

_tpu_cache = dict()


class BackEndTensorFlow(BackEnd):
    """
    A class that provides methods to work with arrays of matrices or block-diagonal matrices, represented as ndarrays,
    where the first two dimensions are those of the matrix, and the final dimensions are the coordinates over
    which the operations are parallelized and the Fourier transforms are applied.
    """
    def __init__(self, nb_dims: int, grid: Grid, hardware_dtype=tf.complex128, device: str = None, address: str = None):
        """
        Construct object to handle parallel operations on square matrices of nb_rows x nb_rows elements.
        The matrices refer to points in space on a uniform plaid grid.

        :param nb_dims: The number of rows and columns in each matrix. 1 for scalar operations, 3 for polarization
        :param grid: The grid that defines the position of the matrices.
        :param hardware_dtype: (optional) The data type to use for operations.
        :param device: (optional) 'cpu', 'gpu', or 'tpu' to indicate where the calculation will happen.
        :param address: (optional)
        """
        if hardware_dtype == np.complex64:
            hardware_dtype = tf.complex64
        elif hardware_dtype == np.complex128:
            hardware_dtype = tf.complex128
        super().__init__(nb_dims, grid, hardware_dtype)

        if device is None or device.lower() == 'tpu':
            if address is None:
                if 'COLAB_TPU_ADDR' in os.environ:
                    address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
                else:
                    log.debug('No TPU address specified. Should be of the form grpc://IP:port')

            if ((address is None) and ('None' in _tpu_cache)) or (address in _tpu_cache):
                tpus = _tpu_cache[address if address is not None else 'None']
            else:
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
                if address is not None:
                    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=address)
                    tf.config.experimental_connect_to_cluster(resolver)
                    tf.tpu.experimental.initialize_tpu_system(resolver)  # Initialize once in program
                tpus = tf.config.list_logical_devices('TPU')
                _tpu_cache[address if address is not None else 'None'] = tpus  # cache
            if len(tpus) > 0:
                log.info('Found TPU devices: ' + tpus)
            else:
                log.info('No TPU found.')
                if device is not None:
                    raise ValueError('No TPU found!')
        if device is None or device.lower() == 'gpu':
            gpus = tf.config.list_physical_devices("GPU")
            if len(gpus) > 0:
                log.info('Found GPU devices: ' + gpus)
            else:
                log.info('No GPUs found.')
                if device is not None:
                    raise ValueError('No GPU found!')
        if device is None:
            device = 'cpu'

        self.__device = '/' + device.upper()  # Do not specify a specific number

        self.__longitudinal_projection = None  # scalar array for the calculation of longitudinal_projection_ft

    @property
    def numpy_dtype(self):
        """The equivalent hardware data type in numpy"""
        dtype = self.hardware_dtype
        if dtype == tf.complex64:
            numpy_dtype = np.complex64
        elif dtype == tf.complex128:
            numpy_dtype = np.complex128
        else:
            numpy_dtype = np.complex128
        return numpy_dtype

    @property
    def eps(self) -> float:
        return np.finfo(self.numpy_dtype).eps  # TODO: this should extend to non-numpy-equivalent types

    def astype(self, arr: array_like, dtype=None) -> tensor_type:
        """
        As necessary, convert the ndarray arr to the type dtype.
        """
        # tf.autograph.set_verbosity(3, True)
        if dtype is None:
            dtype = self.hardware_dtype
        elif dtype == np.complex64:
            dtype = tf.complex64
        elif dtype in [np.complex128, np.complex, complex]:
            dtype = tf.complex128
        elif dtype == np.float32:
            dtype = tf.float32
        elif dtype in [np.float64, np.float, float]:
            dtype = tf.float64
        if not isinstance(arr, tf.Tensor):
            arr = tf.convert_to_tensor(arr)
        if arr.dtype != dtype:
            arr = tf.cast(arr, dtype=dtype)
        return arr

    def asnumpy(self, arr: array_like) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            arr = arr.numpy()  #tf.make_ndarray(tf.make_tensor_proto(arr))  # todo: not working yet!
        return arr

    def assign(self, arr, out) -> tensor_type:
        arr = self.to_matrix_field(arr)
        if np.any(arr.shape[-self.grid.ndim:] != self.grid.shape):
            arr = tf.tile(arr, np.array(out.shape) // np.array(arr.shape))
        out = self.assign_exact(arr, out)
        return out

    def assign_exact(self, arr, out) -> tensor_type:
        out = tf.raw_ops.Copy(input=self.astype(arr))  # TODO: Is this even needed for TensorFlow? Or better use Assign?
        return out

    def allocate_array(self, shape: array_like = None, dtype=None, fill_value: Complex = None) -> tensor_type:
        """Allocates a new vector array of shape grid.shape and word-aligned for efficient calculations."""
        if shape is None:
            shape = [self.vector_length, 1, *self.grid.shape]
        with tf.device(self.__device):
            if fill_value is None:
                fill_value = 0.0
            arr = tf.fill(shape, value=tf.constant(fill_value, dtype=self.hardware_dtype))
        return arr

    def copy(self, arr: array_like) -> tensor_type:
        """Makes an independent copy of an ndarray."""
        return tf.raw_ops.Copy(input=arr)

    def ravel(self, arr: array_like) -> tensor_type:
        """Returns a flattened view of the array."""
        return tf.reshape(arr, [-1])

    def sign(self, arr: array_like) -> tensor_type:
        return tf.math.sign(arr)

    def swapaxes(self, arr: array_like, ax_from: int, ax_to: int) -> tensor_type:
        """Transpose (permute) two axes of an ndarray."""
        p = np.arange(2 + self.grid.ndim, dtype=int)
        p[ax_to], p[ax_from] = ax_from, ax_to
        return tf.transpose(arr, perm=p)

    @staticmethod
    def expand_dims(arr: array_like, axis: int) -> tensor_type:
        """Inserts a new singleton axis at the indicated position, thus increasing ndim by 1."""
        return tf.expand_dims(arr, axis)

    def abs(self, arr) -> tensor_type:
        return tf.math.abs(self.astype(arr))

    def real(self, arr: array_like) -> tensor_type:
        return tf.math.real(arr)

    def conj(self, arr) -> tensor_type:
        return tf.math.conj(self.astype(arr))

    def any(self, arr: array_like):
        """Returns True if all elements of the array are True."""
        return tf.reduce_any(self.astype(arr, dtype=bool))

    def allclose(self, arr: array_like, other: array_like = 0.0) -> bool:
        """Returns True if all elements in arr are close to other."""
        return tf.reduce_all(tf.abs(arr - self.astype(other)) < 2 * self.eps)

    def amax(self, arr):
        """Returns the maximum of the flattened array."""
        return tf.reduce_max(self.astype(arr, dtype=float))

    def sort(self, arr: array_like) -> tensor_type:
        """Sorts array elements along the first (left-most) axis."""
        return tf.sort(self.real(arr), axis=0)

    def ft(self, arr: array_like) -> tensor_type:
        """
        Calculates the discrete Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\\log(N))``.

        :param arr: An ndarray representing a vector field.

        :return: An ndarray holding the Fourier transform of the vector field E.
        """
        arr = self.astype(arr)
        p = [0, 1, self.ft_axes[-1], *self.ft_axes[:-1]]  # self.ft_axes must be all axes on the right
        for axis in self.ft_axes:
            arr = tf.signal.fft(arr)
            arr = tf.transpose(arr, perm=p)
        return arr

    def ift(self, arr: array_like) -> tensor_type:
        """
        Calculates the inverse Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\\log(N))``.
        The scaling is so that ``E == self.ift(self.ft(E))``

        :param arr: An ndarray representing a Fourier-transformed vector field.

        :return: An ndarray holding the inverse Fourier transform of the vector field E.
        """
        arr = self.astype(arr)
        p = [0, 1, *self.ft_axes[1:], self.ft_axes[0]]  # self.ft_axes must be all axes on the right
        for _ in self.ft_axes:
            arr = tf.transpose(arr, perm=p)
            arr = tf.signal.ifft(arr)
        return arr

    # @tf.function
    def convolve(self, operation_ft: Callable[[array_like], array_like], arr: array_like) -> tensor_type:
        return super().convolve(operation_ft=operation_ft, arr=arr)

    def adjoint(self, mat: array_like) -> tensor_type:
        """
        Transposes the elements of individual matrices with complex conjugation.

        :param mat: The ndarray with the matrices in the first two dimensions.

        :return: An ndarray with the complex conjugate transposed matrices.
        """
        p = [1, 0, *(2 + np.arange(self.grid.ndim))]
        return tf.math.conj(tf.transpose(self.astype(mat), perm=p))

    def subtract(self, left_term: array_like, right_term: array_like) -> np.ndarray:
        """
        Point-wise difference of A and B.

        :param left_term: The left matrix array, must start with dimensions n x m
        :param right_term: The right matrix array, must have matching or singleton dimensions to those
            of A. In case of missing dimensions, singletons are assumed.

        :return: The point-wise difference of both sets of matrices. Singleton dimensions are expanded.
        """
        left_term = self.astype(left_term)
        if not self.is_scalar(left_term) or not self.is_scalar(right_term):
            if self.is_scalar(left_term):
                left_term = self.eye * self.to_matrix_field(left_term)
            if self.is_scalar(right_term):
                right_term = self.eye * self.to_matrix_field(right_term)
        return left_term - right_term

    def is_scalar(self, arr: array_like) -> bool:
        """
        Tests if A represents a scalar field (as opposed to a vector field).

        :param arr: The ndarray to be tested.

        :return: A boolean indicating whether A represents a scalar field (True) or not (False).
        """
        return np.isscalar(arr) or len(arr.shape) == 0 or (arr.shape[0] == 1 and (len(arr.shape) == 1 or arr.shape[1] == 1))

    @tf.function
    def mul(self, left_factor: array_like, right_factor: array_like, out: tf.Tensor = None) -> tensor_type:
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
            result = self.astype(left_factor) * self.astype(right_factor)
        else:
            p = [*(2 + np.arange(self.grid.ndim)), 0, 1]
            ip = [self.grid.ndim, self.grid.ndim + 1, *np.arange(self.grid.ndim)]
            left_factor = tf.transpose(self.astype(left_factor), perm=p)
            right_factor = tf.transpose(self.astype(right_factor), perm=p)
            result = tf.linalg.matmul(left_factor, right_factor)
            result = tf.transpose(result, perm=ip)
        return result

    def ldivide(self, denominator: array_like, numerator: array_like = 1.0) -> tensor_type:
        """
        Parallel matrix left division, A^{-1}B, on the final two dimensions of A and B
        result_lm = A_kl \\ B_km

        A and B must have have all but the final dimension identical or singletons.
        B defaults to the identity matrix.

        :param denominator: The set of denominator matrices.
        :param numerator: The set of numerator matrices.

        :return: The set of divided matrices.
        """
        denominator = self.to_matrix_field(denominator)  # convert scalar to array if needed
        numerator = self.to_matrix_field(numerator)  # convert scalar to array if needed

        shape_A = denominator.shape[:2]
        if self.is_scalar(denominator):
            return self.astype(numerator) / denominator
        else:
            denominator = self.asnumpy(denominator)  # TODO: Keep this in Tensorflow
            numerator = self.asnumpy(numerator)  # TODO: Keep this in Tensorflow
            total_dims = 2 + self.grid.ndim
            new_order = np.roll(np.arange(total_dims), -2)
            denominator = denominator.transpose(new_order)
            if self.is_scalar(numerator):
                if shape_A[0] == shape_A[1]:
                    Y = np.linalg.inv(denominator) * numerator
                else:
                    Y = np.linalg.pinv(denominator) * numerator
            else:
                numerator = numerator.transpose(new_order)
                if shape_A[0] == shape_A[1]:
                    Y = np.linalg.solve(denominator, numerator)
                else:
                    Y = np.linalg.lstsq(denominator, numerator)[0]
            old_order = np.roll(np.arange(total_dims), 2)
            result = Y.transpose(old_order)
            return self.astype(result)

    def norm(self, arr: array_like) -> float:
        return float(tf.linalg.norm(arr))

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
        field_array_ft = self.astype(field_array_ft)

        zero_k2 = tuple(0 for _ in range(nb_data_dims))

        # Store the DC components for later use
        field_dc = [field_array_ft[out_dim_idx, 0][zero_k2] for out_dim_idx in range(nb_data_dims)]

        # Pre-alocate a working array
        if self.__longitudinal_projection is None:
            self.__longitudinal_projection = self.allocate_array(shape=data_shape)

        # (K x K) . xFt == K x (K . xFt)
        self.__longitudinal_projection = self.assign(self.k[0] * field_array_ft[0, 0], self.__longitudinal_projection)  # overwrite with new data
        # Project on k vectors
        for in_dim_idx in range(1, nb_data_dims):
            self.__longitudinal_projection += self.k[in_dim_idx] * field_array_ft[in_dim_idx, 0]

        # Divide by K**2 but handle division by zero separately
        self.__longitudinal_projection /= (self.k2 + self.astype(self.k2 == 0))  # avoid / 0 at the origin
        # self.ravel(self.__longitudinal_projection)[1:] /= self.ravel(self.k2)[1:]  # skip the / 0 at the origin

        polarizations = []
        for out_dim_idx in range(nb_data_dims):  # Save time by not storing the complete tensor
            # stretch each k vector to be as long as the projection
            polarization = self.k[out_dim_idx] * self.__longitudinal_projection
            polarization = tf.tensor_scatter_nd_update(polarization, [(0, 0, *zero_k2)], [field_dc[out_dim_idx]])  # undefined origin => project the DC as longitudinal
            polarizations.append(polarization)
        for out_dim_idx in range(nb_data_dims, nb_output_dims):  # Make sure to add zeros to fit the number of dimensions
            polarizations.append(tf.zeros(shape=[1, 1, *self.grid.shape], dtype=self.hardware_dtype))

        result = tf.concat(polarizations, axis=0)
        # result = tf.reshape(result, shape=[result.shape[0], 1, *result.shape[1:]])
        return result  # const.piLF <- (K x K)/K**2

    def transversal_projection_ft(self, field_array_ft: array_like) -> tensor_type:
        """
        Projects the Fourier transform of a vector E array onto its transversal component.

        :param field_array_ft: The Fourier transform of the input vector E array.

        :return: The Fourier transform of the transversal projection.
        """
        transversal_ft = self.astype(field_array_ft)  # Get it in place for an Inverse Fourier Transform
        transversal_ft -= self.longitudinal_projection_ft(field_array_ft)
        return transversal_ft

    def div(self, field_array: array_like) -> tensor_type:
        """
        Calculates the divergence of input field_array.

        :param field_array: The input array representing the field in all spatial dimensions.
            The input is of the shape ``[m, n, x, y, z, ...]``
        :return: The divergence of the field in the shape ``[n, 1, x, y, z]``.
        """
        field_array_ft = self.ft(field_array)

        div_field_array_ft = self.allocate_array(shape=field_array_ft.shape[1:], dtype=self.hardware_dtype, fill_value=0)
        for dim_idx in range(np.minimum(self.grid.ndim, len(field_array_ft.shape))):
            div_field_array_ft += 1j * self.k[dim_idx] * field_array_ft[dim_idx, :, ...]

        arr_ft = self.expand_dims(div_field_array_ft, 1)
        arr = self.ift(arr_ft)

        return arr

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
        field_array_ft = self.astype(field_array_ft)
        # Calculate the curl
        # curl_field_array_ft = self.allocate_array(shape=[self.vector_length, *field_array_ft.shape[1:]], dtype=self.hardware_dtype)
        curl_field_array_ft_polarizations = []
        # as the cross product without representing the first factor in full
        for dim_idx in range(self.vector_length):
            other_dims = (dim_idx + np.array([-1, 1])) % self.vector_length
            if other_dims[0] < field_array_ft.shape[0] and other_dims[1] < len(self.k):
                res = self.k[other_dims[1]] * field_array_ft[other_dims[0]]
            else:
                res = tf.zeros(shape=[1, *self.grid.shape], dtype=self.hardware_dtype)
            if other_dims[1] < field_array_ft.shape[0] and other_dims[0] < len(self.k):
                res -= self.k[other_dims[0]] * field_array_ft[other_dims[1]]
            curl_field_array_ft_polarizations.append(res)

        curl_field_array_ft = tf.stack(curl_field_array_ft_polarizations)
        return curl_field_array_ft * tf.constant(1.0j, dtype=self.hardware_dtype)

    def mat3_eigh(self, arr: array_like) -> tensor_type:
        """
        Calculates the eigenvalues of the 3x3 Hermitian matrices represented by A and returns a new array of 3-vectors,
        one for each matrix in A and of the same dimensions, baring the second dimension. When the first two
        dimensions are 3x1 or 1x3, a diagonal matrix is assumed. When the first two dimensions are singletons (1x1),
        a constant diagonal matrix is assumed and only one eigenvalue is returned.
        Returns an array of one dimension less: 3 x data_shape.
        With the exception of the first dimension, the shape is maintained.

        :param arr: The set of 3x3 input matrices for which the eigenvalues are requested.
                  This must be an ndarray with the first two dimensions of size 3.

        :return: The set of eigenvalue-triples contained in an ndarray with its first dimension of size 3,
                 and the remaining dimensions equal to all but the first two input dimensions.
        """
        arr = self.astype(arr)
        arr = tf.transpose(arr, perm=[*(2 + np.arange(self.grid.ndim)), 0, 1])
        result = tf.linalg.eigvalsh(arr)
        result = tf.transpose(result, perm=[self.grid.ndim, *np.arange(self.grid.ndim)])
        return result
