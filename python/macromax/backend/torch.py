"""The module providing the PyTorch back-end implementation."""
import numpy as np
from typing import Union
from numbers import Complex
import torch
import torch.fft as ft
import logging
import gc

from macromax.utils.ft import Grid
from .__init__ import BackEnd, array_like

log = logging.getLogger(__name__)

tensor_type = torch.Tensor
array_like = Union[array_like, tensor_type]


class BackEndTorch(BackEnd):
    """
    A class that provides methods to work with arrays of matrices or block-diagonal matrices, represented as ndarrays,
    where the first two dimensions are those of the matrix, and the final dimensions are the coordinates over
    which the operations are parallelized and the Fourier transforms are applied.
    """
    def __init__(self, nb_dims: int, grid: Grid, hardware_dtype=torch.complex128, device: str = None):
        """
        Construct object to handle parallel operations on square matrices of nb_rows x nb_rows elements.
        The matrices refer to points in space on a uniform plaid grid.

        :param nb_dims: The number of rows and columns in each matrix. 1 for scalar operations, 3 for polarization
        :param grid: The grid that defines the position of the matrices.
        :param hardware_dtype: (optional) The data type to use for operations.
        :param device: (optional) 'cuda' or 'cpu', to indicate where the calculation will happen.
        """
        if hardware_dtype == np.complex64:
            hardware_dtype = torch.complex64
        elif hardware_dtype == np.complex128:
            hardware_dtype = torch.complex128
        super().__init__(nb_dims, grid, hardware_dtype)

        if device is None or device == 'cuda':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if len(device) > 4:
            device_ordinal = int(device.strip().lower()[5:])
            if device_ordinal > torch.cuda.device_count() - 1:
                log.warning(f'CUDA Device {device} not found, falling back to default device.')
                device = 'cuda'

        self.__device = torch.device(device)

        if device == 'cuda':
            try:  # Check whether CUDA actually works
                arr = self.allocate_array()
                self.subtract(arr, arr)
                del arr
            except RuntimeError as re:
                log.warning(f'PyTorch failed to use CUDA ({re}), possibly due to incompatible CUDA version.')
                raise re

    @property
    def numpy_dtype(self):
        """The equivalent hardware data type in numpy"""
        dtype = self.hardware_dtype
        if dtype == torch.complex64:
            numpy_dtype = np.complex64
        elif dtype == torch.complex128:
            numpy_dtype = np.complex128
        else:
            numpy_dtype = np.complex128
        return numpy_dtype

    @property
    def eps(self) -> float:
        return torch.finfo(self.hardware_dtype).eps

    def astype(self, arr: array_like, dtype=None) -> tensor_type:
        """
        As necessary, convert the ndarray arr to the type dtype.
        """
        if dtype is None:
            dtype = self.hardware_dtype
        elif dtype == np.complex64:
            dtype = torch.complex64
        elif dtype in [np.complex128, complex]:
            dtype = torch.complex128
        elif dtype == np.float32:
            dtype = torch.float32
        elif dtype in [np.float64, float]:
            dtype = torch.float64
        if not isinstance(arr, torch.Tensor):
            arr = torch.tensor(arr, dtype=dtype, device=self.__device)
        elif arr.dtype != dtype or arr.device != self.__device:
            arr = arr.to(dtype=dtype, device=self.__device)
        return arr

    def asnumpy(self, arr: array_like) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu()
            arr = arr.resolve_conj().numpy()
        else:
            arr = np.asarray(arr)
        return arr

    def assign(self, arr, out) -> tensor_type:
        arr = self.to_matrix_field(arr)
        if np.any(arr.shape[-self.grid.ndim:] != self.grid.shape):
            arr = arr.repeat(*(np.array(out.shape) // np.array(arr.shape)))
        out = self.assign_exact(arr, out)
        return out

    def assign_exact(self, arr, out) -> tensor_type:
        out[:] = arr
        return out

    def allocate_array(self, shape: array_like = None, dtype=None, fill_value: Complex = None) -> tensor_type:
        """Allocates a new vector array of shape grid.shape and word-aligned for efficient calculations."""
        if shape is None:
            shape = (self.vector_length, 1, *self.grid.shape)
        elif not isinstance(shape, tuple):
            shape = tuple(shape)

        if dtype is None:
            dtype = self.hardware_dtype
        elif dtype == np.complex64:
            dtype = torch.complex64
        elif dtype in (np.complex128, complex):
            dtype = torch.complex128
        elif dtype == np.float32:
            dtype = torch.float32
        elif dtype in (np.float64, float):
            dtype = torch.float64
        arr = torch.empty(shape, dtype=dtype).to(self.__device)
        if fill_value is not None:
            arr[:] = fill_value
        # import traceback
        # log.info(f'Allocating array of shape {shape} and dtype {dtype} with fill value {fill_value} ({np.prod(arr.size()) * arr.element_size() / 1024**3:0.1f}GB)\nat{traceback.format_stack()[-2]}.')
        return arr

    def copy(self, arr: array_like) -> tensor_type:
        """Makes an independent copy of an ndarray."""
        return arr.detach().clone()

    def ravel(self, arr: array_like) -> tensor_type:
        """Returns a flattened view of the array."""
        return arr.flatten()

    def sign(self, arr: array_like) -> tensor_type:
        return torch.sign(arr)

    def swapaxes(self, arr: array_like, ax_from: int, ax_to: int) -> tensor_type:
        """Transpose (permute) two axes of an ndarray."""
        return arr.transpose(ax_from, ax_to)

    @staticmethod
    def expand_dims(arr: array_like, axis: int) -> tensor_type:
        """Inserts a new singleton axis at the indicated position, thus increasing ndim by 1."""
        return arr.unsqueeze(axis)

    def abs(self, arr) -> tensor_type:
        return self.astype(arr).abs()

    def conj(self, arr) -> tensor_type:
        return self.astype(arr).conj()

    def any(self, arr: array_like):
        """Returns True if all elements of the array are True."""
        return torch.any(self.astype(arr, dtype=bool))

    def allclose(self, arr: array_like, other: array_like = 0.0) -> bool:
        """Returns True if all elements in arr are close to other."""
        return torch.allclose(arr, self.astype(other))

    def amax(self, arr):
        """Returns the maximum of the flattened array."""
        return torch.amax(self.astype(arr, dtype=float)).item()

    def sort(self, arr: array_like) -> tensor_type:
        """Sorts array elements along the first (left-most) axis."""
        # indices = torch.argsort(arr.real, dim=0)
        # for dim in range(arr.shape[0]):
        #     arr[dim, :, :] = arr[dim, indices[dim], 0]
        # TODO: do not move between cpu-numpy-gpu as it is done here
        arr = arr.to(device='cpu')
        arr = np.sort(arr, axis=0)
        arr = self.astype(arr)
        return arr

    def ft(self, arr: array_like) -> tensor_type:
        """
        Calculates the discrete Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\\log(N))``.

        :param arr: An ndarray representing a vector field.

        :return: An ndarray holding the Fourier transform of the vector field E.
        """
        return ft.fftn(self.astype(arr), dim=self.ft_axes)

    def ift(self, arr: array_like) -> tensor_type:
        """
        Calculates the inverse Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\\log(N))``.
        The scaling is so that ``E == self.ift(self.ft(E))``

        :param arr: An ndarray representing a Fourier-transformed vector field.

        :return: An ndarray holding the inverse Fourier transform of the vector field E.
        """
        arr = self.astype(arr)
        return ft.ifftn(arr, dim=self.ft_axes)

    def adjoint(self, mat: array_like) -> tensor_type:
        """
        Transposes the elements of individual matrices with complex conjugation.

        :param mat: The ndarray with the matrices in the first two dimensions.

        :return: An ndarray with the complex conjugate transposed matrices.
        """
        return torch.conj(self.astype(mat).transpose(0, 1))

    def real(self, arr: array_like) -> tensor_type:
        return arr.real

    def mul(self, left_factor: array_like, right_factor: array_like, out: torch.Tensor = None) -> tensor_type:
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
                result = self.astype(out)
                result *= left_factor  # Scalars are assumed to be proportional to the identity matrix
            else:
                result = self.astype(left_factor) * self.astype(right_factor)
        else:
            # Multiply real and imaginary parts separately because PyTorch
            #(a+bi) * (c+di) = ac - bd + i(ad) + i(bc)
            left_factor = self.astype(left_factor).movedim(0, -1).movedim(0, -1)
            right_factor = self.astype(right_factor).movedim(0, -1).movedim(0, -1)
            result = left_factor.real @ right_factor.real
            result -= left_factor.imag @ right_factor.imag
            result = result + 1j * (left_factor.real @ right_factor.imag)
            result += 1j * (left_factor.imag @ right_factor.real)
            result = result.movedim(-1, 0).movedim(-1, 0)
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
            denominator = self.asnumpy(denominator)  # TODO: Keep this in PyTorch
            numerator = self.asnumpy(numerator)  # TODO: Keep this in PyTorch
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

    # def mat3_eigh(self, arr: array_like) -> tensor_type:
    #     """
    #     Calculates the eigenvalues of the 3x3 matrices represented by A and returns a new array of 3-vectors,
    #     one for each matrix in A and of the same dimensions, baring the second dimension. When the first two
    #     dimensions are 3x1 or 1x3, a diagonal matrix is assumed. When the first two dimensions are singletons (1x1),
    #     a constant diagonal matrix is assumed and only one eigenvalue is returned.
    #     Returns an array of one dimension less: 3 x data_shape.
    #     With the exception of the first dimension, the shape is maintained.
    #
    #     :param arr: The set of 3x3 input matrices for which the eigenvalues are requested.
    #               This must be an ndarray with the first two dimensions of size 3.
    #     :return: The set of eigenvalue-triples contained in an ndarray with its first dimension of size 3,
    #              and the remaining dimensions equal to all but the first two input dimensions.
    #     """
    #     arr = self.astype(arr)
    #     arr = arr.permute(*(2 + np.arange(self.grid.ndim)), 0, 1)
    #     result = torch.linalg.eigvalsh(arr)  # gets eigenvalues for matrices in the right-most axes
    #     result = result.permute(-1, *np.arange(self.grid.ndim))
    #     return result

    @staticmethod
    def clear_cache():
        torch.cuda.empty_cache()
        gc.collect()

    def norm(self, arr: array_like) -> float:
        return float(torch.linalg.norm(torch.view_as_real(arr)))
