"""The module providing the pure-python NumPy back-end implementation."""
import os
from typing import Union
from numbers import Complex
import logging

import numpy as np

from macromax.utils import ft
from macromax.backend.__init__ import BackEnd, array_like

log = logging.getLogger(__name__)

array_like = Union[array_like, np.ndarray]


class BackEndNumpy(BackEnd):
    """
    A class that provides methods to work with arrays of matrices or block-diagonal matrices, represented as ndarrays,
    where the first two dimensions are those of the matrix, and the final dimensions are the coordinates over
    which the operations are parallelized and the Fourier transforms are applied.
    """
    def __init__(self, nb_dims: int, grid: ft.Grid, hardware_dtype=np.complex128):
        """
        Construct object to handle parallel operations on square matrices of nb_rows x nb_rows elements.
        The matrices refer to points in space on a uniform plaid grid.

        :param nb_dims: The number of rows and columns in each matrix. 1 for scalar operations, 3 for polarization
        :param grid: The grid that defines the position of the matrices.
        :param hardware_dtype: (optional) The data type to use for operations.
        """
        super().__init__(nb_dims, grid, hardware_dtype)

        if (nb_threads := os.cpu_count()) is not None:
            for _ in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
                os.environ[_] = str(nb_threads)
            try:
                import mkl
                mkl.set_num_threads(nb_threads)
            except (ImportError, TypeError):
                pass
            log.info(f'Set maximum number of threads to {nb_threads}.')
        else:
            nb_threads = 1

        try:
            import pyfftw

            log.debug('Module pyfftw imported, using FFTW.')

            ftflags = ('FFTW_DESTROY_INPUT', 'FFTW_ESTIMATE', )
            # ftflags = ('FFTW_DESTROY_INPUT', 'FFTW_PATIENT', )
            # ftflags = ('FFTW_DESTROY_INPUT', 'FFTW_MEASURE', )
            # ftflags = ('FFTW_DESTROY_INPUT', 'FFTW_EXHAUSTIVE', ) # very slow, little gain in general

            self.__empty_word_aligned = \
                lambda shape, dtype: pyfftw.empty_aligned(shape=shape, dtype=dtype, n=pyfftw.simd_alignment, order='C')

            log.debug('Initializing FFTW''s forward Fourier transform.')
            fft_vec_object = pyfftw.FFTW(self.array_ft_input, self.array_ft_output, axes=self.ft_axes, flags=ftflags,
                                         direction='FFTW_FORWARD', planning_timelimit=None, threads=nb_threads)
            log.debug('Initializing FFTW''s backward Fourier transform.')
            ifft_vec_object = pyfftw.FFTW(self.array_ift_input, self.array_ift_output, axes=self.ft_axes, flags=ftflags,
                                          direction='FFTW_BACKWARD', planning_timelimit=None, threads=nb_threads)
            log.debug('FFTW''s wisdoms generated.')

            # Redefine the default method
            def fft_impl(E):
                if np.any(np.array(E.shape[-self.grid.ndim:]) > 1):
                    assert(E.ndim-2 == self.grid.ndim and np.all(E.shape[-self.grid.ndim:] == self.grid.shape))
                    # assert(E is self.array_ft_input)  # just for debugging
                    if np.all(E.shape == fft_vec_object.input_shape):
                        fft_vec_object(E, self.array_ft_output)
                        return self.array_ft_output
                    log.info('Fourier Transform: Array shape not standard, falling back to default interface.')
                    return ft.fftn(E, axes=self.ft_axes)
                return E.copy()

            # Redefine the default method
            def ifft_impl(E):
                if np.any(np.array(E.shape[2:]) > 1):
                    assert(E.ndim-2 == self.grid.ndim and np.all(E.shape[-self.grid.ndim:] == self.grid.shape))
                    # assert(E is self.array_ift_input) # just for debugging
                    if np.all(E.shape == ifft_vec_object.input_shape):
                        ifft_vec_object(E, self.array_ift_output)
                        return self.array_ift_output
                    log.info('Inverse Fourier Transform: Array shape not standard, falling back to default interface.')
                    return ft.ifftn(E, axes=self.ft_axes)
                return E.copy()
        except ModuleNotFoundError:
            log.debug('Module pyfftw not imported, using stock FFT.')
            self.__empty_word_aligned = lambda shape, dtype: np.empty(shape=shape, dtype=dtype, order='C')
            fft_impl = lambda _: ft.fftn(_, axes=self.ft_axes)
            ifft_impl = lambda _: ft.ifftn(_, axes=self.ft_axes)
        self.__ft = fft_impl
        self.__ift = ifft_impl

    def allocate_array(self, shape: array_like = None, dtype = None, fill_value: Complex = None) -> np.ndarray:
        """Allocates a new vector array of shape grid.shape and word-aligned for efficient calculations."""
        if shape is None:
            shape = [self.vector_length, 1, *self.grid.shape]
        if dtype is None:
            dtype = self.hardware_dtype
        arr = self.__empty_word_aligned(shape, dtype=dtype)
        if fill_value is not None:
            arr[:] = fill_value
        # import traceback
        # log.info(f'Allocating array of shape {shape} and dtype {dtype} with fill value {fill_value} ({arr.nbytes / 1024**3:0.1f}GB)\nat{traceback.format_stack()[-2]}.')
        # log.info(''.join(traceback.format_stack()[:-1]))
        return arr

    def ft(self, arr: array_like) -> np.ndarray:
        """
        Calculates the discrete Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\\log(N))``.

        :param arr: An ndarray representing a vector field.

        :return: An ndarray holding the Fourier transform of the vector field E.
        """
        return self.__ft(arr)  # Defined in __init__ depending on imports

    def ift(self, arr: array_like) -> np.ndarray:
        """
        Calculates the inverse Fourier transform over the spatial dimensions of E.
        The computational complexity is that of a Fast Fourier Transform: ``O(N\\log(N))``.
        The scaling is so that ``E == self.ift(self.ft(E))``

        :param arr: An ndarray representing a Fourier-transformed vector field.

        :return: An ndarray holding the inverse Fourier transform of the vector field E.
        """
        return self.__ift(arr)  # Defined in __init__ depending on imports

    def ldivide(self, denominator: array_like, numerator: array_like = 1.0) -> np.ndarray:
        r"""
        Parallel matrix left division, A^{-1}B, on the final two dimensions of A and B
        result_lm = A_kl \ B_km

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
            return np.asarray(numerator, dtype=self.hardware_dtype) / denominator
        else:
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
            return Y.transpose(old_order)
