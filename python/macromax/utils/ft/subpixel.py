"""
Classes and functions to register a subject array to a reference array with subpixel precision. This is based on the
algorithm described in:
Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image registration algorithms,"
Optics Letters. 33, 156-158 (2008).
"""
import numpy as np
from typing import Union, Sequence, Optional

from . import ft_implementation as ft
from macromax.utils.ft.grid import Grid


__all__ = ['register', 'roll', 'roll_ft', 'Registration', 'Reference']

array_like = Union[int, float, Sequence, np.ndarray]


class Registration:
    """A class to represent the result of a registration of reference class."""
    def __init__(self, shift: array_like, factor: complex = 1.0, error: float = 0.0,
                 original_ft: Optional[array_like] = None, original: Optional[array_like] = None,
                 registered_ft: Optional[array_like] = None, registered: Optional[array_like] = None):
        """
        Constructs a registration result.

        :param shift: translation in pixels of the registered with respect to original reference
        :param factor: scaling factor between registered and original reference image
        :param error: The root-mean-square difference after registration. See ```register``` for more details.
        :param original_ft: The Fourier transform of the original image, prior to registration.
        :param original: The original image, prior to registration.
        :param registered_ft: The Fourier transform of the registered image, identical to the original but with a sub-pixel shift.
        :param registered: The registered image, identical to the original but with a sub-pixel shift.
        """
        self.__shift = np.atleast_1d(shift)

        self.__factor = factor
        self.__error = error

        if original_ft is None and original is not None:
            original_ft = ft.fftn(original, axes=np.arange(-self.ndim, 0))
        self.__original_ft = original_ft

        if registered_ft is None and registered is not None:
            registered_ft = ft.fftn(registered, axes=np.arange(-self.ndim, 0))
        self.__registered_ft = registered_ft

    @property
    def shift(self) -> np.ndarray:
        """
        Vector indicating subpixel shift between the original and registration image.
        """
        return self.__shift

    @shift.setter
    def shift(self, new_shift):
        """
        Vector indicating subpixel shift between the original and registration image.
        """
        if self.__registered_ft is not None:  # makes sure the original image exists
            self.__original_ft = self.original_ft
            self.__registered_ft = None  # delete previous registration
        self.__shift = np.atleast_1d(new_shift)

    @property
    def ndim(self) -> int:
        """The number of registration dimensions."""
        return self.shift.size

    @property
    def factor(self):
        """(Complex) scaling factor indicating the ratio between original and registered image."""
        return self.__factor

    @property
    def error(self) -> float:
        """The rms difference between the registered and the original image including rescaling factor."""
        return self.__error

    @property
    def image_ft(self) -> np.ndarray:
        """Fourier transform of the registered and renormalized array."""
        if self.__registered_ft is None:
            if not np.isclose(self.factor, 0.0):
                self.__registered_ft = roll_ft(self.original_ft, -self.shift) / self.factor
            else:
                self.__registered_ft = np.zeros_like(self.original_ft)

        view = self.__registered_ft.view()
        view.setflags(write=False)
        return view

    @property
    def image(self) -> np.ndarray:
        """Registered and renormalized array.
        It is shifted and scaled so that it is as close as possible to the reference. I.e. it minimizes the l2-norm of
        the difference.
        """
        return ft.ifftn(self.image_ft, axes=np.arange(-self.ndim, 0))

    def __array__(self) -> np.ndarray:
        """Return the registered and renormalized image as an array."""
        return self.image

    @property
    def original_ft(self) -> np.ndarray:
        """Fourier transform of the original reference array."""
        if self.__original_ft is None:
            self.__original_ft = roll_ft(self.image_ft, self.shift) * self.factor
        view = self.__original_ft.view()
        view.setflags(write=False)
        return view

    @property
    def original(self) -> np.ndarray:
        """Original reference array."""
        return ft.ifftn(self.original_ft, axes=np.arange(-self.ndim, 0))

    def __str__(self) -> str:
        return f"Registration(shift={self.shift}, factor={self.factor}, error={self.error})"


class Reference:
    """
    Represents a reference 'image' (which can be n-dimensional) to be used to register against.
    Multi-channel arrays should be handled iteratively or by averaging over the channels.
    """
    def __init__(self, reference_image: Optional[array_like] = None, precision: Optional[float] = None,
                 axes: Optional[array_like] = None,
                 ndim: Optional[int] = None, reference_image_ft: Optional[array_like] = None):
        """
        Construct a reference 'image' object.
        If neither `reference_image` or its Fourier transform `reference_image_ft` are specified, a point-reference is
        assumed, where the point is in the first element of the nd-array (top-left corner).

        :param reference_image: The reference image nd-array. As an alternative, its unshifted Fourier transform can
            be specified as reference_image_ft.
        :param precision: (optional) The default sub-pixel precision (default: 1/128).
        :param axes: (optional) The axes to operate on. If not specified, all dimensions of the reference image or its
            Fourier transform are used.
        :param ndim: The number of dimensions is usually determined from . If neither is specified, ndim determines the
            number of dimensions to operate on.
        :param reference_image_ft:
        """
        if reference_image is not None:
            reference_image = np.array(reference_image)
        if precision is None:
            precision = 1/128
        if reference_image_ft is not None:
            reference_image_ft = np.array(reference_image_ft)
        if axes is not None:
            axes = np.asarray(axes, dtype=int)
            ndim = axes.size
        if ndim is None:
            if reference_image_ft is not None:
                ndim = reference_image_ft.ndim
            elif reference_image is not None:
                ndim = reference_image.ndim
        else:
            ndim = int(ndim)
        self.__ndim = ndim
        if axes is None:
            axes = np.arange(-self.ndim, 0, dtype=int)
        self.__roll_axes = axes

        if reference_image_ft is None:
            if reference_image is not None:
                reference_image_ft = ft.fftn(reference_image, axes=self.__roll_axes)
            else:
                reference_image_ft = np.ones(np.ones(self.ndim, dtype=int))

        # Store some working variables
        self.__reference_norm = np.sqrt(np.mean(np.abs(reference_image_ft) ** 2, axis=tuple(self.__roll_axes)))
        self.__reference_ft_conj = np.conj(reference_image_ft) / self.__reference_norm

        self.__precision = precision

    @property
    def ndim(self) -> int:
        return self.__ndim
    
    @property
    def shape(self):
        return np.array(self.__reference_ft_conj.shape)

    def register(self, subject: Union[np.array, None] = None,
                 subject_ft: Union[np.array, None] = None, precision: float = None) -> Registration:
        """
        Register an nd-image to sub-pixel precision.

        Algorithm based on the 2-d implementation of:
        Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
        "Efficient subpixel image registration algorithms," Opt. Lett. 33, 156-158 (2008).

        :param subject: The subject image as an nd-array. If this is not specified, its Fourier transform should be.
        :param subject_ft: (optional) The (non-fftshifted) Fourier transform of the subject image.
        :param precision: (optional) The sub-pixel precision in units of pixel (default: 1/128).
        :return: A Registration object representing the registered image as well as the shift, global phase change, and error.
        """
        if subject_ft is None:
            subject_ft = ft.fftn(subject, axes=self.__roll_axes)
        if precision is None:
            precision = self.__precision

        cross_correlation_ft = self.__reference_ft_conj * subject_ft
        working_shape = np.asarray(cross_correlation_ft.shape)[self.__roll_axes]
        if precision == 0:  # calculate the error without shifting, no fft needed
            location = np.zeros(self.ndim, dtype=int)
            cross_corr_max = np.mean(cross_correlation_ft, axis=tuple(self.__roll_axes))
        else:
            cross_correlation = ft.ifftn(cross_correlation_ft, axes=self.__roll_axes)
            location = np.array(np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape))
            cross_corr_max = cross_correlation[(*location, )]

        if np.isclose(precision - np.round(precision), 0.0):  # integer pixel-shift, do not upsample
            shift = location
            wrapped = location > (working_shape / 2).astype(int)
            shift[wrapped] += working_shape[wrapped]
        else:  # Partial-pixel shift
            # Find the approximate shift by upsampling by an initial factor
            initial_precision = 1/2
            initial_upsampled_shape = (working_shape / initial_precision).astype(int)
            initial_upsampled_center = (initial_upsampled_shape / 2).astype(int)
            # Zero-pad the Fourier transform of the cross correlation
            cross_correlation_x_ft = np.zeros(shape=initial_upsampled_shape, dtype=complex)
            grid = Grid(cross_correlation_ft.shape, center=initial_upsampled_center)
            cross_correlation_x_ft[(*grid,)] = ft.fftshift(cross_correlation_ft)

            # Compute cross-correlation and locate the peak
            cross_correlation = ft.ifftn(ft.ifftshift(cross_correlation_x_ft, axes=self.__roll_axes),
                                         axes=self.__roll_axes)
            location = np.array(np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape))
            cross_corr_max = cross_correlation[(*location, )] / (initial_precision**self.ndim)

            # Obtain shift in original pixel grid from the position of the cross-correlation peak
            shift = (np.mod(location + initial_upsampled_center, initial_upsampled_shape)
                     - initial_upsampled_center) * initial_precision

            if precision < 1/2:  # refine estimate with discrete Fourier transform
                # initial shift estimate in upsampled grid
                shift = np.round(shift / precision) * precision
                cross_corr_shape = np.ceil(1.5 / precision)  # cover 1.5x the voxel width
                dft_shift = np.floor(cross_corr_shape / 2)  # the center of output array is at dft_shift
                # Matrix multiply DFT around the current shift estimate
                cross_correlation_subset = self.__zoom_ft(cross_correlation_ft,
                                                          cross_corr_shape, precision,
                                                          dft_shift + shift / precision
                                                          ) / np.prod(self.shape)
                # Locate maximum and map back to original pixel grid
                location = np.array(np.unravel_index(np.argmax(np.abs(cross_correlation_subset)),
                                                     cross_correlation_subset.shape))
                cross_corr_max = cross_correlation_subset[(*location, )]
                shift += (-location + dft_shift) * precision

        reg_norm_2 = np.mean(np.abs(subject_ft)**2, axis=tuple(self.__roll_axes))
        reg_norm = np.sqrt(reg_norm_2)

        if not np.isclose(cross_corr_max, 0.0):
            factor = reg_norm_2 / (np.conj(cross_corr_max) * self.__reference_norm)
        else:
            factor = np.inf

        cos_theta = np.minimum(1.0, np.abs(np.abs(cross_corr_max) / reg_norm))  # avoid rounding errors
        error = np.sqrt(1 - cos_theta**2)

        return Registration(shift=shift, factor=factor, error=error, original_ft=subject_ft)

    def __zoom_ft(self, input_data_cube, nb_output_pixels, step, offset=None):
        if offset is None:
            offset = np.zeros(self.ndim)

        input_grid = Grid(np.array(input_data_cube.shape)[self.__roll_axes], step=1/step).k.as_flat
        output_grid = Grid(nb_output_pixels, first=-np.array(offset)).as_flat

        zoomed = input_data_cube
        for axis_idx, axis in enumerate(self.__roll_axes):
            fourier_basis_section = np.exp(-1j * output_grid[axis_idx][:, np.newaxis] * input_grid[axis][np.newaxis, :])
            zoomed = (fourier_basis_section @ zoomed.swapaxes(0, axis)).swapaxes(0, axis)

        return zoomed


def roll_ft(subject_ft: np.ndarray, shift: Optional[array_like],
            axes: Union[int, Sequence, np.ndarray, None] = None):
    """
    Rolls (shifting with wrapping around) and nd-array with sub-pixel precision.
    The input and output array are Fourier transformed.

    :param subject_ft: The Fourier transform of the to-be-shifted nd-array.
    :param shift: The (fractional) shift.
    :param axes: Optional int or sequence of ints. The axis or axes along which elements are shifted.
        Unlike numpy's roll function, by default, the left-most axes are used.

    :return: The Fourier transform of the shifted nd-array.
    """
    shift = np.array(shift).ravel()

    translated_ft = subject_ft.copy()  # todo: necessary?

    # if not np.allclose(shift, 0):
    #     if axes is None:
    #         axes = range(-shift.size, 0)
    #     axes = np.asarray(axes, dtype=int)
    #
    #     grid_step = np.ones(subject_ft.ndim, dtype=np.float32) * np.inf
    #     with np.errstate(divide='ignore'):
    #         grid_step[axes] = 1 / shift
    #     grid_k = Grid(subject_ft.shape, step=grid_step).k
    #
    #     for phase_range, step in zip(grid_k, grid_k.step):
    #         if not np.isclose(step, 0):
    #             if translated_ft.dtype not in (np.complex64, np.complex128):
    #                 translated_ft = translated_ft.astype(np.complex64 if translated_ft.dtype == np.float32 else np.complex128)
    #             translated_ft *= np.exp(-1j * phase_range)

    if not np.allclose(shift, 0):
        if axes is None:
            axes = range(-shift.size, 0)
        axes = np.asarray(axes, dtype=int)

        grid_k_step = np.zeros_like(subject_ft.shape, dtype=np.float64)
        grid_k_step[axes] = 2 * np.pi * shift / np.asarray(subject_ft.shape)[axes]

        grid_k = Grid(subject_ft.shape, step=grid_k_step, origin_at_center=False)
        for phase_range, step in zip(grid_k, grid_k.step):
            if not np.isclose(step, 0):
                if translated_ft.dtype not in (np.complex64, np.complex128):
                    translated_ft = translated_ft.astype(np.complex64 if translated_ft.dtype == np.float32 else np.complex128)
                translated_ft *= np.exp(-1j * phase_range)

    return translated_ft


def roll(subject: np.ndarray, shift: Optional[array_like],
         axis: Union[int, Sequence, np.ndarray, None] = None):
    """
    Rolls (shifting with wrapping around) and nd-array with sub-pixel precision.

    :param subject: The to-be-shifted nd-array. Default: all zeros except first element.
    :param shift: The (fractional) shift. Default: all zeros except first element.
    :param axis: Optional int or sequence of ints. The axis or axes along which elements are shifted.
        Unlike numpy's roll function, by default, the left-most axes are used.

    :return: The shifted nd-array.
    """
    shift = np.array(shift).ravel()
    if axis is None:
        axis = range(-shift.size, 0)
    axis = np.atleast_1d(axis).astype(int)

    subject_ft = ft.fftn(subject, axes=axis)
    translated_ft = roll_ft(subject_ft, shift, axes=axis)

    return ft.ifftn(translated_ft, axes=axis)


def register(subject: Optional[array_like] = None, reference: Optional[array_like] = None,
             precision: Optional[float] = None) -> Registration:
    """
    Registers a subject array to a reference array with subpixel precision. This is based on the algorithm described in
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image registration algorithms,"
    Optics Letters. 33, 156-158 (2008).

    :param subject: The subject image array (n-dimensional) .
    :param reference: The optional reference array (n-dimensional).
    :param precision: The registration precision in units of fracional pixels (default).

    :return: A Registration instance describing the registered image, the shift, and the amplitude.
    """
    if reference is None:
        reference = np.zeros_like(subject)
        reference.ravel()[0] = 1
    if subject is None:
        subject = np.zeros_like(reference)
        subject.ravel()[0] = 1
    reference_object = Reference(reference_image=reference, precision=precision)
    return reference_object.register(subject)
