from __future__ import annotations

import numpy as np
from typing import Union, Optional, Callable, Sequence, Iterable, Generator
from numbers import Complex, Real
import functools
from inspect import signature

from . import ft
from macromax.utils.ft.grid import Grid


__all__ = ['Beam', 'BeamSection']

array_type = Union[Complex, Sequence, np.array]
array_or_callable = Union[array_type, Callable[[array_type, array_type], array_type]]


class BeamSection:
    def __init__(self, grid: Grid, propagation_axis: int = -3,
                 vacuum_wavenumber: Optional[Complex] = 1.0, vacuum_wavelength: Optional[Complex] = None,
                 background_permittivity: Optional[Complex] = 1.0, background_refractive_index: Optional[Complex] = None,
                 field: Optional[array_or_callable] = None, field_ft: Optional[array_or_callable] = None,
                 dtype=None, vectorial: Optional[bool] = None):
        """
        Represents a single transversal section of the wave equation solution that is propagated using the beam-propagation method.
        Its main purpose is to propagate the field from one plane to the next using the ```propagate(...)``` method,
        for use in the ```Beam``` class.
        The field can be scalar or vectorial. In the latter case, the polarization must be represented by axis -4 in the
        field and field_ft arrays. Singleton dimensions must be added for lower-dimensional calculations.

        :param grid: The regularly-spaced grid at which the field values are defined. This grid can have up to three
            dimensions. By default, the third dimension from the right is the propagation dimension.
            If the grid has less than 3 dimensions, new dimensions will be added on the left.
        :param propagation_axis: The grid index of the propagation axis and that of the longitudinal polarization.
            Default: -3
        :param vacuum_wavenumber: (optional) The vacuum wavenumber in rad/m, can also be specified as wavelength.
        :param vacuum_wavelength: (optional) Vacuum wavelength, alternative for wavenumber in units of meter.
        :param background_permittivity: (optional) The homogeneous background permittivity as a scalar, default 1.
            Heterogeneous distributions are specified when querying the results from this object.
        :param background_refractive_index: (optional) Alternative to the above, the square of the permittivity.
        :param field: (optional) The field specification (at propagation distance 0).
            If it is vectorial, the polarization axis should be 4th from the right (-4).
        :param field_ft: (optional) Alternative field specification as its Fourier transform (not fftshifted).
            If the number of dimensions of field or field_ft equals that of the grid, a singleton dimension is prepended on
            the left to indicate a scalar field. Otherwise, the polarization_axis indicates whether this is a vectorial
            calculation or not. If it is vectorial, the polarization axis should be 4th from the right (-4).
        :param dtype: (optional) The complex dtype of the returned results. Default: that of field or field_ft.
        :param vectorial: (optional) Indicates scalar (False) or vectorial = polarized (True) calculations.
            Default: consisted with the size of the polarization axis of the field or field_ft array.
        """
        self.__propagation_axis = ((propagation_axis if propagation_axis is not None else 0) % 3) - 3
        # Expand the grid to 3 dimensions, collapse the propagation axis, and make sure that it is immutable and non-flat.
        slice_shape = [*([1] * (3 - grid.ndim)), *grid.shape]
        slice_shape[self.propagation_axis] = 1
        slice_step = [*([0] * (3 - grid.ndim)), *grid.step]
        slice_center = [*([0] * (3 - grid.ndim)), *grid.center]
        slice_center[self.propagation_axis] = 0.0
        self.__grid = Grid(shape=slice_shape, step=slice_step, center=slice_center)
        self.__transverse_grid = None
        # Define a Fourier-space shift in case the grid is not centered at 0.
        self.__shift_ft = np.exp(1j * sum(k * f for k, f in zip(self.grid.k, self.grid.first)))  # Used to fftshift and ifftshift in Fourier space
        # Define the transverse axes
        self.__transverse_ft_axes = np.where(self.grid.shape > 1)[0] - self.grid.ndim
        # Define the wavenumber and background refractive index
        if vacuum_wavelength is not None:
            vacuum_wavenumber = 2 * np.pi / vacuum_wavelength
        self.__vacuum_wavenumber = vacuum_wavenumber
        if background_refractive_index is None:
            background_refractive_index = background_permittivity ** (1 / 2)
        self.__background_refractive_index = background_refractive_index

        # The state is kept in Fourier space, fftshifted, and a scalar for lazy calculations
        self.__field_ft = None  # Default to no field
        self.__propagation_relative_distance = 0.0  # relative distance since last update of self.__field_ft

        # Pre-calculate the normalized k-vectors for this section, and
        # the exponent of the unit-propagator (used in self.__calc_propagator_ft() method).
        k_projs = [np.zeros([1] * 3)] * 3  # Number of placeholder 0s equal to the number of polarizations if vectorial
        for axis in self.__transverse_ft_axes:
            k_projs[axis] = self.grid.k[axis] / self.wavenumber
        k_projs[self.propagation_axis] = np.lib.scimath.sqrt(1.0 - sum(_ ** 2 for _ in k_projs)).real  # TODO: Keep imaginary part?
        k_vectors = np.stack(np.broadcast_arrays(*k_projs))  # 4D array with in the first (left-most) dimension the k-vector and the latter 3 corresponding to the grid
        # Normalize the k-vector direction vectors (with axes sorted matching the vectorial field as specified)
        self.__k_dir = k_vectors / np.linalg.norm(k_vectors, axis=0)  # Normalizing also the non-propagating

        # Set start values
        self.__dtype = dtype
        if field_ft is not None:
            self.field_ft = field_ft
        elif field is not None:
            self.field = field
        elif vectorial is not None:
            self.__field_ft = np.zeros([1 + 2*vectorial, *self.grid.shape], dtype=self.__dtype)
        else:
            raise TypeError('If neither field nor field_ft are specified, then vectorial should be specified.')

    @property
    def grid(self) -> Grid:
        """The real space sampling grid."""
        return self.__grid

    @property
    def transverse_grid(self) -> Grid:
        """
        The real space sampling grid of the transverse slice at the center.
        Its 3D shape is the same as the full 3D grid except for the propagation axis which has shape 1.
        """
        if self.__transverse_grid is None:
            transverse_shape = self.grid.shape
            transverse_shape[self.propagation_axis] = 1
            self.__transverse_grid = Grid(shape=transverse_shape, step=self.grid.step, center=self.grid.center)
        return self.__transverse_grid

    @property
    def propagation_axis(self) -> int:
        """
        The propagation axis as a negative index in the inputs and outputs.
        This also corresponds to the longitudinal polarization.
        """
        return self.__propagation_axis

    @property
    def polarization_axis(self) -> int:
        """The polarization axis as a negative index in the inputs and outputs. This is currently fixed to -4."""
        return -self.grid.ndim - 1

    @property
    def vectorial(self) -> bool:
        """Returns True if this is a vectorial beam propagation, and False if it is scalar."""
        return self.shape[self.polarization_axis] > 1

    @property
    def shape(self) -> np.ndarray:
        """
        The shape of field (```BeamSection.field``` or ```BeamSection.field_ft```) of a single slice,
        including the polarisation dimension.
        I.e.: [3, 1, y, x], [3, 1, 1, x], [3, 1, 1, 1], [1, 1, y, x], [1, 1, 1, x], [1, 1, 1, 1], [3, z, 1, x], or ...
        Dimensions on the left are added as broadcast.
        """
        return np.asarray(self.__field_ft.shape)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the values returned by the ```BeamSection.field``` and ```BeamSection.field_ft``` methods."""
        return self.shape.size

    @property
    def dtype(self):
        """The data type of the field return by the ```BeamSection.field``` and ```BeamSection.field_ft``` methods."""
        return self.__field_ft.dtype

    @property
    def vacuum_wavenumber(self) -> Real:
        """The vacuum wavenumber of the wave being propagated."""
        return self.__vacuum_wavenumber

    @property
    def wavenumber(self) -> Real:
        """The wavenumber of the wave being propagated in the background material."""
        return self.vacuum_wavenumber * self.background_refractive_index

    @property
    def vacuum_wavelength(self) -> Real:
        """The vacuum wavelength of the wave being propagated."""
        return 2 * np.pi / self.__vacuum_wavenumber

    @property
    def wavelength(self) -> Real:
        """The wavelength of the wave being propagated in the background material."""
        return self.vacuum_wavelength / self.background_refractive_index

    @property
    def background_permittivity(self) -> Real:
        """The permittivity of the background medium.
        This is the permittivity that is assumed between scattering events."""
        return self.__background_refractive_index ** 2

    @property
    def background_refractive_index(self) -> Real:
        """The refractive index of the background medium.
        This is the refractive index that is assumed between scattering events."""
        return self.__background_refractive_index

    @property
    def field(self) -> np.array:
        """
        The electric field in the current section at the sample points are given by ```BeamSection.transverse_grid```.
        The dimension to the left of that indicates the polarization. Scalar propagation was used if it is a singleton,
        vectorial propagation otherwise.
        """
        if np.any(self.grid.shape > 1):
            field_ft = self.field_ft
            if not np.allclose(field_ft, 0.0):  # todo: Is this check a good trade-off or should we remove it?
                return ft.ifftn(field_ft * self.__shift_ft, axes=self.__transverse_ft_axes % field_ft.ndim)  # Transforms right-most (transverse) axes, todo: Use CZT?
            else:
                return np.zeros_like(field_ft)
        else:
            return self.field_ft  # The Inverse Fourier transform of a scalar is a that scalar

    @field.setter
    def field(self, new_field: array_or_callable):
        """
        Set the electric field in the current section at the sample points are given by ```BeamSection.transverse_grid```.
        If a callable function is specified as input argument, it must take the 3 arguments, z, y, and x, of which the
        one along the propagation axis is to be ignored. This function must return an array that can be broadcast to
        ```self.grid.shape'''.
        For vectorial calculations, the polarization axis should be 4th from the right (-4).
        Scalar propagation is assumed if it is a singleton or not specified, otherwise vectorial propagation will be used.

        Note that only propagating modes will be kept.
        Input argument are broadcasts in real space, not in Fourier space!
        """
        if isinstance(new_field, Callable):
            nb_params = len(signature(new_field).parameters)
            new_field = new_field(*self.transverse_grid[-nb_params:])
        new_field = np.asarray(new_field)  # This does not make a copy of the input argument!
        while new_field.ndim < self.grid.ndim + 1:  # Add singleton dimensions on the left, even if scalar
            new_field = new_field[np.newaxis, ...]
        if np.any(self.transverse_grid.shape > 1):  # A Fourier transform will be required
            if np.any(new_field.shape[-self.transverse_grid.ndim:] != self.transverse_grid.shape):  # Broadcast to calculation shape before doing FFT (TODO: zero pad after FFT instead for efficiency of uniform waves?)
                calc_shape = [*new_field.shape[:-self.transverse_grid.ndim], *self.transverse_grid.shape]
                new_field = np.broadcast_to(new_field, calc_shape)
            new_field_ft = ft.fftn(new_field, axes=self.__transverse_ft_axes % new_field.ndim) * self.__shift_ft.conj()  # Transforms 2 right-most axes, todo: Use CZT?
            self.__update_field_ft(new_field_ft)  # shape and dtype already correct
        else:  # The Fourier transform of a scalar is simply that scalar
            new_field_ft = new_field
            self.field_ft = new_field_ft  # Makes a copy, fixes shape and dtype

    def __array__(self) -> np.ndarray:
        """Returns the current field values. This is the same as BeamSection.field """
        return self.field

    @functools.lru_cache(maxsize=4)
    def __calc_propagator_ft(self, propagation_relative_distance: Real) -> np.ndarray:
        """
        Calculates the Fourier-space propagator for a given propagation distance.

        :param propagation_relative_distance: The distance in metric units.

        :return: A complex nd-array with the propagator factors for each (ifftshifted) Fourier component.
        """
        return np.exp(1j * propagation_relative_distance * self.wavenumber * self.__k_dir[self.propagation_axis])

    @property
    def field_ft(self) -> np.array:
        """
        The electric field in k-space, sampled at the grid specified by ```BeamSection.k``` (i.e. not fftshifted).
        The shape of the polarisation dimension on the left (axis -self.grid.ndim-1) will be 3 when doing a vectorial
        calculation and 1 when doing a scalar calculation.
        """
        if not np.isclose(self.__propagation_relative_distance, 0.0):  # prop distance can be negative
            # Update the field for the travelled distance
            self.__field_ft *= self.__calc_propagator_ft(self.__propagation_relative_distance)
            self.__propagation_relative_distance = 0.0  # reset relative distance

        return self.__field_ft

    @field_ft.setter
    def field_ft(self, new_field_ft: array_or_callable):
        """
        The electric field in k-space, sampled at the grid specified by ```BeamSection.transverse_grid.k``` (i.e. not
        fftshifted, with the origin in the first element). When specified as a function, it should take the three spatial
        frequencies: kz, ky, and kx, of which the one along the propagation axis must be ignored. The resulting array or
        scalar should be broadcastable to  ```BeamSection.transverse_grid.k.shape```.
        The shape of the polarisation dimension on the left (axis -4) must be 3 when doing a vectorial calculation and
        1 when doing a scalar calculation. If the number of dimensions is not larger than that of the grid, a scalar
        calculation is assumed.

        Input argument broadcasts in Fourier space, not in real space!
        Note that only propagating modes will be kept.

        :param new_field_ft: The Fourier transform of the field as an ```numpy.ndarray```, something that can be
            converted to it, or a callable function that takes as argument the k-space grid coordinates and returns an
            ```numpy.ndarray```.
        """
        if isinstance(new_field_ft, Callable):
            nb_params = len(signature(new_field_ft).parameters)
            new_field_ft = new_field_ft(*self.transverse_grid.k[-nb_params:])
        new_field_ft = np.asarray(new_field_ft)
        # Ensure that the representation is complex
        if np.isrealobj(new_field_ft):
            new_field_ft = new_field_ft.astype(np.complex64 if new_field_ft.dtype == np.float32 else np.complex128)
        # Make sure that it has a polarization axis, even if it is a scalar field.
        while new_field_ft.ndim < self.grid.ndim + 1:
            new_field_ft = new_field_ft[np.newaxis, ...]
        # Broadcast to calculation shape if needed
        if np.any(new_field_ft.shape[-self.transverse_grid.ndim:] != self.transverse_grid.shape):
            calc_shape = (*new_field_ft.shape[:-self.transverse_grid.ndim], *self.transverse_grid.shape)
            new_field_ft = np.broadcast_to(new_field_ft, calc_shape)
        # Make writable copy
        new_field_ft = new_field_ft.copy()

        self.__update_field_ft(new_field_ft)

    def __transverse_projection_field_ft(self, field_ft: np.ndarray) -> np.ndarray:
        """
        Returns the Fourier transform of the transverse projection of a Fourier transformed field.

        :param field_ft: The Fourier transformed version of the to-be-projected field.

        :return: The Fourier transform of the transverse projected field.
        """
        # Useful definitions for below
        vectorial = field_ft.shape[self.polarization_axis] > 1

        if vectorial:
            # Project field onto the propagation direction using a dot-product and keep dims
            field_longitudinal_project = np.einsum('...i,...i',
                                                   np.moveaxis(self.__k_dir, self.polarization_axis, -1),
                                                   np.moveaxis(field_ft, self.polarization_axis, -1))
            field_longitudinal_project = np.expand_dims(field_longitudinal_project, self.polarization_axis)
            longitudinal_field = self.__k_dir * field_longitudinal_project
            transverse_field = field_ft - longitudinal_field
            return transverse_field
        else:
            return field_ft

    def __update_field_ft(self, new_field_ft: np.ndarray):
        """
        Updates underlying representation of the field by setting a new Fourier transform of it at the current position.
        Note that the input argument may be overwritten!

        :param new_field_ft: The Fourier transform of the field as an ```numpy.ndarray```.
        """
        self.__field_ft = self.__transverse_projection_field_ft(new_field_ft)
        self.__propagation_relative_distance = 0.0  # reset relative distance

    def propagate(self, distance: float,
                  permittivity: Optional[array_or_callable] = None,
                  refractive_index: Optional[array_or_callable] = None) -> BeamSection:
        """
        Propagates the beam section forward by a distance `distance`, optionally through a heterogeneous material
        with refractive index or permittivity as specified. If no material properties are specified, the homogeneous
        background medium is assumed. If the propagation distance is negative, time-reversal is assumed. I.e. the
        phases changes in the opposite direction and attenuation becomes gain.

        The current BeamSection is updated and returned.

        :param distance: The distance (in meters) to propagate forward.
        :param permittivity: The permittivity at the spatial grid points or as a function of (y, x).
        :param refractive_index: The refractive index at the spatial grid points or as a function of (y, x).

        :return: The BeamSection propagated to the new position.
        """
        # Obtain the refractive index distribution that we will be propagating through
        if refractive_index is None and permittivity is not None:
            if isinstance(permittivity, Callable):
                permittivity = permittivity(*self.grid)
            refractive_index = np.lib.scimath.sqrt(permittivity)
        elif isinstance(refractive_index, Callable):
            refractive_index = refractive_index(*self.grid)
        # refractive_index should now be array_like or None

        if distance < 0:  # back-propagate the specified distance (lazily calculated)
            self.__propagation_relative_distance += distance

        if refractive_index is not None:
            refractive_index = np.asarray(refractive_index)
            # Convert identical values to a scalar
            if not np.isscalar(refractive_index) and np.allclose(refractive_index, refractive_index.ravel()[0]):
                refractive_index = refractive_index.ravel()[0]
            delta_n = refractive_index / self.__background_refractive_index - 1
            # Interact with the material, either in real space or in Fourier space.
            if not np.isscalar(delta_n):
                # interact locally, simulating a thickness equal to distance (resets __propagation_relative_distance)
                self.field *= np.exp(1j * self.wavenumber * delta_n * distance)  # Convolves self.__field_ft internally
            elif not np.isclose(delta_n, 0.0):  # interact directly in Fourier space (if there would be any effect)
                self.__field_ft *= np.exp(1j * self.wavenumber * delta_n * distance)  # No need to reset the self.__propagation_relative_distance!

        if distance > 0.0:  # propagate over the given distance (lazily calculated)
            self.__propagation_relative_distance += distance

        return self  # for convenience


class Beam:
    def __init__(self, grid: Grid, propagation_axis: Optional[int] = None,
                 vacuum_wavenumber: Optional[Complex] = 1.0, vacuum_wavelength: Optional[Real] = None,
                 background_permittivity: Optional[Complex] = 1.0, background_refractive_index: Optional[Complex] = None,
                 field: Optional[array_or_callable] = None, field_ft: Optional[array_or_callable] = None):
        """
        Represents the wave equation solution that is propagated using the beam-propagation method. The grid must be one
        dimension higher than that of the beam sections. The field or field_ft argument specify sections as for the
        associated BeamSection object.

        :param grid: The regularly-spaced grid at which to calculate the field values. This grid includes the
            propagation dimension (indicated the propagation_axis argument).
        :param propagation_axis: The propagation axes. Default: -grid.ndim.
        :param vacuum_wavenumber: (optional) The vacuum wavenumber in rad/m, can also be specified as wavelength.
        :param vacuum_wavelength: (optional) Vacuum wavelength, alternative for wavenumber in units of meter.
        :param background_permittivity: (optional) The homogeneous background permittivity as a scalar, default 1.
            Heterogeneous distributions are specified when querying the results from this object.
        :param background_refractive_index: (optional) Alternative to the above, the square of the permittivity.
        :param field: (optional) The field specification at propagation distance 0 as specified on the transverse
            dimensions of the grid.
        :param field_ft: (optional) Alternative field specification as its fft (not fftshifted).
        """
        # Expand the grid to 3 dimensions on the left and make sure that it is immutable and non-flat.
        propagation_axis = (propagation_axis % grid.ndim if propagation_axis is not None else 0) - grid.ndim
        grid_shape = [*([1] * (3 - grid.ndim)), *grid.shape]
        grid_step = [*([0] * (3 - grid.ndim)), *grid.step]
        grid_center = [*([0] * (3 - grid.ndim)), *grid.center]
        self.__grid = Grid(shape=grid_shape, step=grid_step, center=grid_center)

        beam_section = BeamSection(grid=self.grid, propagation_axis=propagation_axis,
                                   vacuum_wavenumber=vacuum_wavenumber, vacuum_wavelength=vacuum_wavelength,
                                   background_permittivity=background_permittivity, background_refractive_index=background_refractive_index,
                                   field=field, field_ft=field_ft)
        # Propagate (back) from the origin to just before the first slice. Note that this is as if in the background medium!
        beam_section.propagate(distance=self.grid.first[beam_section.propagation_axis]
                                        - self.grid.step[beam_section.propagation_axis])  # The field is lazily calculated later
        self.__beam_section = beam_section

    @property
    def grid(self) -> Grid:
        """
        The grid of sample points at which the beam's field is calculated and at which the material properties are
        defined.
        """
        return self.__grid

    @property
    def propagation_axis(self) -> int:
        """The propagation axis index in the polarization vector, i.e. the longitudinal polarization."""
        return self.__beam_section.propagation_axis

    @property
    def vectorial(self) -> bool:
        """Returns True if this is a vectorial beam propagation, and False if it is scalar."""
        return self.__beam_section.vectorial

    @property
    def shape(self) -> np.ndarray:
        """
        The shape of the returned ```field``` or ```field_ft``` array.
        """
        return np.array((*self.__beam_section.shape[:-3], *self.grid.shape))

    @property
    def dtype(self):
        """The data type of the field return in the iterator and by the ```Beam.field``` method."""
        return self.__beam_section.dtype

    def beam_section_at_exit(self, permittivity=None, refractive_index=None) -> BeamSection:
        """
        Propagates the field through the calculation volume and returns the beam section at the exit surface.

        :param permittivity: (optional) The permittivity distribution within the calculation volume.
            Default: the background permittivity, unless refractive_index is specified.
        :param refractive_index: (optional) The refractive index distribution within the calculation volume.
            Default: the background refractive index, unless permittivity is specified.

        :return: The BeamSection object after propagating through the calculation volume.
        """
        beam_section = None  # This should never be returned, self.__iter__() should always return at least 1 section.
        for beam_section in self.__iter__(permittivity=permittivity, refractive_index=refractive_index):
            pass

        if beam_section is None:  # No iteration was required, just use the BeamSection at the entrance
            beam_section = self.__beam_section

        return beam_section

    def __iter__(self, permittivity: Optional[array_type] = None, refractive_index: Optional[array_type] = None) \
            -> Generator[BeamSection, None, None]:
        """
        Iterator returning BeamSections for propagation in an arbitrary medium (by default homogeneous)
        A BeamSection object is generated _after_ propagating through each section of material.

        :param permittivity: (optional) The (complex) permittivity (distribution) that the beam traverses prior to
            yielding each field-section. This should be a sequence/generator of permittivity slices or None.
        :param refractive_index: (optional) Alternative specification of the permittivity squared.
            This should be a sequence/generator of refractive index slices or None.

        :return: A Generator object producing BeamSections.
        """
        # Turn None-arguments into None-Generators:
        if permittivity is None:
            permittivity = (None for _ in range(self.grid.shape[self.propagation_axis]))
        if refractive_index is None:
            refractive_index = (None for _ in range(self.grid.shape[self.propagation_axis]))

        beam_section = self.__beam_section
        distance = self.grid.step[self.propagation_axis]

        # Determine the distances traveled for each section, averaging the spacings before and after it.
        for p, n in zip(permittivity, refractive_index):
            yield beam_section.propagate(distance=distance, permittivity=p, refractive_index=n)

    def field(self, permittivity: Union[Sequence, Iterable] = None, refractive_index: Union[Sequence, Iterable] = None,
              out: array_type = None) -> np.ndarray:
        """
        Calculates the field at the grid points self.grid for the specified (or default)
        permittivity / refractive index.

        :param permittivity: (optional) The 3D permittivity distribution as a sequence or iterable.
        :param refractive_index: (optional) The 3D refractive index distribution as a sequence or iterable.
        :param out: (optional) The output array where to store the result.

        :return: A ```numpy.ndarray``` of which the final four dimensions are polarization, z, y, and x.
        """
        if out is None:
            out = np.zeros(self.shape, dtype=self.dtype)  # TODO: Use empty and zero-pad as needed for efficiency
        out_t = np.moveaxis(out, self.propagation_axis, 0)  # A view into out
        for _, beam_section in enumerate(self.__iter__(permittivity=permittivity, refractive_index=refractive_index)):
            out_t[_] = np.moveaxis(beam_section.field, self.propagation_axis, 0)[0]  # Assign to view
        return out  # Return the underlying array

    def __array__(self) -> np.ndarray:
        """Returns all field values. This is the same as Beam.field()."""
        return self.field()

