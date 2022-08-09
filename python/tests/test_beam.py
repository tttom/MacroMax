import unittest
import numpy.testing as npt

from macromax.utils.beam import Beam, BeamSection
from macromax.utils.array import add_dims_on_right
from macromax.utils import ft, Grid

import numpy as np


class TestBeamSection(unittest.TestCase):
    def setUp(self):
        self.vacuum_wavelength = 500e-9  # in vacuum
        self.n0 = 1.2
        self.epsilon0 = self.n0 ** 2
        self.k0 = self.n0 * 2 * np.pi / self.vacuum_wavelength  # NOT in vacuum

        # Section grids of 1 dimension less than the beam volume
        self.section_grids = [Grid([16, 4], self.vacuum_wavelength / 4, first=[0, 0]),
                              Grid([], []),  # 0-dimensional Grid for 1D propagation
                              Grid([256], self.vacuum_wavelength / 16),  # Note the need to specify the shape as a sequence!
                              Grid([256, 64], self.vacuum_wavelength / 16),
                              Grid(256, [self.vacuum_wavelength / 16, self.vacuum_wavelength / 8]),
                              ]
        self.polarizations = [1, np.array([0, 1, 0]), np.array([0, 0, 1])]

    def test_constructor_field_ft(self):
        for polarization in self.polarizations:
            for grid in self.section_grids:
                desc = f'calculation on {grid.shape}-grid with polarization {polarization}'
                ft_axes = np.arange(-grid.ndim, 0)
                field = np.ones(grid.shape)
                vectorial = np.asarray(polarization).size > 1
                if vectorial:
                    field = add_dims_on_right(polarization, 3) * field
                field_ft = ft.fftn(ft.ifftshift(field, axes=ft_axes), axes=ft_axes % field.ndim) if grid.ndim > 0 else field
                beam_section = BeamSection(grid, vacuum_wavelength=self.vacuum_wavelength, background_permittivity=self.epsilon0, field_ft=field_ft)
                while field.ndim < 3 + 1:  # Check if singleton dimension added automatically
                    field = field[np.newaxis, ...]
                    field_ft = field_ft[np.newaxis, ...]
                npt.assert_array_equal(beam_section.vectorial, vectorial, err_msg=f'BeamSection.field_ft not set correctly for {desc}.')
                npt.assert_almost_equal(beam_section.wavenumber, self.k0, err_msg=f'wavenumber incorrectly set or reported for {desc}.')
                npt.assert_almost_equal(beam_section.vacuum_wavenumber, self.k0 / self.n0, err_msg=f'vacuum wavenumber incorrectly set or reported for {desc}.')
                npt.assert_almost_equal(beam_section.vacuum_wavelength, self.vacuum_wavelength, err_msg=f'vacuum wavelength incorrectly set or reported for {desc}.')
                npt.assert_almost_equal(beam_section.wavelength, self.vacuum_wavelength / self.n0, err_msg=f'wavelength incorrectly set or reported for {desc}.')
                npt.assert_equal(beam_section.propagation_axis, -3, err_msg=f'propagation axis reported incorrectly for {desc}.')
                npt.assert_equal(beam_section.background_permittivity, self.epsilon0, err_msg=f'background permittivity incorrectly set or reported for {desc}.')
                npt.assert_array_equal(beam_section.background_refractive_index, self.n0, err_msg=f'background refractive index incorrectly set or reported for {desc}.')
                npt.assert_equal(beam_section.vectorial, vectorial, err_msg=f'vectorial/scalar incorrectly set or reported for {desc}.')
                npt.assert_equal(beam_section.grid.ndim, 3, err_msg=f'grid incorrectly set or reported as {beam_section.grid} for {desc}.')
                npt.assert_equal(beam_section.grid.shape[3-grid.ndim:], grid.shape, err_msg=f'grid incorrectly set or reported as {beam_section.grid} for {desc}.')
                npt.assert_equal(beam_section.grid.step[3-grid.ndim:], grid.step, err_msg=f'grid incorrectly set or reported as {beam_section.grid} for {desc}.')
                npt.assert_array_equal(beam_section.field_ft, field_ft, err_msg=f'BeamSection.field_ft not set correctly for {desc}.')
                npt.assert_array_equal(beam_section.field, field, err_msg=f'BeamSection.field not set correctly for {desc}.')

    def test_constructor_field(self):
        for polarization in self.polarizations:
            for grid in self.section_grids:
                desc = f'calculation on {grid.shape}-grid with polarization {polarization}'
                ft_axes = np.arange(-grid.ndim, 0)
                field = np.ones(grid.shape)
                vectorial = np.asarray(polarization).size > 1
                if vectorial:
                    field = add_dims_on_right(polarization, 3) * field
                field_ft = ft.fftn(ft.ifftshift(field, axes=ft_axes), axes=ft_axes % field.ndim) if grid.ndim > 0 else field
                beam_section = BeamSection(grid, vacuum_wavelength=self.vacuum_wavelength,
                                           background_permittivity=self.epsilon0, field=field)
                while field.ndim < 3 + 1:  # Check if singleton dimension added automatically
                    field = field[np.newaxis, ...]
                    field_ft = field_ft[np.newaxis, ...]
                npt.assert_array_equal(beam_section.vectorial, vectorial, err_msg=f'BeamSection.field_ft not set correctly for {desc}.')
                npt.assert_almost_equal(beam_section.wavenumber, self.k0, err_msg=f'wavenumber incorrectly set or reported for {desc}.')
                npt.assert_almost_equal(beam_section.vacuum_wavenumber, self.k0 / self.n0, err_msg=f'vacuum wavenumber incorrectly set or reported for {desc}.')
                npt.assert_almost_equal(beam_section.vacuum_wavelength, self.vacuum_wavelength, err_msg=f'vacuum wavelength incorrectly set or reported for {desc}.')
                npt.assert_almost_equal(beam_section.wavelength, self.vacuum_wavelength / self.n0, err_msg=f'wavelength incorrectly set or reported for {desc}.')
                npt.assert_equal(beam_section.propagation_axis, -3, err_msg=f'propagation axis reported incorrectly for {desc}.')
                npt.assert_equal(beam_section.background_permittivity, self.epsilon0, err_msg=f'background permittivity incorrectly set or reported for {desc}.')
                npt.assert_array_equal(beam_section.background_refractive_index, self.n0, err_msg=f'background refractive index incorrectly set or reported for {desc}.')
                npt.assert_equal(beam_section.vectorial, vectorial, err_msg=f'vectorial/scalar incorrectly set or reported for {desc}.')
                npt.assert_equal(beam_section.grid.ndim, 3, err_msg=f'grid incorrectly set or reported as {beam_section.grid} for {desc}.')
                npt.assert_equal(beam_section.grid.shape[3-grid.ndim:], grid.shape, err_msg=f'grid incorrectly set or reported as {beam_section.grid} for {desc}.')
                npt.assert_equal(beam_section.grid.step[3-grid.ndim:], grid.step, err_msg=f'grid incorrectly set or reported as {beam_section.grid} for {desc}.')
                npt.assert_array_equal(beam_section.field_ft, field_ft, err_msg=f'BeamSection.field_ft not set correctly for {desc}.')
                npt.assert_array_equal(beam_section.field, field, err_msg=f'BeamSection.field not set correctly for {desc}.')

    def test_constructor_field_broadcasted(self):
        for polarization in self.polarizations:
            for grid in self.section_grids:
                if grid.ndim == 0:
                    fields = [1.0]
                elif grid.ndim == 1:
                    fields = [1, np.ones(1)]
                else:
                    fields = [np.ones((1, 1)), np.ones((grid.shape[0], 1)), np.ones((1, grid.shape[1]))]
                for field in fields:
                    desc = f'calculation on {grid.shape}-grid with polarization {polarization} with field of shape {np.asarray(field).shape}.'
                    ft_axes = np.arange(-grid.ndim, 0)
                    vectorial = np.asarray(polarization).size > 1
                    if not vectorial or np.asarray(field).ndim == grid.ndim:  # If vectorial, only consider correctly-dimensioned inputs
                        if vectorial:
                            field = add_dims_on_right(polarization, 3) * field
                        beam_section = BeamSection(grid, vacuum_wavelength=self.vacuum_wavelength, background_permittivity=self.epsilon0, field=field)
                        field = np.asarray(field)
                        while field.ndim < 1 + grid.ndim:  # Correct the number of dimensions
                            field = field[np.newaxis, ...]
                        # Broadcast input field
                        if grid.ndim > 0:
                            field = np.broadcast_to(field, (*field.shape[:-grid.ndim], *grid.shape))
                        while field.ndim < 4:
                            field = field[np.newaxis]
                        field_ft = ft.fftn(ft.ifftshift(field, axes=ft_axes), axes=ft_axes % field.ndim) if grid.ndim > 0 else field
                        npt.assert_array_equal(beam_section.vectorial, vectorial, err_msg=f'BeamSection.field_ft not set correctly for {desc}.')
                        npt.assert_almost_equal(beam_section.wavenumber, self.k0, err_msg=f'wavenumber incorrectly set or reported for {desc}.')
                        npt.assert_almost_equal(beam_section.vacuum_wavenumber, self.k0 / self.n0, err_msg=f'vacuum wavenumber incorrectly set or reported for {desc}.')
                        npt.assert_almost_equal(beam_section.vacuum_wavelength, self.vacuum_wavelength, err_msg=f'vacuum wavelength incorrectly set or reported for {desc}.')
                        npt.assert_almost_equal(beam_section.wavelength, self.vacuum_wavelength / self.n0, err_msg=f'wavelength incorrectly set or reported for {desc}.')
                        npt.assert_equal(beam_section.propagation_axis, -3, err_msg=f'propagation axis incorrectly reported for {desc}.')
                        npt.assert_equal(beam_section.background_permittivity, self.epsilon0, err_msg=f'background permittivity incorrectly set or reported for {desc}.')
                        npt.assert_array_equal(beam_section.background_refractive_index, self.n0, err_msg=f'background refractive index incorrectly set or reported for {desc}.')
                        npt.assert_equal(beam_section.vectorial, vectorial, err_msg=f'vectorial/scalar incorrectly set or reported for {desc}.')
                        npt.assert_equal(beam_section.grid.ndim, 3, err_msg=f'grid incorrectly set or reported as {beam_section.grid} for {desc}.')
                        npt.assert_equal(beam_section.grid.shape[3-grid.ndim:], grid.shape, err_msg=f'grid incorrectly set or reported as {beam_section.grid} for {desc}.')
                        npt.assert_equal(beam_section.grid.step[3-grid.ndim:], grid.step, err_msg=f'grid incorrectly set or reported as {beam_section.grid} for {desc}.')
                        npt.assert_array_equal(beam_section.field_ft, field_ft, err_msg=f'BeamSection.field_ft not set correctly for {desc}.')
                        npt.assert_array_equal(beam_section.field, field, err_msg=f'BeamSection.field not set correctly for {desc}.')

    def test_plane_wave_normal_incidence(self):
        for polarization in self.polarizations:
            for grid in self.section_grids:
                desc = f'calculation on {grid.shape}-grid with polarization {polarization}'
                ft_axes = np.arange(-grid.ndim, 0)
                field = np.ones(grid.shape)
                vectorial = np.asarray(polarization).size > 1
                if vectorial:
                    field = add_dims_on_right(polarization, 3) * field
                field_ft = ft.fftn(ft.ifftshift(field, axes=ft_axes), axes=ft_axes % field.ndim) if grid.ndim > 0 else field
                beam_section = BeamSection(grid, vacuum_wavelength=self.vacuum_wavelength, background_permittivity=self.epsilon0, field_ft=field_ft)
                while field.ndim < 3 + 1:  # Check if singleton dimension added automatically
                    field = field[np.newaxis, ...]
                    field_ft = field_ft[np.newaxis, ...]
                # Propagate forward and check
                distance = self.vacuum_wavelength
                beam_section.propagate(distance)
                npt.assert_array_almost_equal(beam_section.field, field * np.exp(1j * self.k0 * distance), err_msg=f'BeamSection.field not propagated correctly for {desc}.')
                npt.assert_array_almost_equal(beam_section.field_ft, field_ft * np.exp(1j * self.k0 * distance), err_msg=f'BeamSection.field_ft not propagated correctly for {desc}.')
                # Reset and try with a fractional distance
                beam_section.propagate(-distance)
                distance = self.vacuum_wavelength * np.pi
                beam_section.propagate(distance)
                npt.assert_array_almost_equal(beam_section.field, field * np.exp(1j * self.k0 * distance), err_msg=f'BeamSection.field not propagated correctly for {desc}.')
                npt.assert_array_almost_equal(beam_section.field_ft, field_ft * np.exp(1j * self.k0 * distance), err_msg=f'BeamSection.field_ft not propagated correctly for {desc}.')

    def test_refractive_index_normal_incidence(self):
        for polarization in self.polarizations:
            for grid in self.section_grids:
                desc = f'calculation on {grid.shape}-grid with polarization {polarization}'
                ft_axes = np.arange(-grid.ndim, 0)
                field = np.ones(grid.shape)
                vectorial = np.asarray(polarization).size > 1
                if vectorial:
                    field = add_dims_on_right(polarization, 3) * field
                field_ft = ft.fftn(ft.ifftshift(field, axes=ft_axes), axes=ft_axes % field.ndim) if grid.ndim > 0 else field
                beam_section = BeamSection(grid, vacuum_wavelength= self.vacuum_wavelength, background_permittivity=self.epsilon0, field_ft=field_ft)
                while field.ndim < 3 + 1:  # Check if singleton dimension added automatically
                    field = field[np.newaxis, ...]
                    field_ft = field_ft[np.newaxis, ...]
                # Propagate forward and check
                distance = self.vacuum_wavelength * np.pi
                refractive_index = 1.5
                phasor = np.exp(1j * self.k0 * (1 + (refractive_index / self.n0 - 1)) * distance)  # first refract, then propagate
                beam_section.propagate(distance, refractive_index=refractive_index)
                npt.assert_array_almost_equal(beam_section.field_ft, field_ft * phasor, err_msg=f'BeamSection.field_ft not propagated correctly through scalar refractive index for {desc}.')
                npt.assert_array_almost_equal(beam_section.field, field * phasor, err_msg=f'BeamSection.field not propagated correctly through scalar refractive index for {desc}.')
                beam_section.propagate(-distance, refractive_index=refractive_index)  # Reset
                distance = self.vacuum_wavelength * np.exp(1)
                refractive_index = 1.5 * np.ones(grid.shape)
                phasor = np.exp(1j * self.k0 * (1 + (refractive_index / self.n0 - 1)) * distance)  # first refract, then propagate
                beam_section.propagate(distance, refractive_index=refractive_index)
                npt.assert_array_almost_equal(beam_section.field_ft, field_ft * phasor, err_msg=f'BeamSection.field_ft not propagated correctly through constant array of refractive indices for {desc}.')
                npt.assert_array_almost_equal(beam_section.field, field * phasor, err_msg=f'BeamSection.field not propagated correctly through constant array of refractive indices for {desc}.')
                beam_section.propagate(-distance, refractive_index=refractive_index)  # Reset
                distance = self.vacuum_wavelength * 1.23
                refractive_index = (1.5 + 0.1j) * np.ones(grid.shape)
                phasor = np.exp(1j * self.k0 * (1 + (refractive_index / self.n0 - 1)) * distance)  # first refract, then propagate
                beam_section.propagate(distance, refractive_index=refractive_index)
                npt.assert_array_almost_equal(beam_section.field_ft, field_ft * phasor, err_msg=f'BeamSection.field_ft not propagated correctly through attenuation for {desc}.')
                npt.assert_array_almost_equal(beam_section.field, field * phasor, err_msg=f'BeamSection.field not propagated correctly through attenuation for {desc}.')

    def test_plane_wave_at_angle(self):
        for polarization in self.polarizations:
            for section_grid in self.section_grids:
                if section_grid.ndim > 0:  # Skip 1-dimensional calculations
                    propagation_axis = -section_grid.ndim-1
                    transverse_ft_axes = np.arange(-section_grid.ndim, 0)
                    # tilt the beam in the first (left-most) dimension of the section grid.
                    kts = np.array([1, -1, 2, section_grid.shape[0] // 8, section_grid.shape[0] // 4]) * 2 * np.pi / section_grid.extent[0]  # Must be periodic in the transverse dimension for the tests below to work
                    for kt in kts:
                        if kt < self.k0:  # Do not check evanescent components
                            kz = np.sqrt(1.0 - (kt / self.k0) ** 2) * self.k0
                            desc = f'calculation on {section_grid.shape}-grid with polarization {polarization} at {np.arcsin(kt / self.k0) * 180 / np.pi:0.1f} degrees.'
                            field = np.exp(1j * kt * section_grid[0])
                            field = np.broadcast_to(field, (*field.shape[:-section_grid.ndim], *section_grid.shape))
                            vectorial = np.asarray(polarization).size > 1
                            if vectorial:  # Rotate polarization vector and change scalar field to vectorial
                                c, s = kz / self.k0, kt / self.k0
                                rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                                if propagation_axis == -1:
                                    order = [0, 2, 1]
                                    rot = rot[order, order]
                                rotated_polarization = (rot @ polarization[:, np.newaxis])[:, 0]
                                field = add_dims_on_right(rotated_polarization, 3 + 1) * field
                            shifted_field = np.array(field)
                            for f, ax in zip(section_grid.first, transverse_ft_axes):
                                if f != 0:
                                    shifted_field = ft.ifftshift(shifted_field, axes=ax)
                            field_ft = ft.fftn(shifted_field, axes=transverse_ft_axes % field.ndim) if section_grid.ndim > 0 else field
                            beam_section = BeamSection(section_grid, vacuum_wavelength= self.vacuum_wavelength,
                                                       background_permittivity=self.epsilon0, field_ft=field_ft,
                                                       propagation_axis=propagation_axis)
                            while field.ndim < 3 + 1:  # When scalar, singleton dimensions should have been added
                                field = field[np.newaxis]
                                field_ft = field_ft[np.newaxis]
                            npt.assert_array_almost_equal(beam_section.field_ft, field_ft, err_msg=f'BeamSection.field_ft not set correctly for {desc}.')
                            npt.assert_array_almost_equal(beam_section.field, field, err_msg=f'BeamSection.field not set correctly for {desc}.')
                            # Propagate forward and check
                            distance = self.vacuum_wavelength
                            beam_section.propagate(distance)
                            phasor = np.exp(1j * kz * distance)  # propagate only
                            npt.assert_array_almost_equal(beam_section.field_ft, field_ft * phasor, err_msg=f'BeamSection.field_ft not propagated correctly for {desc}.')
                            npt.assert_array_almost_equal(beam_section.field, field * phasor, err_msg=f'BeamSection.field not propagated correctly for {desc}.')
                            beam_section.propagate(-distance) # Reset and try with a fractional distance and refractive index
                            distance = self.vacuum_wavelength * np.pi
                            refractive_index = 1.5 + 0.1j
                            beam_section.propagate(distance, refractive_index=refractive_index)
                            phasor = np.exp(1j * (kz  + self.k0 * (refractive_index / self.n0 - 1)) * distance)  # Refraction is applied instantaneously (angle independent), propagation is angle dependent
                            npt.assert_array_almost_equal(beam_section.field_ft, field_ft * phasor, err_msg=f'BeamSection.field_ft not propagated correctly at angle through glass for {desc}.')
                            npt.assert_array_almost_equal(beam_section.field, field * phasor, err_msg=f'BeamSection.field not propagated correctly at angle through glass for {desc}.')
                            beam_section.propagate(-distance, refractive_index=refractive_index)
                            npt.assert_array_almost_equal(beam_section.field_ft, field_ft, err_msg=f'BeamSection.field_ft not propagated back correctly at angle through glass for {desc}.')
                            npt.assert_array_almost_equal(beam_section.field, field, err_msg=f'BeamSection.field not propagated back correctly at angle through glass for {desc}.')


class TestBeam(unittest.TestCase):
    def setUp(self):
        self.vacuum_wavelength = 500e-9  # in vacuum
        self.n0 = 1.2
        self.epsilon0 = self.n0 ** 2
        self.k0 = self.n0 * 2 * np.pi / self.vacuum_wavelength  # NOT in vacuum

        self.grids = [Grid([8, 16], self.vacuum_wavelength / 16),
                      Grid([4, 16], self.vacuum_wavelength / 4, first=[0, 0]),
                      Grid([8, 16, 3], self.vacuum_wavelength / 4),
                      Grid([8, 16, 3], self.vacuum_wavelength / 4, first=[0, 0, 0]),
                      Grid([8], self.vacuum_wavelength / 16),  # Note the need to specify it as a sequence, not scalar!
                      Grid(8, [self.vacuum_wavelength / 16, self.vacuum_wavelength / 8]),
                      ]
        self.polarizations = [1, np.array([0, 1, 0]), np.array([0, 0, 1])]

    def test_constructor(self):
        for grid in self.grids:
            prop_axis = -grid.ndim
            for polarization in self.polarizations:
                desc = f'grid shape {grid.shape} with polarization {polarization}'
                field = 1.0
                vectorial = np.asarray(polarization).size > 1
                if vectorial:
                    if prop_axis == -2:
                        polarization = polarization[[1, 0, 2]]
                    elif prop_axis == -1:
                        polarization = polarization[[2, 1, 0]]
                    field = add_dims_on_right(polarization, 3) * field
                beam = Beam(grid, vacuum_wavelength=self.vacuum_wavelength, background_permittivity=self.epsilon0,
                            field=field, propagation_axis=prop_axis)
                propagated_field = beam.field()
                field = np.broadcast_to(field, [1 + vectorial*2, *([1] * (3 - grid.ndim)), *grid.shape])
                npt.assert_array_almost_equal(propagated_field,
                                              field * np.exp(1j * self.k0 * grid[prop_axis]),
                                              err_msg=f'Propagated field incorrect for {desc}.')

    def test_propagation_along_different_axis(self):
        for grid in self.grids:
            if grid.ndim > 1:
                prop_axis = -grid.ndim + 1
                for polarization in self.polarizations:
                    desc = f'grid shape {grid.shape} with polarization {polarization} propagating along axis {prop_axis}.'
                    field = 1.0
                    vectorial = np.asarray(polarization).size > 1
                    if vectorial:
                        if prop_axis == -2:
                            polarization = polarization[[1, 0, 2]]
                        elif prop_axis == -1:
                            polarization = polarization[[2, 1, 0]]
                        field = add_dims_on_right(polarization, 3) * field
                    beam = Beam(grid, vacuum_wavelength=self.vacuum_wavelength, background_permittivity=self.epsilon0,
                                field=field, propagation_axis=prop_axis)
                    propagated_field = beam.field()
                    field = np.broadcast_to(field, [1 + vectorial*2, *([1] * (3 - grid.ndim)), *grid.shape])
                    npt.assert_array_almost_equal(np.abs(propagated_field), np.abs(field * np.exp(1j * self.k0 * grid[prop_axis])),
                                                  err_msg=f'Absolute value of propagated field incorrect for {desc}.')
                    npt.assert_array_almost_equal(propagated_field, field * np.exp(1j * self.k0 * grid[prop_axis]),
                                                  err_msg=f'Propagated field incorrect for {desc}.')


if __name__ == '__main__':
    unittest.main()

