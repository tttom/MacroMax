import unittest
import numpy.testing as npt

from macromax.solver import Solution, solve
from macromax.utils.ft import Grid
from macromax.bound import LinearBound
import numpy as np
import scipy.constants as const


class TestSolution(unittest.TestCase):
    def setUp(self):
        self.wavelength = 4
        self.grid = Grid(shape=[50, 100, 200], step=self.wavelength/4)
        self.__SOL = None  # reset

    @property
    def SOL(self) -> Solution:
        if self.__SOL is None:
            current_density = np.zeros([3, *self.grid.shape])
            bound_thickness = 5 * self.wavelength
            center_px = np.asarray(current_density.shape) // 2
            current_density[:, center_px[0], center_px[1], center_px[2]] = np.array([0.0, 1.0, 0.0])
            dist_in_boundary = np.maximum(0.0,
                                          np.maximum(self.grid[0].ravel()[0]+bound_thickness - self.grid[0],
                                                     self.grid[0].ravel()[-1]-bound_thickness - self.grid[0]) / bound_thickness
                                          )
            permittivity = 1.0 + 0.2j * dist_in_boundary
            self.__SOL = Solution(self.grid, vacuum_wavelength=self.wavelength, epsilon=permittivity,
                                  current_density=current_density, dtype=np.complex128)
        return self.__SOL

    def test_grid(self):
        npt.assert_equal(self.SOL.grid == self.grid, True, err_msg='grid not set correctly')

    def test_wavenumber(self):
        npt.assert_almost_equal(self.SOL.wavenumber, 2*np.pi/self.wavelength)

    def test_angular_frequency(self):
        npt.assert_almost_equal(self.SOL.angular_frequency / (2*np.pi * const.c / self.wavelength), 1.0)

    def test_wavelength(self):
        npt.assert_almost_equal(self.SOL.wavelength, self.wavelength)

    def test_iteration(self):
        npt.assert_almost_equal(self.SOL.iteration, 0)
        iterator = self.SOL.__iter__()
        iterator.__next__()
        npt.assert_almost_equal(self.SOL.iteration, 1)
        self.SOL.solve(lambda sol: sol.iteration < 5)
        npt.assert_almost_equal(self.SOL.iteration, 5)
        self.SOL.iteration = 1
        iterator.__next__()
        npt.assert_almost_equal(self.SOL.iteration, 2)
        self.SOL.__iter__().__next__()
        npt.assert_almost_equal(self.SOL.iteration, 1)

    def test_last_update_norm(self):
        field_0 = self.SOL.E.copy()
        self.SOL.__iter__().__next__()
        field_1 = self.SOL.E.copy()
        npt.assert_almost_equal(self.SOL.previous_update_norm, np.sqrt(np.sum(np.abs(field_1 - field_0) ** 2)))
        self.SOL.__iter__().__next__()
        field_2 = self.SOL.E.copy()
        npt.assert_almost_equal(self.SOL.previous_update_norm, np.sqrt(np.sum(np.abs(field_2 - field_1) ** 2)))

    def test_residue(self):
        def norm(a):
            return np.sqrt(np.sum(np.abs(a).flatten() ** 2))
        field_0 = self.SOL.E.copy()
        self.SOL.__iter__().__next__()
        field_1 = self.SOL.E.copy()
        npt.assert_almost_equal(self.SOL.residue, norm(field_1-field_0) / norm(field_1))
        self.SOL.__iter__().__next__()
        field_2 = self.SOL.E.copy()
        npt.assert_almost_equal(self.SOL.residue, norm(field_2-field_1) / norm(field_2))

    def test_set_E(self):
        field_0 = self.SOL.E
        self.SOL.E = field_0
        npt.assert_almost_equal(self.SOL.E, field_0, err_msg='Setting of E-field did not work as expected.')
        self.SOL.solve(lambda _: _.iteration < 10)
        field_sol = self.SOL.E
        self.SOL.E = field_0
        npt.assert_almost_equal(self.SOL.E, field_0, err_msg='Setting of E-field did not work as expected.')
        self.SOL.solve(lambda _: _.iteration < 10)
        npt.assert_almost_equal(self.SOL.E, field_sol,
                                err_msg='Setting of E-field and solving did not work as expected.')

    def test_scalar(self):
        #
        # Define the material properties
        #
        wavelength = 500e-9  # [ m ] In SI units as everything else here
        k0 = 2 * np.pi / wavelength  # [rad / m]
        current_density_amplitude = 1.0  # [ A m^-2 ]

        # Set the sampling grid
        nb_samples = 1024
        sample_pitch = wavelength / 16  # [ m ]  # Sub-sample for display
        boundary_thickness = 10e-6  # [ m ]
        x_range = sample_pitch * np.arange(nb_samples) - boundary_thickness  # [ m ]

        # Define the medium
        fraction_in_boundary = np.maximum((x_range[0] + boundary_thickness) - x_range,
                                          x_range - (x_range[-1] - boundary_thickness)) / boundary_thickness
        fraction_in_boundary = np.maximum(0, fraction_in_boundary)
        extinction_coefficient = 0.1
        refractive_index = 1 + 1j * extinction_coefficient * fraction_in_boundary
        permittivity = refractive_index**2  # [ F m^-1 = C V^-1 m^-1 ]

        #
        # Define the illumination source
        #
        # point source at x = 0
        # current_density = -current_density_amplitude * sample_pitch * (np.abs(x_range) < sample_pitch / 4)
        current_density = np.zeros(x_range.shape)
        source_index = np.argmin(np.abs(x_range))
        current_density[source_index] = current_density_amplitude
        current_density = current_density.astype(np.complex64)  # Somewhat lower precision, but half the memory.

        #
        # Solve Maxwell's equations
        #
        # (the actual work is done in this line)
        solution = solve(x_range, vacuum_wavelength=wavelength, current_density=current_density, epsilon=permittivity,
                         callback=lambda s: s.residue > 1e-6 and s.iteration < 1e4)
        npt.assert_equal(solution.residue < 1e-6, True,
                         err_msg=f'The iteration did not converge as expected {solution.residue} >= 1e-6.')
        npt.assert_equal(solution.iteration <= 70, True,
                         err_msg=f'The iteration did not converge as fast as expected {solution.iteration} > 70.')

        #
        # Check the results
        #
        x_range = solution.grid[0]  # coordinates
        selected = (wavelength * 10 < x_range) & (x_range < x_range[-1] - boundary_thickness - wavelength * 10)
        analytic_B = const.mu_0 * sample_pitch * current_density_amplitude / 2  # The / 2 is because of Ampère's circuital law: half the wave is traveling forward while the other half is traveling backward.
        analytic_E = analytic_B * const.c

        reference_E = analytic_E * np.exp(1j * k0 * np.abs(x_range)) \
                      * np.exp(-extinction_coefficient * k0 * boundary_thickness * fraction_in_boundary**2 / 2)

        # x = boundary_thickness * f
        # attenuation = extinction_coefficient * k0 * boundary_thickness * f**2 / 2
        error_E = solution.E - reference_E
        # print(f'numerical rms: {np.sqrt(np.mean(np.abs(solution.E[:, selected] / (solution.wavenumber**2) )**2))}')
        # print(f' analytic rms: {np.sqrt(np.mean(np.abs(reference_E[:, selected])**2))}')
        npt.assert_almost_equal(np.sqrt(np.mean(np.abs(error_E[:, selected])**2)) / np.sqrt(np.mean(np.abs(solution.E[:, selected])**2)),
                                0, decimal=3, err_msg='Plane wave electric field incorrect.')
        npt.assert_almost_equal(np.sqrt(np.mean(np.abs(error_E)**2)) / np.sqrt(np.mean(np.abs(solution.E)**2)),
                                0, decimal=2, err_msg='Absorption in the boundaries not as expected.')

        E = solution.E[0, selected]  # Electric field in y
        B = solution.B[0, selected]  # Magnetic field in z
        H = solution.H[0, selected]  # Magnetizing field in z

        npt.assert_array_almost_equal(B * const.c,  E,
                                      err_msg='The product c.|B| is not almost equal to |E|.', decimal=4)
        npt.assert_array_almost_equal(B / const.mu_0,  H, err_msg='The fraction B/mu_0 is not equal to H.', decimal=14)

        npt.assert_equal(solution.E.dtype == np.complex64, True, err_msg='solution.E.dtype not correct')
        npt.assert_equal(solution.B.dtype == np.complex64, True, err_msg='solution.B.dtype not correct')
        npt.assert_equal(solution.D.dtype == np.complex64, True, err_msg='solution.D.dtype not correct')
        npt.assert_equal(solution.H.dtype == np.complex64, True, err_msg='solution.H.dtype not correct')
        npt.assert_equal(solution.S.dtype == np.float32, True, err_msg='solution.S.dtype not correct')
        # npt.assert_equal(solution.dtype == np.complex64, True, err_msg='dtype not correctly set')  # todo: backend dependent

    def test_solve_vectorial(self):
        #
        # Define the material properties
        #
        wavelength = 500e-9  # [ m ] In SI units as everything else here
        k0 = 2 * np.pi / wavelength  # [rad / m]
        current_density_amplitude = 1.0  # [ A m^-2 ]
        source_polarization = np.array([0, 1, 0])[:, np.newaxis]  # y-polarized

        # Set the sampling grid
        nb_samples = 1024
        sample_pitch = wavelength / 16  # [ m ]  # Sub-sample for display
        boundary_thickness = 10e-6  # [ m ]
        x_range = sample_pitch * np.arange(nb_samples) - boundary_thickness  # [ m ]

        # Define the medium
        fraction_in_boundary = np.maximum((x_range[0] + boundary_thickness) - x_range,
                                          x_range - (x_range[-1] - boundary_thickness)) / boundary_thickness
        fraction_in_boundary = np.maximum(0, fraction_in_boundary)
        extinction_coefficient = 0.1
        refractive_index = 1 + 1j * extinction_coefficient * fraction_in_boundary
        permittivity = refractive_index**2  # [ F m^-1 = C V^-1 m^-1 ]

        #
        # Define the illumination source
        #
        # point source at x = 0
        # current_density = -current_density_amplitude * sample_pitch * (np.abs(x_range) < sample_pitch / 4)
        current_density = np.zeros(x_range.shape)
        source_index = np.argmin(np.abs(x_range))
        current_density[source_index] = current_density_amplitude
        current_density = source_polarization * current_density[np.newaxis, :]  # [ A m^-2 ]
        current_density = current_density.astype(np.complex64)  # Somewhat lower precision, but half the memory.

        #
        # Solve Maxwell's equations
        #
        # (the actual work is done in this line)
        solution = solve(x_range, vacuum_wavelength=wavelength, current_density=current_density, epsilon=permittivity,
                         callback=lambda s: s.residue > 1e-6 and s.iteration < 1e4)
        npt.assert_equal(solution.residue < 1e-6, True, err_msg=f'The iteration did not converge as expected ({solution.residue} >= 1e-6).')
        npt.assert_equal(solution.iteration <= 70, True, err_msg=f'The iteration did not converge as fast as expected ({solution.iteration} > 70).')

        #
        # Check the results
        #
        x_range = solution.grid[0]  # coordinates
        selected = (wavelength * 10 < x_range) & (x_range < x_range[-1] - boundary_thickness - wavelength * 10)
        analytic_B = const.mu_0 * sample_pitch * current_density_amplitude / 2  # The / 2 is because of Ampère's circuital law: half the wave is traveling forward while the other half is traveling backward.
        analytic_E = analytic_B * const.c

        reference_E = source_polarization * analytic_E * np.exp(1j * k0 * np.abs(x_range)) \
                      * np.exp(-extinction_coefficient * k0 * boundary_thickness * fraction_in_boundary**2 / 2)

        error_E = solution.E - reference_E
        npt.assert_almost_equal(np.sqrt(np.mean(np.abs(error_E[:, selected])**2)) / np.sqrt(np.mean(np.abs(solution.E[:, selected])**2)),
                                0, decimal=3, err_msg='Plane wave electric field incorrect.')
        npt.assert_almost_equal(np.sqrt(np.mean(np.abs(error_E)**2)) / np.sqrt(np.mean(np.abs(solution.E)**2)),
                                0, decimal=2, err_msg='Absorption in the boundaries not as expected.')

        E = solution.E[1, selected]  # Electric field in y
        B = solution.B[2, selected]  # Magnetic field in z
        H = solution.H[2, selected]  # Magnetizing field in z
        S = solution.S[0, selected]  # Poynting vector in x
        f = solution.f[0, selected]  # Optical force in x

        npt.assert_array_equal(solution.E[[0, 2], :],  0, err_msg='The vector field E is not aligned with the y-axis.')
        npt.assert_array_equal(solution.B[[0, 1], :],  0, err_msg='The vector field B is not aligned with the z-axis.')
        npt.assert_array_almost_equal(B * const.c,  E,
                                      err_msg='The product c.|B| is not almost equal to |E|.', decimal=4)
        npt.assert_array_almost_equal(B / const.mu_0,  H, err_msg='The fraction B/mu_0 is not equal to H.', decimal=14)
        npt.assert_array_equal(solution.S[[1, 2], :], 0, err_msg='The vector field S is not aligned with the x-axis.')
        npt.assert_array_almost_equal(E * H / 2,  S, err_msg='The Poynting vector S is not equal to ExH/2.', decimal=13)

        npt.assert_equal(solution.E.dtype == np.complex64, True, err_msg='solution.E.dtype not correct')
        npt.assert_equal(solution.B.dtype == np.complex64, True, err_msg='solution.B.dtype not correct')
        npt.assert_equal(solution.D.dtype == np.complex64, True, err_msg='solution.D.dtype not correct')
        npt.assert_equal(solution.H.dtype == np.complex64, True, err_msg='solution.H.dtype not correct')
        npt.assert_equal(solution.S.dtype == np.float32, True, err_msg='solution.S.dtype not correct')
        # npt.assert_equal(solution.dtype == np.complex64, True, err_msg='dtype not correctly set')  # todo: backend dependent

    def test_solve_anisotropic(self):
        #
        # Define the material properties
        #
        wavelength = 500e-9  # [ m ] In SI units as everything else here
        k0 = 2 * np.pi / wavelength  # [rad / m]
        current_density_amplitude = 1.0  # [ A m^-2 ]
        source_polarization = np.array([0, 1, 0])[:, np.newaxis]  # y-polarized

        # Set the sampling grid
        nb_samples = 1024
        sample_pitch = wavelength / 16  # [ m ]  # Sub-sample for display
        boundary_thickness = 10e-6  # [ m ]
        x_range = sample_pitch * np.arange(nb_samples) - boundary_thickness  # [ m ]

        # Define the medium
        fraction_in_boundary = np.maximum((x_range[0] + boundary_thickness) - x_range,
                                          x_range - (x_range[-1] - boundary_thickness)) / boundary_thickness
        fraction_in_boundary = np.maximum(0, fraction_in_boundary)
        extinction_coefficient = 0.1
        refractive_index = 1 + 1j * extinction_coefficient * fraction_in_boundary
        permittivity = refractive_index**2  # [ F m^-1 = C V^-1 m^-1 ]
        mat = np.eye(3)
        permittivity = permittivity * mat[..., np.newaxis]  # Indicate that the material is isotropic with singleton dims
        permittivity[:, :, fraction_in_boundary <= 0] = np.array([[1.1, 0.0, 0.3], [0.0, 1.0, 0.0], [0.3, 0.0, 1.1]])[..., np.newaxis]

        #
        # Define the illumination source
        #
        # point source at x = 0
        # current_density = -current_density_amplitude * sample_pitch * (np.abs(x_range) < sample_pitch / 4)
        current_density = np.zeros(x_range.shape)
        source_index = np.argmin(np.abs(x_range))
        current_density[source_index] = current_density_amplitude
        current_density = source_polarization * current_density[np.newaxis, :]  # [ A m^-2 ]
        current_density = current_density.astype(np.complex64)  # Somewhat lower precision, but half the memory.

        #
        # Solve Maxwell's equations
        #
        # (the actual work is done in this line)
        solution = solve(x_range, vacuum_wavelength=wavelength, current_density=current_density, epsilon=permittivity,
                         callback=lambda s: s.residue > 1e-6 and s.iteration < 1e4)
        npt.assert_equal(solution.residue < 1e-6, True, err_msg='The iteration did not converge as expected.')
        npt.assert_equal(solution.iteration <= 85, True,
                         err_msg=f'The iteration did not converge as fast as expected ({solution.iteration} > 85).')

        #
        # Check the results
        #
        x_range = solution.grid[0]  # coordinates
        selected = (wavelength * 10 < x_range) & (x_range < x_range[-1] - boundary_thickness - wavelength * 10)
        analytic_B = const.mu_0 * sample_pitch * current_density_amplitude / 2  # The / 2 is because of Ampère's circuital law: half the wave is traveling forward while the other half is traveling backward.
        analytic_E = analytic_B * const.c

        reference_E = source_polarization * analytic_E * np.exp(1j * k0 * np.abs(x_range)) \
                      * np.exp(-extinction_coefficient * k0 * boundary_thickness * fraction_in_boundary**2 / 2)

        # x = boundary_thickness * f
        # attenuation = extinction_coefficient * k0 * boundary_thickness * f**2 / 2
        error_E = solution.E - reference_E
        npt.assert_almost_equal(np.sqrt(np.mean(np.abs(error_E[:, selected])**2)) / np.sqrt(np.mean(np.abs(solution.E[:, selected])**2)),
                                0, decimal=3, err_msg='Plane wave electric field incorrect.')
        npt.assert_almost_equal(np.sqrt(np.mean(np.abs(error_E)**2)) / np.sqrt(np.mean(np.abs(solution.E)**2)),
                                0, decimal=2, err_msg='Absorption in the boundaries not as expected.')

        E = solution.E[1, selected]  # Electric field in y
        B = solution.B[2, selected]  # Magnetic field in z
        H = solution.H[2, selected]  # Magnetizing field in z
        S = solution.S[0, selected]  # Poynting vector in x
        f = solution.f[0, selected]  # Optical force in x

        npt.assert_array_equal(solution.E[[0, 2], :],  0, err_msg='The vector field E is not aligned with the y-axis.')
        npt.assert_array_equal(solution.B[[0, 1], :],  0, err_msg='The vector field B is not aligned with the z-axis.')
        npt.assert_array_almost_equal(B * const.c,  E,
                                      err_msg='The product c.|B| is not almost equal to |E|.', decimal=4)
        npt.assert_array_almost_equal(B / const.mu_0,  H, err_msg='The fraction B/mu_0 is not equal to H.', decimal=14)
        npt.assert_array_equal(solution.S[[1, 2], :], 0, err_msg='The vector field S is not aligned with the x-axis.')
        npt.assert_array_almost_equal(E * H / 2,  S, err_msg='The Poynting vector S is not equal to ExH/2.', decimal=13)

        npt.assert_equal(solution.E.dtype == np.complex64, True, err_msg='solution.E.dtype not correct')
        npt.assert_equal(solution.B.dtype == np.complex64, True, err_msg='solution.B.dtype not correct')
        npt.assert_equal(solution.D.dtype == np.complex64, True, err_msg='solution.D.dtype not correct')
        npt.assert_equal(solution.H.dtype == np.complex64, True, err_msg='solution.H.dtype not correct')
        npt.assert_equal(solution.S.dtype == np.float32, True, err_msg='solution.S.dtype not correct')
        # npt.assert_equal(solution.dtype == np.complex64, True, err_msg='dtype not correctly set')  # todo: backend dependent

    def test_solve_magnetic(self):
        #
        # Define the material properties
        #
        wavelength = 500e-9
        boundary_thickness = 2e-6
        beam_diameter = 5e-6
        plate_thickness = 5e-6
        plate_refractive_index = -1.5  # try making this negative (prepare to be patient though - don't oversample!)

        k0 = 2 * np.pi / wavelength
        grid = Grid(np.ones(2) * 128, wavelength / 4)
        incident_angle = 30 * np.pi / 180

        def rot_Z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        incident_k = rot_Z(incident_angle) * k0 @ np.array([1, 0, 0])
        source_polarization = (rot_Z(incident_angle) @ np.array([0, 1, 1j]) / np.sqrt(2))[:, np.newaxis, np.newaxis]
        current_density = np.exp(1j * (incident_k[0]*grid[0] + incident_k[1]*grid[1]))
        source_pixel = grid.shape[0] - int(boundary_thickness / grid.step[0])
        current_density[:source_pixel, :] = 0
        current_density[source_pixel+1:, :] = 0
        current_density = current_density * np.exp(-0.5*((grid[1] - grid[1].ravel()[grid.shape[0]//3])/(beam_diameter/2))**2)  # beam aperture
        current_density = current_density[np.newaxis, ...]
        current_density = current_density * source_polarization

        # define the plate
        refractive_index = 1 + (plate_refractive_index - 1) * np.ones(grid[1].shape) * (np.abs(grid[0]) < plate_thickness/2)

        # Set the boundary conditions
        bound = LinearBound(grid, thickness=boundary_thickness, max_extinction_coefficient=0.50)

        # The actual work is done here:
        solution = solve(grid, vacuum_wavelength=wavelength, current_density=current_density,
                         refractive_index=refractive_index, bound=bound,
                         callback=lambda s: s.iteration < 1e4 and s.residue > 1e-3, dtype=np.complex64
                         )

        npt.assert_equal(solution.residue < 1e-3, True,
                         err_msg=f'The iteration did not converge as expected ({solution.residue} >= 1e-3).')
        npt.assert_equal(solution.iteration <= 1000, True,
                         err_msg=f'The iteration did not converge as fast as expected ({solution.iteration} > 1000).')
    
        #
        # Check the results
        #
        front = solution.E[:, source_pixel+1:source_pixel+8, :]
        back = solution.E[:, -source_pixel-8:-source_pixel-1, :]

        npt.assert_almost_equal(np.linalg.norm(back.ravel()), 0.0001864, decimal=5,
                                err_msg='Output field not as expected.')
        npt.assert_almost_equal(np.linalg.norm(front.ravel()), 0.0002582, decimal=5,
                                err_msg='Input field not as expected.')

        npt.assert_equal(solution.E.dtype, np.complex64, err_msg='solution.E.dtype not correct')
        npt.assert_equal(solution.B.dtype, np.complex64, err_msg='solution.B.dtype not correct')
        npt.assert_equal(solution.D.dtype, np.complex64, err_msg='solution.D.dtype not correct')
        npt.assert_equal(solution.H.dtype, np.complex64, err_msg='solution.H.dtype not correct')
        npt.assert_equal(solution.S.dtype, np.float32, err_msg='solution.S.dtype not correct')
        # npt.assert_equal(solution.dtype == np.complex64, True, err_msg='dtype not correctly set')  # todo: backend dependent


