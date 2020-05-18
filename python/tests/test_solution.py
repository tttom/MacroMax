import unittest
import numpy.testing as npt

from macromax.solver import Solution, solve
from macromax.utils.array import Grid, calc_ranges
import numpy as np
import scipy.constants as const


class TestSolution(unittest.TestCase):
    def setUp(self):
        self.data_shape = np.array([50, 100, 200])
        self.sample_pitch = np.array([1, 1, 1])
        self.ranges = calc_ranges(self.data_shape, self.sample_pitch)
        self.grid = Grid(self.data_shape, self.sample_pitch)
        current_density = np.zeros([3, 1, *self.data_shape], dtype=np.complex64)
        thickness = 5
        current_density[:, 0, 2*thickness, 2*thickness, 2*thickness] = np.array([0.0, 1.0, 0.0])
        dist_in_boundary = np.maximum(0.0,
                                      np.maximum(self.grid[0].ravel()[0]+thickness - self.grid[0].ravel(),
                                                 self.grid[0].ravel()[-1]-thickness - self.grid[0].ravel()) / thickness
                                      )
        permittivity = np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis] \
                       * (1.0 + 0.8j * dist_in_boundary[:, np.newaxis, np.newaxis])
        self.wavelength = 4.0
        self.SOL = Solution(self.ranges, vacuum_wavelength=self.wavelength, epsilon=permittivity,
                            current_density=current_density)

    def test_ranges(self):
        for (rng_sol, rng) in zip(self.SOL.ranges, self.ranges):
            npt.assert_almost_equal(rng_sol.ravel(), rng)

    def test_grid(self):
        npt.assert_equal(self.SOL.grid == self.grid, True, err_msg='grid not set correctly')

    def test_shape(self):
        npt.assert_almost_equal(self.SOL.shape, self.data_shape)

    def test_sample_pitch(self):
        npt.assert_almost_equal(self.SOL.sample_pitch, self.sample_pitch)

    def test_volume(self):
        npt.assert_almost_equal(self.SOL.volume, self.sample_pitch * self.data_shape)

    def test_wavenumber(self):
        npt.assert_almost_equal(self.SOL.wavenumber, 2*np.pi/self.wavelength)

    def test_angular_frequency(self):
        npt.assert_almost_equal(self.SOL.angular_frequency / (2*np.pi * const.c / self.wavelength), 1.0)

    def test_wavelength(self):
        npt.assert_almost_equal(self.SOL.wavelength, self.wavelength)

    # def test_field(self):
    #     self.fail()
    #
    # def test_field(self):
    #     self.fail()
    #
    def test_iteration(self):
        npt.assert_almost_equal(self.SOL.iteration, 0)
        self.SOL = self.SOL.__iter__().__next__()
        npt.assert_almost_equal(self.SOL.iteration, 1)
        self.SOL = self.SOL.solve(lambda sol: sol.iteration < 5)
        npt.assert_almost_equal(self.SOL.iteration, 5)
        self.SOL.iteration = 1
        self.SOL = self.SOL.__iter__().__next__()
        npt.assert_almost_equal(self.SOL.iteration, 2)

    def test_last_update_norm(self):
        field_0 = self.SOL.E.copy()
        self.SOL = self.SOL.__iter__().__next__()
        field_1 = self.SOL.E.copy()
        npt.assert_almost_equal(self.SOL.last_update_norm, np.sqrt(np.sum(np.abs(field_1-field_0).flatten() ** 2)) )
        self.SOL = self.SOL.__iter__().__next__()
        field_2 = self.SOL.E.copy()
        npt.assert_almost_equal(self.SOL.last_update_norm, np.sqrt(np.sum(np.abs(field_2-field_1).flatten() ** 2)) )

    def test_residue(self):
        def norm(a):
            return np.sqrt(np.sum(np.abs(a).flatten() ** 2))
        field_0 = self.SOL.E.copy()
        self.SOL = self.SOL.__iter__().__next__()
        field_1 = self.SOL.E.copy()
        npt.assert_almost_equal(self.SOL.residue, norm(field_1-field_0) / norm(field_1))
        self.SOL = self.SOL.__iter__().__next__()
        field_2 = self.SOL.E.copy()
        npt.assert_almost_equal(self.SOL.residue, norm(field_2-field_1) / norm(field_2))

    def test_solve(self):
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
        permittivity = permittivity[np.newaxis, np.newaxis, :]  # Indicate that the material is isotropic with singleton dims

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
        npt.assert_equal(solution.iteration <= 140, True, err_msg='The iteration did not converge as fast as expected.')

        #
        # Display the results
        #
        x_range = solution.grid[0]  # coordinates
        selected = (wavelength * 10 < x_range) & (x_range < x_range[-1] - boundary_thickness - wavelength * 10)
        analytic_B = const.mu_0 * sample_pitch * current_density_amplitude / 2
        analytic_E = analytic_B * const.c

        reference_E = source_polarization * analytic_E * np.exp(1j * k0 * np.abs(x_range)) \
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

        E = solution.E[1, selected]  # Electric field in y
        B = solution.B[2, selected]  # Magnetizing field in z
        H = solution.H[2, selected]  # Magnetizing field in z
        S = solution.S[0, selected]  # Poynting vector in x
        f = solution.f[0, selected]  # Optical force in x

        npt.assert_array_equal(solution.E[[0, 2], :],  0, err_msg='The vector field E is not aligned with the y-axis.')
        npt.assert_array_equal(solution.B[[0, 1], :],  0, err_msg='The vector field B is not aligned with the z-axis.')
        npt.assert_array_almost_equal(B * const.c,  E,
                                      err_msg='The product c.|B| is not almost equal to |E|.', decimal=4)
        npt.assert_array_almost_equal(B / const.mu_0,  H, err_msg='The fraction B/mu_0 is not equal to H.', decimal=14)
        npt.assert_array_equal(solution.S[[1, 2], :],  0, err_msg='The vector field S is not aligned with the x-axis.')
        npt.assert_array_almost_equal(E * H / 2,  S, err_msg='The Poynting vector S is not equal to ExH/2.', decimal=13)

        npt.assert_equal(solution.dtype == np.complex64, True, err_msg='dtype not correctly set')
        npt.assert_equal(solution.E.dtype == np.complex64, True, err_msg='solution.E.dtype not correct')
        npt.assert_equal(solution.B.dtype == np.complex64, True, err_msg='solution.B.dtype not correct')
        npt.assert_equal(solution.D.dtype == np.complex64, True, err_msg='solution.D.dtype not correct')
        npt.assert_equal(solution.H.dtype == np.complex64, True, err_msg='solution.H.dtype not correct')
        npt.assert_equal(solution.S.dtype == np.float32, True, err_msg='solution.S.dtype not correct')
