import unittest
import numpy.testing as npt

from macromax.solver import Solution
import macromax.utils as utils
import numpy as np
import scipy.constants as const


class TestSolution(unittest.TestCase):
    def setUp(self):
        self.data_shape = np.array([50, 100, 200])
        self.sample_pitch = np.array([1, 1, 1])
        self.ranges = utils.calc_ranges(self.data_shape, self.sample_pitch)
        source = np.zeros([3, 1, *self.data_shape], dtype=np.complex64)
        thickness = 5
        source[:, 0, 2*thickness, 2*thickness, 2*thickness] = np.array([0.0, 1.0, 0.0])
        dist_in_boundary = np.maximum(0.0,
                                      np.maximum(self.ranges[0][0]+thickness - self.ranges[0],
                                                 self.ranges[0][-1]-thickness - self.ranges[0]) / thickness
                                      )
        permittivity = np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis] \
                       * (1.0 + 0.8j * dist_in_boundary[:, np.newaxis, np.newaxis])
        self.wavelength = 4.0
        self.SOL = Solution(self.ranges, vacuum_wavelength=self.wavelength, epsilon=permittivity,
                            source_distribution=source)

    def test_ranges(self):
        for (rng_sol, rng) in zip(self.SOL.ranges, self.ranges):
            npt.assert_almost_equal(rng_sol, rng)

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

    # def test_solve(self):
    #     self.fail()
