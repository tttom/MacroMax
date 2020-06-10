import unittest
import numpy as np
import numpy.testing as npt

from macromax.utils.bound import Bound, AbsorbingBound, LinearBound, PeriodicBound
from macromax.utils.array import Grid


class TestBound(unittest.TestCase):
    def test_scalar(self):
        grid = Grid([10], 1e-6)
        bound = LinearBound(grid, thickness=4e-6, max_extinction_coefficient=0.1)
        kappa = np.array([0.100, 0.075, 0.050, 0.025,  0.0, 0.0, 0.025, 0.050, 0.075, 0.100])
        npt.assert_array_almost_equal(bound.extinction, kappa)
        npt.assert_equal(bound.is_electric, True)
        npt.assert_array_almost_equal(bound.electric_susceptibility, 1j*kappa)
        npt.assert_equal(bound.background_permittivity, 1.0)
        npt.assert_array_almost_equal(bound.permittivity, bound.background_permittivity + 1j*kappa)
        npt.assert_equal(bound.is_magnetic, False)
        npt.assert_array_almost_equal(bound.magnetic_susceptibility, 0*kappa)
        npt.assert_array_equal(bound.thickness, np.ones((1, 2)) * 4e-6)

    def test_asymmetric(self):
        grid = Grid([10], 1e-6)
        bound = LinearBound(grid, thickness=[4e-6, 2e-6], max_extinction_coefficient=0.1)
        npt.assert_array_equal(bound.thickness, [[4e-6, 2e-6]])
        kappa = [0.100, 0.075, 0.050, 0.025,  0.0, 0.0, 0.0, 0.0, 0.050, 0.100]
        npt.assert_array_almost_equal(bound.extinction, kappa)
        bound = LinearBound(grid, thickness=4e-6, max_extinction_coefficient=[0.1, 0.2])
        npt.assert_array_equal(bound.thickness, [[4e-6, 4e-6]])
        kappa = [0.100, 0.075, 0.050, 0.025,  0.0, 0.0, 0.050, 0.100, 0.150, 0.200]
        npt.assert_array_almost_equal(bound.extinction, kappa)
        bound = LinearBound(grid, thickness=[4e-6, 2e-6], max_extinction_coefficient=(0.1, 0.2))
        npt.assert_array_equal(bound.thickness, [[4e-6, 2e-6]])
        kappa = [0.100, 0.075, 0.050, 0.025,  0.0, 0.0, 0.0, 0.0, 0.100, 0.200]
        npt.assert_array_almost_equal(bound.extinction, kappa)
        bound = LinearBound(grid, thickness=[4e-6, 0e-6], max_extinction_coefficient=(0.1, 0.2))
        npt.assert_array_equal(bound.thickness, [[4e-6, 0e-6]])
        kappa = [0.100, 0.075, 0.050, 0.025,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        npt.assert_array_almost_equal(bound.extinction, kappa)
        bound = LinearBound(grid, thickness=[0, 0], max_extinction_coefficient=(0.1, 0.2))
        npt.assert_array_equal(bound.thickness, [[0, 0]])
        kappa = [0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        npt.assert_array_almost_equal(bound.extinction, kappa)

    def test_2d(self):
        grid = Grid([2, 10], 1e-6)
        bound = LinearBound(grid, thickness=[4e-6, 2e-6], max_extinction_coefficient=0.1)
        npt.assert_array_equal(bound.thickness, [[4e-6, 2e-6], [4e-6, 2e-6]])
        npt.assert_array_almost_equal(bound.extinction, 0.1)

        bound = LinearBound(grid, thickness=[[0, 0], [4e-6, 2e-6]], max_extinction_coefficient=[0.1, 0.1])
        kappa = [0.100, 0.075, 0.050, 0.025,  0.0, 0.0, 0.0, 0.0, 0.050, 0.100]
        npt.assert_array_equal(bound.thickness, [[0, 0], [4e-6, 2e-6]])
        npt.assert_array_almost_equal(bound.extinction, [kappa])

        bound = LinearBound(grid, thickness=[4e-6, 2e-6], max_extinction_coefficient=[[0, 0], [0.1, 0.1]])
        kappa = [0.100, 0.075, 0.050, 0.025,  0.0, 0.0, 0.0, 0.0, 0.050, 0.100]
        npt.assert_array_equal(bound.thickness, [[4e-6, 2e-6], [4e-6, 2e-6]])
        npt.assert_array_almost_equal(bound.extinction, [kappa, kappa])

        bound = LinearBound(grid, thickness=[[0, 0], [4e-6, 2e-6]], max_extinction_coefficient=[0.1, 0.1])
        kappa = [0.100, 0.075, 0.050, 0.025,  0.0, 0.0, 0.0, 0.0, 0.050, 0.100]
        npt.assert_array_equal(bound.thickness, [[0, 0], [4e-6, 2e-6]])
        npt.assert_array_almost_equal(bound.extinction, [kappa])

    def test_periodic(self):
        grid = Grid([2, 10], 1e-6)
        bound = PeriodicBound(grid)
        npt.assert_array_equal(bound.thickness, [[0, 0], [0, 0]])
        npt.assert_array_equal(bound.is_electric, False)
        npt.assert_array_equal(bound.is_magnetic, False)
        npt.assert_array_equal(bound.electric_susceptibility, 0)
        npt.assert_array_equal(bound.magnetic_susceptibility, 0)


if __name__ == '__main__':
    unittest.main()
