import unittest
import numpy.testing as npt

from macromax.utils.ft.grid import Grid, MutableGrid

import numpy as np


class TestGrid(unittest.TestCase):
    def test_grid_no_shape(self):
        npt.assert_array_equal(Grid(extent=10, step=1), np.arange(-5, 5),
                               err_msg='Grid failed to calculate correct shape.')
        npt.assert_array_equal(Grid(extent=11, step=1), np.arange(-5, 6),
                               err_msg='Grid failed to calculate correct shape.')
        npt.assert_array_equal(Grid(extent=10, step=2), np.arange(-4, 6, 2),
                               err_msg='Grid failed to calculate correct shape or centering.')
        npt.assert_array_equal(Grid(extent=11, step=2), np.arange(-6, 5, 2),
                               err_msg='Grid failed to calculate correct shape or centering.')
        npt.assert_array_equal(Grid(extent=10, step=3), np.arange(-6, 4, 3),
                               err_msg='Grid failed to calculate correct shape or centering.')
        npt.assert_array_equal(Grid(extent=11, step=3), np.arange(-6, 6, 3),
                               err_msg='Grid failed to calculate correct shape or centering.')
        npt.assert_array_equal(Grid(extent=11, step=-3), np.arange(6, -6, -3),
                               err_msg='Grid failed to calculate correct shape or centering for negative step.')
        npt.assert_array_equal(Grid(extent=12, step=3), np.arange(-6, 6, 3),
                               err_msg='Grid failed to calculate correct shape or centering.')
        npt.assert_array_equal(Grid(first=-5, last=5, step=1), np.arange(-5, 5),
                               err_msg='Grid failed to calculate correct shape without specifying extent.')
        npt.assert_array_equal(Grid(first=-5, center=0, step=1), np.arange(-5, 5),
                               err_msg='Grid failed to calculate correct shape without specifying extent.')
        npt.assert_array_equal(Grid(center=0, last=5, step=1), np.arange(-5, 5),
                               err_msg='Grid failed to calculate correct shape without specifying extent.')
        npt.assert_array_equal(Grid(first=-2, last=3, step=1), np.arange(-2, 3, 1),
                               err_msg='Grid failed to calculate correct shape or centering.')
        npt.assert_array_equal(Grid(first=-1.5, center=0.5, step=1), np.arange(-1.5, 1.5 + 1, 1),
                               err_msg='Grid failed to calculate correct shape or centering.')

    def test_grid_step_1(self):
        npt.assert_array_equal(Grid(1), np.array([0]),
                               err_msg='Grid failed for single-element vector.')
        npt.assert_array_equal(Grid(4), np.array([-2, -1, 0, 1]),
                               err_msg='Grid failed for even length vector.')
        npt.assert_array_equal(Grid(5), np.array([-2, -1, 0, 1, 2]),
                               err_msg='Grid failed for odd length vector.')
        npt.assert_array_equal(Grid(4, center=int(4 / 2)), np.array([0, 1, 2, 3]),
                               err_msg='Grid failed for even length vector with offsets.')
        npt.assert_array_equal(Grid(5, center=int(5 / 2)), np.array([0, 1, 2, 3, 4]),
                               err_msg='Grid failed for odd length vector with offsets.')

    def test_equals(self):
        npt.assert_equal(Grid(10, 0.5) == Grid(10, 0.5), True)
        npt.assert_equal(Grid(10, 0.5) == Grid(10, -0.5), False)
        npt.assert_equal(Grid(10, 0.5) == Grid(10, 0.5, center=0), True)
        npt.assert_equal(Grid(10, 0.5) == Grid(10, 0.5, center=1), False)
        npt.assert_equal(Grid(10, 0.5) == Grid(10, 0.5, first=-2.5), True)
        npt.assert_equal(Grid(10, 0.5) == Grid(10, 0.5, first=0), False)
        npt.assert_equal(Grid(10, 0.5) == Grid(10, extent=5), True)
        npt.assert_equal(Grid(10, 0.5) == Grid(10, extent=6), False)
        npt.assert_equal(Grid(10, 2) == Grid(10.0, 2.0), False)
        npt.assert_equal(Grid([10, 10], 0.5) == Grid(10, [0.5, 0.5]), True)

    def test_extent(self):
        npt.assert_equal(Grid(256, 0.125) == Grid(step=0.125, extent=32), True)
        npt.assert_equal(Grid([256, 128], 0.125) == Grid(step=0.125, extent=[32, 16]), True)
        npt.assert_equal(Grid(extent=[32, 16]) == Grid(shape=[1, 1], extent=[32, 16]), True,
                         err_msg=f"Grid specified with just the extent failed.")

    def test_arithmetic_neg(self):
        npt.assert_equal(-Grid(10, 0.5), Grid(10, -0.5))
        npt.assert_equal(-Grid(10, 0.5, first=0), Grid(10, -0.5, first=0))
        npt.assert_equal(-Grid(10, 0.5), Grid(10, -0.5, first=2.5))

    def test_arithmetic_div(self):
        npt.assert_array_equal(Grid(1, 0.5), Grid(1) / 2)
        npt.assert_array_equal(Grid(4, 0.5), Grid(4) / 2)
        npt.assert_array_equal(Grid(5, 0.5), Grid(5) / 2)
        npt.assert_array_equal(Grid(4, 0.5, center=0.5 * int(4 / 2)),
                               (Grid.from_ranges(np.array([0, 1, 2, 3])) / 2)[0])
        npt.assert_array_equal(Grid(5, 0.5, center=0.5 * int(5 / 2)),
                               (Grid.from_ranges(np.array([0, 1, 2, 3, 4])) / 2)[0])

    def test_grid_scaled_down(self):
        npt.assert_array_equal(Grid(1, 0.5), np.array([0]),
                               err_msg='Grid failed for single-element vector.')
        npt.assert_array_equal(Grid(4, 0.5), np.array([-2, -1, 0, 1]) / 2,
                               err_msg='Grid failed for even length vector.')
        npt.assert_array_equal(Grid(5, 0.5), np.array([-2, -1, 0, 1, 2]) / 2,
                               err_msg='Grid failed for odd length vector.')
        npt.assert_array_equal(Grid(4, 0.5, center=0.5 * int(4 / 2)), np.array([0, 1, 2, 3]) / 2,
                               err_msg='Grid failed for even length vector with offsets.')
        npt.assert_array_equal(Grid(5, 0.5, center=0.5 * int(5 / 2)), np.array([0, 1, 2, 3, 4]) / 2,
                               err_msg='Grid failed for odd length vector with offsets.')

    def test_grid_scaled_up(self):
        npt.assert_array_equal(Grid(1, 2), np.array([0]),
                               err_msg='Grid failed for single-element vector.')
        npt.assert_array_equal(Grid(4, 2), np.array([-2, -1, 0, 1]) * 2,
                               err_msg='Grid failed for even length vector.')
        npt.assert_array_equal(Grid(5, 2), np.array([-2, -1, 0, 1, 2]) * 2,
                               err_msg='Grid failed for odd length vector.')
        npt.assert_array_equal(Grid(4, 2, center=2 * int(4 / 2)), np.array([0, 1, 2, 3]) * 2,
                               err_msg='Grid failed for even length vector with offsets.')
        npt.assert_array_equal(Grid(5, 2, center=2 * int(5 / 2)), np.array([0, 1, 2, 3, 4]) * 2,
                               err_msg='Grid failed for odd length vector with offsets.')

    def test_grid_first(self):
        npt.assert_array_equal(Grid(1, first=0), np.array([0]),
                               err_msg='Grid failed for single-element vector.')
        npt.assert_array_equal(Grid(4, first=-2), np.array([-2, -1, 0, 1]),
                               err_msg='Grid failed for even length vector.')
        npt.assert_array_equal(Grid(5, first=-2), np.array([-2, -1, 0, 1, 2]),
                               err_msg='Grid failed for odd length vector.')
        npt.assert_array_equal(Grid(4, first=0), np.array([0, 1, 2, 3]),
                               err_msg='Grid failed for even length vector with offsets.')
        npt.assert_array_equal(Grid(5, first=0), np.array([0, 1, 2, 3, 4]),
                               err_msg='Grid failed for odd length vector with offsets.')

    def test_grid_last(self):
        npt.assert_array_equal(Grid(4, last=2), np.array([-2, -1, 0, 1]),
                               err_msg='Grid failed for even length vector.')
        npt.assert_array_equal(Grid(5, last=3), np.array([-2, -1, 0, 1, 2]),
                               err_msg='Grid failed for odd length vector.')
        npt.assert_array_equal(Grid(4, last=4), np.array([0, 1, 2, 3]),
                               err_msg='Grid failed for even length vector with offsets.')
        npt.assert_array_equal(Grid(5, last=5), np.array([0, 1, 2, 3, 4]),
                               err_msg='Grid failed for odd length vector with offsets.')

    def test_grid_last_included(self):
        npt.assert_array_equal(Grid(1, last=0, include_last=True), np.array([0]),
                               err_msg='Grid failed for single-element vector.')
        npt.assert_array_equal(Grid(1, last=1, include_last=True), np.array([1]),
                               err_msg='Grid failed for single-element vector.')
        npt.assert_array_equal(Grid(1, last=5, include_last=True), np.array([5]),
                               err_msg='Grid failed for single-element vector.')
        npt.assert_array_equal(Grid(4, last=1, include_last=True), np.array([-2, -1, 0, 1]),
                               err_msg='Grid failed for even length vector.')
        npt.assert_array_equal(Grid(5, last=2, include_last=True), np.array([-2, -1, 0, 1, 2]),
                               err_msg='Grid failed for odd length vector.')
        npt.assert_array_equal(Grid(4, last=3, include_last=True), np.array([0, 1, 2, 3]),
                               err_msg='Grid failed for even length vector with offsets.')
        npt.assert_array_equal(Grid(5, last=4, include_last=True), np.array([0, 1, 2, 3, 4]),
                               err_msg='Grid failed for odd length vector with offsets.')

    def test_grid_last_not_included(self):
        npt.assert_array_equal(Grid(1, last=1), np.array([0]),
                               err_msg='Grid failed for single-element vector.')
        npt.assert_array_equal(Grid(4, last=2), np.array([-2, -1, 0, 1]),
                               err_msg='Grid failed for even length vector.')
        npt.assert_array_equal(Grid(5, last=3), np.array([-2, -1, 0, 1, 2]),
                               err_msg='Grid failed for odd length vector.')
        npt.assert_array_equal(Grid(4, last=4), np.array([0, 1, 2, 3]),
                               err_msg='Grid failed for even length vector with offsets.')
        npt.assert_array_equal(Grid(5, last=5), np.array([0, 1, 2, 3, 4]),
                               err_msg='Grid failed for odd length vector with offsets.')

    def test_grid_first_last(self):
        npt.assert_array_equal(Grid(1, first=0, last=1), np.array([0]),
                               err_msg='Grid failed without specification of first and last only.')
        # npt.assert_array_equal(Grid(1, first=0, last=5), np.array([0]),
        #                        err_msg='Grid failed without specification of first and last only.')
        npt.assert_array_equal(Grid(1, first=5, last=6), np.array([5]),
                               err_msg='Grid failed without specification of first and last only.')
        npt.assert_array_equal(Grid(1, first=5, last=10), np.array([5]),
                               err_msg='Grid failed without specification of first and last only.')
        npt.assert_array_equal(Grid(5, first=5, last=10), np.array([5, 6, 7, 8, 9]),
                               err_msg='Grid failed without specification of first and last only.')

    def test_grid_first_last_included(self):
        npt.assert_array_equal(Grid(2, first=0, last=1, include_last=True), np.array([0, 1]),
                               err_msg='Grid failed without specification of first and last only.')
        npt.assert_array_equal(Grid(2, first=0, last=10, include_last=True), np.array([0, 10]),
                               err_msg='Grid failed without specification of first and last only.')
        npt.assert_array_equal(Grid(2, first=5, last=10, include_last=True), np.array([5, 10]),
                               err_msg='Grid failed without specification of first and last only.')
        npt.assert_array_equal(Grid(6, first=5, last=10, include_last=True), np.array([5, 6, 7, 8, 9, 10]),
                               err_msg='Grid failed without specification of first and last only.')

    def test_grid_first_last_conflict(self):
        with npt.assert_raises(ValueError):
            Grid(1, first=0, last=1, include_last=True)
        with npt.assert_raises(ValueError):
            Grid(1, first=0, last=5, include_last=True)
        with npt.assert_raises(ValueError):
            Grid(1, first=5, last=6, include_last=True)
        with npt.assert_raises(ValueError):
            Grid(1, first=5, last=10, include_last=True)

    def test_center_at_index(self):
        npt.assert_array_equal(Grid(5, center=0, center_at_index=True), [-2, -1, 0, 1, 2])
        npt.assert_array_equal(Grid(5, center=0, center_at_index=False), [-2, -1, 0, 1, 2])
        npt.assert_array_equal(Grid(4, center=0, center_at_index=True), [-2, -1, 0, 1])
        npt.assert_array_equal(Grid(4, center=0, center_at_index=False), [-1.5, -0.5, 0.5, 1.5])
        npt.assert_array_equal(Grid(5, 2, center=0, center_at_index=True), [-4, -2, 0, 2, 4])
        npt.assert_array_equal(Grid(5, 2, center=0, center_at_index=False), [-4, -2, 0, 2, 4])
        npt.assert_array_equal(Grid(4, 2, center=0, center_at_index=True), [-4, -2, 0, 2])
        npt.assert_array_equal(Grid(4, 2, center=0, center_at_index=False), [-3, -1, 1, 3])
        npt.assert_array_equal(Grid(5, 0.5, center=0, center_at_index=True), [-1.0, -0.5, 0.0, 0.5, 1.0])
        npt.assert_array_equal(Grid(5, 0.5, center=0, center_at_index=False), [-1.0, -0.5, 0.0, 0.5, 1.0])
        npt.assert_array_equal(Grid(4, 0.5, center=0, center_at_index=True), [-1.0, -0.5, 0.0, 0.5])
        npt.assert_array_equal(Grid(4, 0.5, center=0, center_at_index=False), [-0.75, -0.25, 0.25, 0.75])
        npt.assert_array_equal(Grid(5, center_at_index=True), [-2, -1, 0, 1, 2])
        npt.assert_array_equal(Grid(5, center_at_index=False), [-2, -1, 0, 1, 2])
        npt.assert_array_equal(Grid(4, center_at_index=True), [-2, -1, 0, 1])
        npt.assert_array_equal(Grid(4, center_at_index=False), [-1.5, -0.5, 0.5, 1.5])
        npt.assert_array_equal(Grid(5, 2, center_at_index=True), [-4, -2, 0, 2, 4])
        npt.assert_array_equal(Grid(5, 2, center_at_index=False), [-4, -2, 0, 2, 4])
        npt.assert_array_equal(Grid(4, 2, center_at_index=True), [-4, -2, 0, 2])
        npt.assert_array_equal(Grid(4, 2, center_at_index=False), [-3, -1, 1, 3])
        npt.assert_array_equal(Grid(5, 0.5, center_at_index=True), [-1.0, -0.5, 0.0, 0.5, 1.0])
        npt.assert_array_equal(Grid(5, 0.5, center_at_index=False), [-1.0, -0.5, 0.0, 0.5, 1.0])
        npt.assert_array_equal(Grid(4, 0.5, center_at_index=True), [-1.0, -0.5, 0.0, 0.5])
        npt.assert_array_equal(Grid(4, 0.5, center_at_index=False), [-0.75, -0.25, 0.25, 0.75])
        npt.assert_array_equal(Grid(5, first=0, center_at_index=True), [0, 1, 2, 3, 4])
        npt.assert_array_equal(Grid(5, first=0, center_at_index=False), [0, 1, 2, 3, 4])
        npt.assert_array_equal(Grid(4, first=0, center_at_index=True), [0, 1, 2, 3])
        npt.assert_array_equal(Grid(4, first=0, center_at_index=False), [0, 1, 2, 3])
        npt.assert_equal(Grid(4, first=0, center_at_index=False).first, 0)
        npt.assert_equal(Grid(4, first=0, center_at_index=True).first, 0)
        npt.assert_equal(Grid(5, first=0, center_at_index=False).first, 0)
        npt.assert_equal(Grid(5, first=0, center_at_index=True).first, 0)
        npt.assert_equal(Grid(4, first=-2, center_at_index=False).first, -2)
        npt.assert_equal(Grid(4, first=-2, center_at_index=True).first, -2)
        npt.assert_equal(Grid(5, first=-2, center_at_index=False).first, -2)
        npt.assert_equal(Grid(5, first=-2, center_at_index=True).first, -2)

    def test_grid_dtype(self):
        g = Grid(shape=(2, 3), first=(4, 5))
        npt.assert_array_equal(g[0], np.array([[4], [5]]),
                               err_msg='Grid range 0 incorrect.')
        npt.assert_equal(g[0].dtype == int, True,
                         err_msg=f'{g} did not maintain integerness of arguments.')
        npt.assert_array_equal(g[1], np.array([[5, 6, 7]]),
                               err_msg='Grid range 1 incorrect.')
        npt.assert_equal(g[1].dtype == int, True,
                         err_msg=f'{g} did not maintain integerness of arguments.')

    def test_grid_frequency_single(self):
        npt.assert_array_equal(Grid(5).f, np.array([0, 1/5, 2/5, -2/5, -1/5]))
        npt.assert_array_equal(Grid(5).f.as_origin_at_0, np.array([0, 1 / 5, 2 / 5, -2 / 5, -1 / 5]))
        npt.assert_array_equal(Grid(5).f.as_origin_at_center, np.array([-2 / 5, -1 / 5, 0, 1 / 5, 2 / 5]))

        npt.assert_array_equal(Grid(5).k, np.array([0, 1/5, 2/5, -2/5, -1/5]) * 2*np.pi)
        npt.assert_array_equal(Grid(5).k.as_origin_at_0, np.array([0, 1 / 5, 2 / 5, -2 / 5, -1 / 5]) * 2*np.pi)
        npt.assert_array_equal(Grid(5).k.as_origin_at_center, np.array([-2 / 5, -1 / 5, 0, 1 / 5, 2 / 5]) * 2*np.pi)

        npt.assert_array_equal(Grid(5, 2).f, np.array([0, 1/10, 2/10, -2/10, -1/10]))
        npt.assert_array_equal(Grid(5, 2).f.as_origin_at_0, np.array([0, 1/10, 2/10, -2/10, -1/10]))
        npt.assert_array_equal(Grid(5, 2).f.as_origin_at_center, np.array([-2/10, -1/10, 0, 1/10, 2/10]))

    def test_grid_frequency_multi(self):
        npt.assert_array_equal(Grid([5, 2]).f[0], np.array([[0], [1/5], [2/5], [-2/5], [-1/5]]))

    def test_singleton(self):
        grid = Grid(shape=[3, 1], step=[1, 0], first=(1, 4))
        npt.assert_equal(grid[0], np.array([[1], [2], [3]]))
        npt.assert_equal(grid[1], np.array([[4]]))

        grid = Grid(shape=[3, 1], step=[1/2, 0], first=(1, 4))
        npt.assert_equal(grid[0], np.array([[1], [1.5], [2]]))
        npt.assert_equal(grid[1], np.array([[4]]))

    def test_from_ranges(self):
        grid = Grid.from_ranges([1, 2, 3], 4)
        npt.assert_equal(grid[0], np.array([[1], [2], [3]]))
        npt.assert_equal(grid[1], np.array([[4]]))

        grid = Grid.from_ranges([1, 2, 3], (4, 5))
        npt.assert_equal(grid[0], np.array([[1], [2], [3]]))
        npt.assert_equal(grid[1], np.array([[4, 5]]))

        grid = Grid.from_ranges(np.array([1, 2, 3]), np.array([4, 5]))
        npt.assert_equal(grid[0], np.array([[1], [2], [3]]))
        npt.assert_equal(grid[1], np.array([[4, 5]]))

        grid = Grid.from_ranges([[1], [2], [3]], (4, 5))
        npt.assert_equal(grid[0], np.array([[1], [2], [3]]))
        npt.assert_equal(grid[1], np.array([[4, 5]]))

        grid = Grid.from_ranges([[1], [2], [3]], [[4, 5]])
        npt.assert_equal(grid[0], np.array([[1], [2], [3]]))
        npt.assert_equal(grid[1], np.array([[4, 5]]))

        grid = Grid.from_ranges([[1, 1], [2, 2], [3, 3]], [[4, 5], [4, 5], [4, 5]])
        npt.assert_equal(grid[0], np.array([[1], [2], [3]]))
        npt.assert_equal(grid[1], np.array([[4, 5]]))

        grid = Grid.from_ranges(*([[1, 1], [2, 2], [3, 3]], [[4, 5], [4, 5], [4, 5]]))
        npt.assert_equal(grid[0], np.array([[1], [2], [3]]), 'Grid initialization with tuple failed.')
        npt.assert_equal(grid[1], np.array([[4, 5]]), 'Grid initialization with tuple failed.')

        grid = Grid.from_ranges([-2/3, -1/3, 0, 1/3])
        npt.assert_equal(grid.origin_at_center, True)
        npt.assert_equal(grid.step, 1/3)
        npt.assert_equal(grid.first, -2/3)
        npt.assert_equal(grid.center, 0)
        npt.assert_equal(grid.extent, 4/3)
        npt.assert_equal(grid.shape, 4)
        npt.assert_equal(grid[0], [-2/3, -1/3, 0, 1/3])

        grid = Grid.from_ranges([0, 1/3, -2/3, -1/3])
        npt.assert_equal(grid.origin_at_center, False)
        npt.assert_equal(grid.step, 1/3)
        npt.assert_equal(grid.first, -2/3)
        npt.assert_equal(grid.center, 0)
        npt.assert_equal(grid.extent, 4/3)
        npt.assert_equal(grid.shape, 4)
        npt.assert_equal(grid[0], [0, 1/3, -2/3, -1/3])

        grid = Grid.from_ranges([-1/3, 0, 1/3])
        npt.assert_equal(grid.origin_at_center, True, 'Grid initialization with centered odd range failed.')
        npt.assert_equal(grid.step, 1/3, 'Grid initialization with centered odd range failed.')
        npt.assert_equal(grid.first, -1/3, 'Grid initialization with centered odd range failed.')
        npt.assert_equal(grid.center, 0, 'Grid initialization with centered odd range failed.')
        npt.assert_equal(grid.extent, 1, 'Grid initialization with centered odd range failed.')
        npt.assert_equal(grid.shape, 3, 'Grid initialization with centered odd range failed.')
        npt.assert_equal(grid[0], [-1/3, 0, 1/3], 'Grid initialization with centered odd range failed.')

        grid = Grid.from_ranges([0, 1/3, -1/3])
        npt.assert_equal(grid.origin_at_center, False, 'Grid initialization with non-centered odd range failed.')
        npt.assert_equal(grid.step, 1/3, 'Grid initialization with non-centered odd range failed.')
        npt.assert_equal(grid.first, -1/3, 'Grid initialization with non-centered odd range failed.')
        npt.assert_equal(grid.center, 0, 'Grid initialization with non-centered odd range failed.')
        npt.assert_equal(grid.extent, 1, 'Grid initialization with non-centered odd range failed.')
        npt.assert_equal(grid.shape, 3, 'Grid initialization with non-centered odd range failed.')
        npt.assert_equal(grid[0], [0, 1/3, -1/3], 'Grid initialization with non-centered odd range failed.')

    def test_set(self):
        """These should all fail for regular (non-mutable) grids"""
        grid = Grid(4, 1)

        def set_shape():
            grid.shape = 10

        def set_step():
            grid.step = 2

        def set_center():
            grid.center = 5

        def set_first():
            grid.first = 0

        npt.assert_raises(AttributeError, set_shape)
        npt.assert_raises(AttributeError, set_step)
        npt.assert_raises(AttributeError, set_center)
        npt.assert_raises(AttributeError, set_first)


class TestMutableGrid(unittest.TestCase):
    def test_set(self):
        grid = MutableGrid(4, 1)
        grid.step = 2
        npt.assert_equal(grid, Grid(4, 2).mutable, 'step size not updated correctly')
        grid.step = 3.14
        npt.assert_equal(grid, Grid(4, 3.14).mutable, 'step size not updated correctly')
        grid.shape = 10
        grid.step = 2
        grid.dtype = int
        npt.assert_equal(grid, Grid(10, 2).mutable, 'shape not updated correctly')
        grid.first = 0
        npt.assert_equal(grid, Grid(10, 2, first=0).mutable, 'Offset not updated correctly.')
        grid.center = 0
        npt.assert_equal(grid, Grid(10, 2, center=0).mutable, 'Offset not updated correctly.')


if __name__ == '__main__':
    unittest.main()
