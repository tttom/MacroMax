import unittest
import numpy.testing as npt

from macromax.utils.array import Grid, MutableGrid

import numpy as np


class TestGrid(unittest.TestCase):
    def test_grid_step_1(self):
        npt.assert_array_equal(Grid(1), np.array([0]),
                               err_msg="Grid failed for single-element vector.")
        npt.assert_array_equal(Grid(4), np.array([-2, -1, 0, 1]),
                               err_msg="Grid failed for even length vector.")
        npt.assert_array_equal(Grid(5), np.array([-2, -1, 0, 1, 2]),
                               err_msg="Grid failed for odd length vector.")
        npt.assert_array_equal(Grid(4, center=int(4 / 2)), np.array([0, 1, 2, 3]),
                               err_msg="Grid failed for even length vector with offsets.")
        npt.assert_array_equal(Grid(5, center=int(5 / 2)), np.array([0, 1, 2, 3, 4]),
                               err_msg="Grid failed for odd length vector with offsets.")

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

    def test_arithmetic_neg(self):
        npt.assert_equal(-Grid(10, 0.5) == Grid(10, -0.5), True)
        npt.assert_equal(-Grid(10, 0.5, first=0) == Grid(10, -0.5, first=0), True)
        npt.assert_equal(-Grid(10, 0.5) == Grid(10, -0.5, first=2.5), True)

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
                               err_msg="Grid failed for single-element vector.")
        npt.assert_array_equal(Grid(4, 0.5), np.array([-2, -1, 0, 1]) / 2,
                               err_msg="Grid failed for even length vector.")
        npt.assert_array_equal(Grid(5, 0.5), np.array([-2, -1, 0, 1, 2]) / 2,
                               err_msg="Grid failed for odd length vector.")
        npt.assert_array_equal(Grid(4, 0.5, center=0.5 * int(4 / 2)), np.array([0, 1, 2, 3]) / 2,
                               err_msg="Grid failed for even length vector with offsets.")
        npt.assert_array_equal(Grid(5, 0.5, center=0.5 * int(5 / 2)), np.array([0, 1, 2, 3, 4]) / 2,
                               err_msg="Grid failed for odd length vector with offsets.")

    def test_grid_scaled_up(self):
        npt.assert_array_equal(Grid(1, 2), np.array([0]),
                               err_msg="Grid failed for single-element vector.")
        npt.assert_array_equal(Grid(4, 2), np.array([-2, -1, 0, 1]) * 2,
                               err_msg="Grid failed for even length vector.")
        npt.assert_array_equal(Grid(5, 2), np.array([-2, -1, 0, 1, 2]) * 2,
                               err_msg="Grid failed for odd length vector.")
        npt.assert_array_equal(Grid(4, 2, center=2 * int(4 / 2)), np.array([0, 1, 2, 3]) * 2,
                               err_msg="Grid failed for even length vector with offsets.")
        npt.assert_array_equal(Grid(5, 2, center=2 * int(5 / 2)), np.array([0, 1, 2, 3, 4]) * 2,
                               err_msg="Grid failed for odd length vector with offsets.")

    def test_grid_first(self):
        npt.assert_array_equal(Grid(1, first=0), np.array([0]),
                               err_msg="Grid failed for single-element vector.")
        npt.assert_array_equal(Grid(4, first=-2), np.array([-2, -1, 0, 1]),
                               err_msg="Grid failed for even length vector.")
        npt.assert_array_equal(Grid(5, first=-2), np.array([-2, -1, 0, 1, 2]),
                               err_msg="Grid failed for odd length vector.")
        npt.assert_array_equal(Grid(4, first=0), np.array([0, 1, 2, 3]),
                               err_msg="Grid failed for even length vector with offsets.")
        npt.assert_array_equal(Grid(5, first=0), np.array([0, 1, 2, 3, 4]),
                               err_msg="Grid failed for odd length vector with offsets.")

    def test_grid_last(self):
        npt.assert_array_equal(Grid(1, last=0, include_last=True), np.array([0]),
                               err_msg="Grid failed for single-element vector.")
        npt.assert_array_equal(Grid(4, last=1, include_last=True), np.array([-2, -1, 0, 1]),
                               err_msg="Grid failed for even length vector.")
        npt.assert_array_equal(Grid(5, last=2, include_last=True), np.array([-2, -1, 0, 1, 2]),
                               err_msg="Grid failed for odd length vector.")
        npt.assert_array_equal(Grid(4, last=3, include_last=True), np.array([0, 1, 2, 3]),
                               err_msg="Grid failed for even length vector with offsets.")
        npt.assert_array_equal(Grid(5, last=4, include_last=True), np.array([0, 1, 2, 3, 4]),
                               err_msg="Grid failed for odd length vector with offsets.")

    def test_grid_last_not_included(self):
        npt.assert_array_equal(Grid(1, last=1), np.array([0]),
                               err_msg="Grid failed for single-element vector.")
        npt.assert_array_equal(Grid(4, last=2), np.array([-2, -1, 0, 1]),
                               err_msg="Grid failed for even length vector.")
        npt.assert_array_equal(Grid(5, last=3), np.array([-2, -1, 0, 1, 2]),
                               err_msg="Grid failed for odd length vector.")
        npt.assert_array_equal(Grid(4, last=4), np.array([0, 1, 2, 3]),
                               err_msg="Grid failed for even length vector with offsets.")
        npt.assert_array_equal(Grid(5, last=5), np.array([0, 1, 2, 3, 4]),
                               err_msg="Grid failed for odd length vector with offsets.")

    def test_grid_dtype(self):
        g = Grid(shape=(2, 3), first=(4, 5))
        npt.assert_array_equal(g[0], np.array([[4], [5]]),
                               err_msg="Grid range 0 incorrect.")
        npt.assert_equal(g[0].dtype == np.int, True,
                         err_msg="Grid didn't maintain integerness of arguments.")
        npt.assert_array_equal(g[1], np.array([[5, 6, 7]]),
                               err_msg="Grid range 1 incorrect.")
        npt.assert_equal(g[1].dtype == np.int, True,
                         err_msg="Grid didn't maintain integerness of arguments.")

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

        grid = Grid.from_ranges(([[1, 1], [2, 2], [3, 3]], [[4, 5], [4, 5], [4, 5]]))
        npt.assert_equal(grid[0], np.array([[1], [2], [3]]), "Grid initialization with tuple failed.")
        npt.assert_equal(grid[1], np.array([[4, 5]]), "Grid initialization with tuple failed.")

        grid = Grid.from_ranges([-2/3, -1/3, 0, 1/3])
        npt.assert_equal(grid.origin_at_center, True)
        npt.assert_equal(grid.step, 1/3)
        npt.assert_equal(grid.first, -2/3)
        npt.assert_equal(grid.shape, 4)
        npt.assert_equal(grid[0], [-2/3, -1/3, 0, 1/3])

        grid = Grid.from_ranges([0, 1/3, -2/3, -1/3])
        npt.assert_equal(grid.origin_at_center, False)
        npt.assert_equal(grid.step, 1/3)
        npt.assert_equal(grid.first, -2/3)
        npt.assert_equal(grid.shape, 4)
        npt.assert_equal(grid[0], [0, 1/3, -2/3, -1/3])

        grid = Grid.from_ranges([-1/3, 0, 1/3])
        npt.assert_equal(grid.origin_at_center, True, "Grid initialization with centered odd range failed.")
        npt.assert_equal(grid.step, 1/3, "Grid initialization with centered odd range failed.")
        npt.assert_equal(grid.first, -1/3, "Grid initialization with centered odd range failed.")
        npt.assert_equal(grid.shape, 3, "Grid initialization with centered odd range failed.")
        npt.assert_equal(grid[0], [-1/3, 0, 1/3], "Grid initialization with centered odd range failed.")

        grid = Grid.from_ranges([0, 1/3, -1/3])
        npt.assert_equal(grid.origin_at_center, False, "Grid initialization with non-centered odd range failed.")
        npt.assert_equal(grid.step, 1/3, "Grid initialization with non-centered odd range failed.")
        npt.assert_equal(grid.first, -1/3, "Grid initialization with non-centered odd range failed.")
        npt.assert_equal(grid.shape, 3, "Grid initialization with non-centered odd range failed.")
        npt.assert_equal(grid[0], [0, 1/3, -1/3], "Grid initialization with non-centered odd range failed.")

    def test_set(self):
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
        npt.assert_equal(grid == Grid(4, 2), True, 'step size not updated correctly')
        grid.step = 3.14
        npt.assert_equal(grid == Grid(4, 3.14), True, 'step size not updated correctly')
        grid.shape = 10
        grid.step = 2
        npt.assert_equal(grid == Grid(10, 2), True, 'shape not updated correctly')
        grid.first = 0
        npt.assert_equal(grid == Grid(10, 2, first=0), True, 'Offset not updated correctly.')
        grid.center = 0
        npt.assert_equal(grid == Grid(10, 2, center=0), True, 'Offset not updated correctly.')


if __name__ == '__main__':
    unittest.main()
