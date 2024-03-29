import unittest
import numpy as np
import numpy.testing as npt

from macromax.utils.array import word_align, vector_to_axis, add_dims_on_right
from macromax.utils.display import complex2rgb, grid2extent, hsv2rgb


class TestSolution(unittest.TestCase):
    def test_vector_to_axis(self):
        a = vector_to_axis(np.array([3, 1, 4]), axis=0, ndim=1)
        npt.assert_almost_equal(a, np.array([3, 1, 4]))

        a = vector_to_axis(np.array([3, 1, 4]), ndim=2)
        npt.assert_almost_equal(a, np.array([[3], [1], [4]]))

        a = vector_to_axis(np.array([3, 1, 4]), axis=1, ndim=2)
        npt.assert_almost_equal(a, np.array([[3, 1, 4]]))

        a = vector_to_axis(np.array(3), axis=0, ndim=1)
        npt.assert_almost_equal(a, np.array([3]))

    def test_add_dims_on_right(self):
        npt.assert_array_equal(add_dims_on_right([1, 2, 3], 1), np.array([1, 2, 3])[..., np.newaxis], err_msg='Could not add 1 axis to a vector.')
        npt.assert_array_equal(add_dims_on_right([1, 2, 3], 2), np.array([1, 2, 3])[..., np.newaxis, np.newaxis], err_msg='Could not add 3 axes to a vector.')
        npt.assert_array_equal(add_dims_on_right([1, 2, 3], 3), np.array([1, 2, 3])[..., np.newaxis, np.newaxis, np.newaxis], err_msg='Could not add 3 axes to a vector.')
        npt.assert_array_equal(add_dims_on_right([1, 2, 3], 0), np.array([1, 2, 3]), err_msg='Could not add 0 axes')
        npt.assert_array_equal(add_dims_on_right([1, 2, 3]), np.array([1, 2, 3]), err_msg='Number of axes to add does not default to 0.')
        npt.assert_array_equal(add_dims_on_right([[1, 2, 3], [4, 5, 6]], 1), np.array([[1, 2, 3], [4, 5, 6]])[..., np.newaxis], err_msg='Could not add 1 axis to a matrix.')
        npt.assert_array_equal(add_dims_on_right([[1, 2, 3], [4, 5, 6]], 2), np.array([[1, 2, 3], [4, 5, 6]])[..., np.newaxis, np.newaxis], err_msg='Could not add 2 axes to a matrix.')
        npt.assert_array_equal(add_dims_on_right([[1, 2, 3], [4, 5, 6]], 3), np.array([[1, 2, 3], [4, 5, 6]])[..., np.newaxis, np.newaxis, np.newaxis], err_msg='Could not add 3 axes to a matrix.')
        npt.assert_array_equal(add_dims_on_right([[1, 2, 3], [4, 5, 6]], 0), np.array([[1, 2, 3], [4, 5, 6]]), err_msg='Could not add 0 axes')
        npt.assert_array_equal(add_dims_on_right([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), err_msg='Number of axes to add does not default to 0.')
        npt.assert_array_equal(add_dims_on_right([[1], [4]], -1), np.array([1, 4]), err_msg='Could not remove axes on the right')
        npt.assert_array_equal(add_dims_on_right([1], 1), np.array([1])[..., np.newaxis], err_msg='Could not add axis to [1].')
        npt.assert_array_equal(add_dims_on_right([[1]], 1), np.array([[1]])[..., np.newaxis], err_msg='Could not add axis to [[1]].')
        npt.assert_array_equal(add_dims_on_right(1, 1), np.array(1)[..., np.newaxis], err_msg='Could not add axis to a scalar.')
        npt.assert_array_equal(add_dims_on_right([], 1), np.array([])[..., np.newaxis], err_msg='Could not add axis to an empty array of shape (0,) .')

    def test_complex2rgb(self):
        c = np.array([[1, np.exp(2j*np.pi/3)], [np.exp(-2j*np.pi/3), -1]], dtype=np.complex128)
        res = np.array([[[0, 1, 1], [1, 0, 1]], [[1, 1, 0], [1, 0, 0]]], dtype=np.float64)

        rgb = complex2rgb(c)
        npt.assert_almost_equal(rgb, res)

        # Check saturation
        rgb = complex2rgb(10.0 * c)
        npt.assert_almost_equal(rgb, np.ones_like(res), err_msg='Saturated values are not as expected.')

        # Check intensity scaling
        rgb = complex2rgb(0.5 * c)
        npt.assert_almost_equal(rgb, 0.5 * res, err_msg='The intensity does not scale linearly with the amplitude.')

        c[1, 1] /= 2.0
        res[1, 1, :] /= 2.0
        rgb = complex2rgb(0.5 * c)
        npt.assert_almost_equal(rgb, 0.5 * res, err_msg='Non-uniform amplitudes are not represented correctly.')

    def test_hsv2rgb(self):
        hsv = np.array([[[0, 1, 1], [1.0/3.0, 1, 1]], [[2.0/3.0, 1, 1], [1, 1, 0.5]]])
        res = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0.5, 0, 0]]])

        rgb = hsv2rgb(hsv)
        npt.assert_almost_equal(rgb, res, err_msg='HSV to RGB hue test failed')

    def test_ranges2extent(self):
        npt.assert_almost_equal(grid2extent(np.arange(5), np.arange(10)), [-0.5, 9.5, 4.5, -0.5])
        npt.assert_almost_equal(grid2extent(np.arange(-2, 5 - 2), np.arange(-3, 10 - 3)), [-3.5, 6.5, 2.5, -2.5])
        npt.assert_almost_equal(grid2extent(np.arange(5), np.arange(0, 2, 0.2)), [-0.1, 1.9, 4.5, -0.5])
        npt.assert_almost_equal(grid2extent(np.arange(5), np.arange(-1, 2, 0.2)), [-1.1, 1.9, 4.5, -0.5])
        npt.assert_almost_equal(grid2extent(range(-2, 5 - 2), np.arange(-3, 10 - 3)), [-3.5, 6.5, 2.5, -2.5])
        npt.assert_almost_equal(grid2extent(range(-2, 5 - 2), range(-3, 10 - 3)), [-3.5, 6.5, 2.5, -2.5])
        npt.assert_almost_equal(grid2extent(np.arange(-2, 5 - 2), range(-3, 10 - 3)), [-3.5, 6.5, 2.5, -2.5])

    def test_word_align(self):
        # For now only checking that the arrays didn't change
        def check_array(a):
            npt.assert_almost_equal(word_align(a), a, err_msg='default word_length=32 failed')
            npt.assert_almost_equal(word_align(a, word_length=32), a, err_msg='setting word_length=32 failed')
            npt.assert_almost_equal(word_align(a, word_length=16), a, err_msg='setting word_length=16 failed')

        check_array(np.arange(5))
        check_array(np.arange(8))
        check_array(np.ones([8, 24, 3]))
        check_array(np.arange(512))
        check_array(np.array([5]))
        check_array(np.array(5))


if __name__ == '__main__':
    unittest.main()
