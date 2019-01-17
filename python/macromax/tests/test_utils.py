import unittest
import numpy as np
import numpy.testing as npt
import numpy.fft as ft

import macromax.utils as utils


class TestSolution(unittest.TestCase):
    def test_calc_frequency_ranges(self):
        xf_range, = utils.calc_frequency_ranges(np.arange(10), centered=True)
        yf_range, = utils.calc_frequency_ranges(np.arange(-3, 10 - 3), centered=True)
        zf_range, = utils.calc_frequency_ranges(np.arange(5), centered=True)
        npt.assert_almost_equal(np.arange(-0.5, 0.5, 0.1), xf_range)
        npt.assert_almost_equal(np.arange(-0.5, 0.5, 0.1), yf_range)
        npt.assert_almost_equal(np.array([-0.4, -0.2, 0.0, 0.2, 0.4]), zf_range)

        xf_range, = utils.calc_frequency_ranges(np.arange(10))
        yf_range, = utils.calc_frequency_ranges(np.arange(-3, 10 - 3))
        npt.assert_almost_equal(ft.ifftshift(np.arange(-0.5, 0.5, 0.1)), xf_range)
        npt.assert_almost_equal(np.array([0.0, 0.1, 0.2, 0.3, 0.4, -0.5, -0.4, -0.3, -0.2, -0.1]), yf_range)

        xf_range, = utils.calc_frequency_ranges(np.arange(0, 1, 0.1), centered=True)
        yf_range, = utils.calc_frequency_ranges(np.arange(-0.3, 1 - 0.3, 0.1), centered=True)
        npt.assert_almost_equal(np.arange(-5.0, 5.0), xf_range)
        npt.assert_almost_equal(np.arange(-5.0, 5.0), yf_range)

        xf_range, = utils.calc_frequency_ranges(np.arange(0, 1, 0.1))
        yf_range, = utils.calc_frequency_ranges(np.arange(-0.3, 1 - 0.3, 0.1))
        npt.assert_almost_equal(ft.ifftshift(np.arange(-5.0, 5.0)), xf_range)
        npt.assert_almost_equal(np.array([0.0, 1, 2, 3, 4, -5, -4, -3, -2, -1]), yf_range)

        xf_range, = utils.calc_frequency_ranges(range(-3, 10 - 3), centered=True)
        npt.assert_almost_equal(np.arange(-0.5, 0.5, 0.1), xf_range)

        xf_range, = utils.calc_frequency_ranges([1, 2, 3])
        npt.assert_almost_equal([0, 1/3, -1/3], xf_range)

        xf_range, = utils.calc_frequency_ranges(range(0))
        npt.assert_almost_equal([], xf_range)

        xf_range, = utils.calc_frequency_ranges(range(1), centered=True)
        yf_range, = utils.calc_frequency_ranges(range(4, 5), centered=True)
        npt.assert_almost_equal(np.arange(1.0), xf_range)
        npt.assert_almost_equal(np.arange(1.0), yf_range)

        xf_range, yf_range = utils.calc_frequency_ranges(range(5, 10+5, 1), np.arange(-0.5, 0.5, 0.1), centered=True)
        npt.assert_almost_equal(np.arange(-0.5, 0.5, 0.1), xf_range)
        npt.assert_almost_equal(np.arange(-5.0, 5.0), yf_range)

    def test_to_dim(self):
        a = utils.to_dim(np.array([3, 1, 4]), 1, axis=0)
        npt.assert_almost_equal(a, np.array([3, 1, 4]))

        a = utils.to_dim(np.array([3, 1, 4]), 2)
        npt.assert_almost_equal(a, np.array([[3], [1], [4]]))

        a = utils.to_dim(np.array([3, 1, 4]), 2, axis=1)
        npt.assert_almost_equal(a, np.array([[3, 1, 4]]))

        a = utils.to_dim(np.array(3), 0, axis=0)
        npt.assert_almost_equal(a, np.array(3))

        a = utils.to_dim(np.array(3), 1, axis=0)
        npt.assert_almost_equal(a, np.array([3]))

    def test_add_dims(self):
        a = utils.add_dims(np.array([3, 1, 4]), 1, 2)
        npt.assert_almost_equal(a, np.array([[3, 1, 4]]))

        a = utils.add_dims(np.array([3, 1, 4]), 0, 1)
        npt.assert_almost_equal(a, np.array([3, 1, 4]))

        a = utils.add_dims(np.array([3, 1, 4]), 0, 2)
        npt.assert_almost_equal(a, np.array([[3], [1], [4]]))

    def test_pad_to_length(self):
        a = utils.pad_to_length(np.array([3, 1, 4]), 5)
        npt.assert_almost_equal(a, np.array([3, 1, 4, 0, 0]))

        a = utils.pad_to_length(np.array([3, 1, 4]), 6, padding_value=5)
        npt.assert_almost_equal(a, np.array([3, 1, 4, 5, 5, 5]))

        a = utils.pad_to_length(np.array([3, 1, 4]), 3)
        npt.assert_almost_equal(a, np.array([3, 1, 4]))

        a = utils.pad_to_length(np.array([np.pi]), 3)
        npt.assert_almost_equal(a, np.array([np.pi, 0, 0]))

        a = utils.pad_to_length([np.pi], 3)
        npt.assert_almost_equal(a, np.array([np.pi, 0, 0]))

        a = utils.pad_to_length(np.pi, 3)
        npt.assert_almost_equal(a, np.array([np.pi, 0, 0]))

        a = utils.pad_to_length(range(3), 5)
        npt.assert_almost_equal(a, np.array([0, 1, 2, 0, 0]))

    def test_extend_to_length(self):
        a = utils.extend_to_length(np.array([3, 1, 4]), 5)
        npt.assert_almost_equal(a, np.array([3, 1, 4, 4, 4]), err_msg='extending by two failed')

        a = utils.extend_to_length([3, 1, 4], 5)
        npt.assert_almost_equal(a, np.array([3, 1, 4, 4, 4]), err_msg='extending a list by two failed')

        a = utils.extend_to_length(np.array([3, 1, 4]), 3)
        npt.assert_almost_equal(a, np.array([3, 1, 4]), err_msg='extending by zero failed')

        a = utils.extend_to_length(np.array([np.pi]), 3)
        npt.assert_almost_equal(a, np.array([np.pi, np.pi, np.pi]), err_msg='extending singleton failed')

        a = utils.extend_to_length([np.pi], 3)
        npt.assert_almost_equal(a, np.array([np.pi, np.pi, np.pi]), err_msg='extending singleton list failed')

        a = utils.extend_to_length(np.pi, 3)
        npt.assert_almost_equal(a, np.array([np.pi, np.pi, np.pi]), err_msg='extending scalar to vector failed')

        a = utils.extend_to_length(range(3), 5)
        npt.assert_almost_equal(a, np.array([0, 1, 2, 2, 2]), err_msg='extending Python range failed')

    def test_complex2rgb(self):
        c = np.array([[1, np.exp(2j*np.pi/3)], [np.exp(-2j*np.pi/3), -1]], dtype=np.complex128)
        res = np.array([[[0, 1, 1], [1, 0, 1]], [[1, 1, 0], [1, 0, 0]]], dtype=np.float64)

        rgb = utils.complex2rgb(c)
        npt.assert_almost_equal(rgb, res)

        # Check saturation
        rgb = utils.complex2rgb(10.0 * c)
        npt.assert_almost_equal(rgb, res, err_msg='Saturated values are not as expected.')

        # Check intensity scaling
        rgb = utils.complex2rgb(0.5 * c)
        npt.assert_almost_equal(rgb, 0.5 * res, err_msg='The intensity does not scale linearly with the amplitude.')

        c[1, 1] /= 2.0
        res[1, 1, :] /= 2.0
        rgb = utils.complex2rgb(0.5 * c)
        npt.assert_almost_equal(rgb, 0.5 * res, err_msg='Non-uniform amplitudes are not represented correctly.')

    def test_hsv2rgb(self):
        hsv = np.array([[[0, 1, 1], [1.0/3.0, 1, 1]], [[2.0/3.0, 1, 1], [1, 1, 0.5]]])
        res = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0.5, 0, 0]]])

        rgb = utils.hsv2rgb(hsv)
        npt.assert_almost_equal(rgb, res, err_msg='HSV to RGB hue test failed')

    def test_ranges2extent(self):
        npt.assert_almost_equal(utils.ranges2extent(np.arange(5), np.arange(10)), [-0.5, 9.5, 4.5, -0.5])
        npt.assert_almost_equal(utils.ranges2extent(np.arange(-2, 5-2), np.arange(-3, 10-3)), [-3.5, 6.5, 2.5, -2.5])
        npt.assert_almost_equal(utils.ranges2extent(np.arange(5), np.arange(0, 2, 0.2)), [-0.1, 1.9, 4.5, -0.5])
        npt.assert_almost_equal(utils.ranges2extent(np.arange(5), np.arange(-1, 2, 0.2)), [-1.1, 1.9, 4.5, -0.5])
        npt.assert_almost_equal(utils.ranges2extent(range(-2, 5-2), np.arange(-3, 10-3)), [-3.5, 6.5, 2.5, -2.5])
        npt.assert_almost_equal(utils.ranges2extent(range(-2, 5-2), range(-3, 10-3)), [-3.5, 6.5, 2.5, -2.5])
        npt.assert_almost_equal(utils.ranges2extent(np.arange(-2, 5-2), range(-3, 10-3)), [-3.5, 6.5, 2.5, -2.5])

    def test_word_align(self):
        # For now only checking that the arrays didn't change
        def check_array(a):
            npt.assert_almost_equal(utils.word_align(a), a, err_msg='default word_length=32 failed')
            npt.assert_almost_equal(utils.word_align(a, word_length=32), a, err_msg='setting word_length=32 failed')
            npt.assert_almost_equal(utils.word_align(a, word_length=16), a, err_msg='setting word_length=16 failed')

        check_array(np.arange(5))
        check_array(np.arange(8))
        check_array(np.ones([8, 24, 3]))
        check_array(np.arange(512))
        check_array(np.array([5]))
        check_array(np.array(5))


if __name__ == '__main__':
    unittest.main()