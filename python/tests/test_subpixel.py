import unittest
import numpy.testing as npt
import numpy as np

from macromax.utils.ft.subpixel import roll, register


class TestRegistration(unittest.TestCase):
    def setUp(self):
        self.ref_int = np.zeros(shape=(3, 4), dtype=complex)
        self.ref_int[1, 1] = 1
        self.ref_int[2, 2] = 0.5
        self.ref_sub = roll(self.ref_int, shift=(-0.19, -0.3))
        self.sub_int = np.zeros_like(self.ref_int)
        self.sub_int[1, 2] = 0.5j
        self.sub_int[2, 3] = 0.25j

    def test_roll(self):
        npt.assert_array_almost_equal(roll(self.ref_sub, shift=(0, 0)), self.ref_sub,
                                      err_msg="(0, 0)-roll changed the array")
        npt.assert_array_almost_equal(roll(self.sub_int, shift=(0, -1)) / 0.5j, self.ref_int,
                                      err_msg="whole pixel roll failed")
        npt.assert_array_almost_equal(roll(self.sub_int, shift=(-0.19, -1.3)) / 0.5j, self.ref_sub,
                                      err_msg="fractional pixel roll failed")

    def test_no_registration(self):
        registered = register(self.sub_int, self.ref_int, precision=0)
        npt.assert_array_almost_equal(registered.image, np.zeros_like(self.sub_int))
        npt.assert_array_almost_equal(registered.shift, [0, 0])
        npt.assert_equal(registered.error, 1.0)
        registered = register(self.sub_int, self.ref_sub, precision=0)
        npt.assert_array_almost_equal(registered.shift, [0, 0])
        npt.assert_array_almost_equal(registered.image, self.sub_int / registered.factor)

    def test_whole_pixel_registration(self):
        registered = register(self.sub_int, self.ref_int, precision=1)
        npt.assert_array_almost_equal(registered.image, self.ref_int)
        npt.assert_array_almost_equal(registered.shift, [0, 1])
        npt.assert_almost_equal(registered.error, 0.0)
        npt.assert_almost_equal(registered.factor, 0.5j)
        registered = register(self.sub_int, self.ref_sub, precision=1)
        npt.assert_array_almost_equal(registered.shift, [0, 1])
        npt.assert_almost_equal(registered.factor, -0.13+0.57j, decimal=2)
        npt.assert_array_almost_equal(registered.image, self.ref_int * 0.5j / registered.factor)
        npt.assert_array_almost_equal(registered.image, np.roll(self.sub_int, -registered.shift) / registered.factor)
        npt.assert_almost_equal(registered.error,
                                np.linalg.norm(registered.image - self.ref_sub) / np.linalg.norm(self.ref_sub))
        npt.assert_equal(registered.error <
                         np.linalg.norm(registered.image * 1.0001 - self.ref_sub) / np.linalg.norm(self.ref_sub), True,
                         err_msg="The registered image could be closer to the reference.")
        npt.assert_equal(registered.error <
                         np.linalg.norm(registered.image * 0.9999 - self.ref_sub) / np.linalg.norm(self.ref_sub), True,
                         err_msg="The registered image could be closer to the reference.")

    def test_half_pixel_registration(self):
        registered = register(self.sub_int, self.ref_int, precision=1 / 2)
        npt.assert_array_almost_equal(registered.image, self.ref_int)
        npt.assert_array_almost_equal(registered.shift, [0, 1])
        npt.assert_almost_equal(registered.error, 0.0)
        registered = register(self.sub_int, self.ref_sub, precision=1 / 2)
        npt.assert_array_almost_equal(registered.shift, [0.0, 1.5])
        npt.assert_almost_equal(registered.factor, 0.09+0.57j, decimal=2)
        npt.assert_array_almost_equal(registered.image, roll(self.sub_int, -registered.shift) / registered.factor)
        npt.assert_almost_equal(registered.error,
                                np.linalg.norm(registered.image - self.ref_sub) / np.linalg.norm(self.ref_sub))
        npt.assert_equal(registered.error <
                         np.linalg.norm(registered.image * 1.0001 - self.ref_sub) / np.linalg.norm(self.ref_sub), True,
                         err_msg="The registered image could be closer to the reference.")
        npt.assert_equal(registered.error <
                         np.linalg.norm(registered.image * 0.9999 - self.ref_sub) / np.linalg.norm(self.ref_sub), True,
                         err_msg="The registered image could be closer to the reference.")

    def test_fractional_upsampling_factor_registration(self):
        registered = register(self.sub_int, self.ref_int, precision=1 / np.pi)
        npt.assert_array_almost_equal(registered.image, self.ref_int, decimal=1)
        npt.assert_array_almost_equal(registered.shift, [0, 0.95], decimal=2)
        npt.assert_almost_equal(registered.error, 0.08, decimal=2)
        registered = register(self.sub_int, self.ref_sub, precision=1 / np.pi)
        npt.assert_array_almost_equal(registered.image, self.ref_sub, decimal=1)
        npt.assert_array_almost_equal(registered.shift, [0.2, 1.3], decimal=1)
        npt.assert_array_almost_equal(registered.image, roll(self.sub_int, -registered.shift) / registered.factor)
        npt.assert_almost_equal(registered.error,
                                np.linalg.norm(registered.image - self.ref_sub) / np.linalg.norm(self.ref_sub))
        npt.assert_equal(registered.error <
                         np.linalg.norm(registered.image * 1.0001 - self.ref_sub) / np.linalg.norm(self.ref_sub), True,
                         err_msg="The registered image could be closer to the reference.")
        npt.assert_equal(registered.error <
                         np.linalg.norm(registered.image * 0.9999 - self.ref_sub) / np.linalg.norm(self.ref_sub), True,
                         err_msg="The registered image could be closer to the reference.")

    def test_large_upsampling_factor_registration(self):
        registered = register(self.sub_int, self.ref_int, precision=1 / 1000)
        npt.assert_array_almost_equal(registered.image, self.ref_int)
        npt.assert_array_almost_equal(registered.shift, [0, 1])
        npt.assert_almost_equal(registered.error, 0.0)
        registered = register(self.sub_int, self.ref_sub, precision=1 / 1000)
        npt.assert_array_almost_equal(registered.image, self.ref_sub)
        npt.assert_array_almost_equal(registered.shift, [0.2, 1.3], decimal=2)
        npt.assert_almost_equal(registered.error, 0.0)

    def test_default_precision(self):
        registered = register(self.sub_int, self.ref_int)
        npt.assert_array_almost_equal(registered.image, self.ref_int)
        npt.assert_array_almost_equal(registered.shift, [0, 1])
        npt.assert_almost_equal(registered.error, 0.0)
        registered = register(self.sub_int, self.ref_sub)
        npt.assert_array_almost_equal(registered.image, self.ref_sub, decimal=2)
        npt.assert_array_almost_equal(registered.shift, [0.2, 1.3], decimal=2)
        npt.assert_almost_equal(registered.error, 0.0, decimal=2)
