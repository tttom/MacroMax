import unittest
import numpy as np
import numpy.testing as npt

from tests.test_backend import BaseTestBackEnd

from macromax.backend.numpy import BackEndNumpy


class TestBackEndNumpy(BaseTestBackEnd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def BE(self) -> BackEndNumpy:
        return super().BE

    def test_calc_roots_of_low_order_polynomial(self):
        C = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.complex64)
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [0.0, 0.0, 0.0], err_msg='numerical accuracy test failed')

        C = np.array([1])
        root = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_equal(root, [], err_msg='failed root finding of constant equation')

        C = np.array([0])
        root = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_equal(root, [], err_msg='failed root finding of constant 0 equation')

        C = np.array([1, 0])  # 1 == 0
        root = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_equal(self.BE.asnumpy(root), np.nan, err_msg='failed root finding of degenerate linear equation')

        C = np.array([[1, 1], [0, 0]])  # 1 == 0
        root = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_equal(self.BE.asnumpy(root), np.nan, err_msg='failed root finding of multiple degenerate linear equation')

        C = np.array([0, 1])  # x == 0
        root = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_almost_equal(self.BE.asnumpy(root), 0.0, err_msg='zero root of linear equation not detected')

        C = np.array([-1, 1])  # (x-1) == 0
        root = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_almost_equal(self.BE.asnumpy(root), 1.0, err_msg='real root of linear equation not detected')

        C = np.array([-1j, 1])  # (x-1j) == 0
        root = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_almost_equal(self.BE.asnumpy(root), 1.0j, err_msg='imaginary root of linear equation not detected')

        C = np.array([-1-1j, 1])  # (x-1-1j) == 0
        root = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_almost_equal(self.BE.asnumpy(root), 1.0+1.0j, err_msg='complex root of linear equation not detected')

        C = np.array([2, 1, 0])  # x + 2 == 0
        C = self.BE.astype(C)
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [-2, np.nan],
                                      err_msg='failed root finding of degenerate quadratic equation')

        C = np.array([1, 0, 0])  # 1 == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [np.nan, np.nan],
                                      err_msg='failed root finding of very degenerate quadratic equation')

        C = np.array([0, 0, 0])  # 0 == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [np.nan, np.nan],
                                      err_msg='failed root finding of very degenerate quadratic equation')

        C = np.array([2, -3, 1])  # (x-1)(x-2) == x^2 - 3 x + 2 == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(self.BE.asnumpy(roots)), [1, 2],
                                      err_msg='two positive real roots of second order polynomial not detected')

        C = np.array([0, -1, 1])  # x(x-1) == x^2 - x == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(self.BE.asnumpy(roots)), [0, 1],
                                      err_msg='positive and zero roots of second order polynomial not detected')

        C = np.array([-1, 0, 1])  # (x-1)(x+1) == x^2 - 1 == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(self.BE.asnumpy(roots)), [-1, 1],
                                      err_msg='positive and negative roots of second order polynomial not detected')

        C = np.array([2, 3, 1])  # (x+2)(x+1) == x^2 + 3x + 2 == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(self.BE.asnumpy(roots)), [-2, -1],
                                      err_msg='negative roots of second order polynomial not detected')

        C = np.array([-1, 0, 1, 0])  # x**2 == 1
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [-1, 1, np.nan],
                                      err_msg='failed root finding of degenerate cubic polynomial')

        C = np.array([-2.7**3, 2.7**2 * 3, -2.7 * 3, 1])  # (x-2.7)**3 == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [2.7, 2.7, 2.7],
                                      err_msg='failed root finding of degenerate cubic polynomial', decimal=1)  # TODO: This is really poor accuracy, but actually sufficient for what we need

        C = np.array([1, 0, 1, 0])  # x**2 == -1
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [-1j, 1j, np.nan],
                                      err_msg='failed root finding of imaginary degenerate cubic polynomials')

        C = np.array([[1], [0], [1], [0]])  # x**2 == -1
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [[-1j], [1j], [np.nan]],
                                      err_msg='failed root finding of imaginary degenerate cubic polynomials with singleton dimension')

        C = np.array([[-1, 1], [0, 0], [1, 1], [0, 0]])  # x**2 == 1, x**2 == -1
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [[-1, -1j], [1, 1j], [np.nan, np.nan]],
                                      err_msg='failed root finding of two degenerate cubic polynomials')

        C = np.array([-6, 11, -6, 1])  # (x-1)(x-2)(x-3) == x^3 - 6 x^2 + 11 x - 6 == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [1, 2, 3],
                                      err_msg='three positive real roots of third order polynomial not detected')

        C = np.array([1, 0, 1])  # (x-1i)(x+1i) == x^2 + 1 == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [-1j, 1j],
                                      err_msg='two complex roots around zero of second order polynomial not detected')

        C = np.array([-1j, 1-1j, 1])  # (x+1)(x-1i) == x^2 + (1-1j)x -1j == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [-1, 1j],
                                      err_msg='two complex roots of second order polynomial not detected')

        C = np.array([0, -1, 0, 1])  # (x+1)(x-0)(x-1) == x^3 - x == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [-1, 0, 1],
                                      err_msg='three real roots, including zero, of third order polynomial not detected')

        C = np.array([6, 11, 6, 1])  # (x+1)(x+2)(x+3) == x^3 + 6 x^2 + 11 x + 6 == 0
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [-3, -2, -1],
                                      err_msg='three negative real roots of third order polynomial not detected')

        C = np.array([0, 1e-2, 0, -1])  # -x^3 + 1e-2 x == 0  => -0.1, 0.0, 0.1
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [-0.1, 0.0, 0.1],
                                      err_msg='third order polynomial real roots around zero not detected')

        C = np.array([0.0, 0.0, 0.100001, -1.0])
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [0.0, 0.0, C[2]],
                                      err_msg='numerical accuracy test failed', decimal=4)  # TODO: Can accuracy be improved here?

        C = np.array([0.0, -0.0064, 0.2, -1])
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(self.BE.asnumpy(roots), [0.0, 0.04, 0.16],
                                      err_msg='numerical accuracy test failed')

    def test_calc_complex_roots(self):
        C = np.array([-1, 0, 0, 1])  # x**3 == 1
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [np.exp(-2j*np.pi/3), np.exp(2j*np.pi/3), 1],
                                      err_msg='failed finding cubic roots centered around zero.')

        C = np.array([1, 2, 3, -4])
        roots = self.BE.calc_roots_of_low_order_polynomial(C)
        for root_idx in range(3):
            npt.assert_almost_equal(self.BE.asnumpy(self.BE.evaluate_polynomial(C, roots[root_idx])), 0.0,
                                    err_msg='third order polynomial root %d not found' % root_idx)

    def test_evaluate_polynomial(self):
        C = np.array([1, 2, 3, -4])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        result = self.BE.evaluate_polynomial(C, np.array([0, 1])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        npt.assert_almost_equal(result, np.array([1, 2])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])


if __name__ == '__main__':
    unittest.main()
