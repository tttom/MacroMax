import unittest
import numpy.testing as npt

from macromax.parallel_ops_column import ParallelOperations
import numpy as np
import scipy


class TestParallelOperations(unittest.TestCase):
    def setUp(self):
        self.nb_pol_dims = 3
        self.shape = np.array([5, 10, 20])
        self.sample_pitch = np.array([1, 1, 1]) * 100e-9
        self.wavenumber = 10e6
        self.PO = ParallelOperations(self.nb_pol_dims, self.shape, self.sample_pitch * self.wavenumber)

    def test_eye(self):
        result = self.PO.eye
        npt.assert_almost_equal(result.shape, np.array([3, 3, 1, 1, 1]), err_msg='eye shape not correct')
        npt.assert_almost_equal(result[:, :, 0, 0, 0], np.eye(3), err_msg='eye property not identity matrix')

    def test_nb_dims(self):
        npt.assert_almost_equal(self.PO.nb_dims, 3, err_msg='returned number of dimensions incorrect')

    def test_matrix_shape(self):
        npt.assert_almost_equal(self.PO.matrix_shape, (3, 3), err_msg='returned matrix shape incorrect')

    def test_data_shape(self):
        npt.assert_almost_equal(self.PO.data_shape, self.shape, err_msg='return data shape incorrect')

    def test_sample_pitch(self):
        npt.assert_almost_equal(self.PO.sample_pitch, (1, 1, 1), err_msg='returned sample pitch incorrect')

    def test_vectorial(self):
        npt.assert_almost_equal(self.PO.vectorial, True, err_msg='vectorial property incorrectly set to False')

    def test_isscalar(self):
        npt.assert_almost_equal(self.PO.is_scalar(np.pi), True, err_msg='is_scalar did not detect pi as a scalar')
        npt.assert_almost_equal(self.PO.is_scalar(np.exp(1j)), True, err_msg='is_scalar did not detect complex number as a scalar')
        npt.assert_almost_equal(self.PO.is_scalar(np.array(1.0)), True, err_msg='is_scalar did not detect singleton matrix as a scalar')
        npt.assert_almost_equal(self.PO.is_scalar(np.array([1.0])), True, err_msg='is_scalar did not detect np.array([1.0]) as a scalar')
        npt.assert_almost_equal(self.PO.is_scalar(np.array([[1.0]])), True, err_msg='is_scalar did not detect np.array([[1.0]]) as a scalar')
        npt.assert_almost_equal(self.PO.is_scalar(np.array([[[1.0]]])), True, err_msg='is_scalar did not detect np.array([[[1.0]]]) as a scalar')
        npt.assert_almost_equal(self.PO.is_scalar(np.array([1.0, 1.0])), False, err_msg='is_scalar detected np.array([1.0, 1.0]) as a scalar')

    def test_to_simple_matrix(self):
        npt.assert_almost_equal(self.PO.to_simple_matrix(2), 2)
        npt.assert_almost_equal(self.PO.to_simple_matrix(np.zeros([3, 3, *self.shape])), 0.0)
        npt.assert_almost_equal(self.PO.to_simple_matrix(np.pi * np.ones([3, 3, *self.shape])), np.pi)
        npt.assert_almost_equal(self.PO.to_simple_matrix(np.pi * np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis]),
                                np.pi * self.PO.eye)
        npt.assert_almost_equal(self.PO.to_simple_matrix(np.arange(3)).shape, (3, 1, 1, 1, 1))
        A = self.PO.to_simple_matrix(np.tile(np.arange(3)[:, np.newaxis, np.newaxis, np.newaxis], [1, *self.shape]))
        npt.assert_almost_equal(A.ndim, 5)
        npt.assert_almost_equal(A.shape[:2], (3, 1))

    def test_to_simple_vector(self):
        npt.assert_almost_equal(self.PO.to_simple_vector(2), 2 * np.ones([3, 1, 1, 1, 1]))
        A = self.PO.to_simple_matrix(np.tile(np.arange(3)[:, np.newaxis, np.newaxis, np.newaxis], [1, *self.shape]))
        npt.assert_almost_equal(A.ndim, 5)
        npt.assert_almost_equal(A.shape[:2], (3, 1))

    def test_to_full_matrix(self):
        npt.assert_almost_equal(self.PO.to_full_matrix(None), 0.0 * self.PO.eye)
        npt.assert_almost_equal(self.PO.to_full_matrix(3), 3.0 * self.PO.eye)
        npt.assert_almost_equal(self.PO.to_full_matrix(np.pi), np.pi * self.PO.eye)
        npt.assert_almost_equal(self.PO.to_full_matrix(np.eye(3)), self.PO.eye)
        npt.assert_almost_equal(self.PO.to_full_matrix(np.arange(9).reshape(3, 3)), np.arange(9).reshape(3, 3, 1, 1, 1))
        npt.assert_almost_equal(self.PO.to_full_matrix(np.ones([3, 3])), np.ones([3, 3, 1, 1, 1]))

    def test_ft(self):
        A = np.zeros([3, 3, *self.shape], dtype=float)
        A[:, :, 0, 0, 0] = 1
        npt.assert_almost_equal(self.PO.ft(A), np.ones([3, 3, *self.shape]))
        B = np.arange(9 * np.prod(self.shape)).reshape(3, 3, *self.shape)
        npt.assert_almost_equal(self.PO.ift(self.PO.ft(B)), B)

    def test_ift(self):
        A = np.zeros([3, 3, *self.shape], dtype=float)
        A[:, :, 0, 0, 0] = 1
        npt.assert_almost_equal(self.PO.ift(A), np.ones([3, 3, *self.shape]) / np.prod(self.shape))
        B = np.arange(9 * np.prod(self.shape)).reshape(3, 3, *self.shape)
        npt.assert_almost_equal(self.PO.ft(self.PO.ift(B)), B)

    def test_transpose(self):
        npt.assert_almost_equal(self.PO.transpose(np.array([[[[[1.0]]]], [[[[1.0j]]]], [[[[0.0]]]]])),
                                np.array([[[[[1.0]]], [[[1.0j]]], [[[0.0]]]]]))

    def test_conjugate_transpose(self):
        npt.assert_almost_equal(self.PO.conjugate_transpose(np.array([[[[[1.0]]]], [[[[1.0j]]]], [[[[0.0]]]]])),
                                np.array([[[[[1.0]]], [[[-1.0j]]], [[[0.0]]]]]))

    def test_mat_add(self):
        npt.assert_almost_equal(self.PO.add(3 * self.PO.eye, 2), 5 * self.PO.eye)
        A = np.arange(9).reshape(3, 3, 1, 1, 1)
        B = np.arange(8, -1, -1).reshape(3, 3, 1, 1, 1)
        npt.assert_almost_equal(self.PO.add(A, B), 0 * A + 8)

    def test_mat_subtract(self):
        npt.assert_almost_equal(self.PO.subtract(3 * self.PO.eye, 2), self.PO.eye)
        A = np.arange(9).reshape(3, 3, 1, 1, 1)
        B = np.arange(8, -1, -1).reshape(3, 3, 1, 1, 1)
        C = 2 * np.arange(9).reshape(3, 3, 1, 1, 1) - 8
        npt.assert_almost_equal(self.PO.subtract(A, B), C)

    def test_mat_mul(self):
        V = np.arange(3).reshape(3, 1, 1, 1, 1)
        A = np.arange(9).reshape(3, 3, 1, 1, 1)
        AF = np.tile(A, [1, 1, *self.PO.data_shape])
        B = np.arange(8, -1, -1).reshape(3, 3, 1, 1, 1)
        npt.assert_almost_equal(self.PO.mul(A, V), A[:, 1:2, ...] + 2 * A[:, 2:3, ...])
        npt.assert_almost_equal(self.PO.mul(AF, V), A[:, 1:2, ...] + 2 * AF[:, 2:3, ...])
        npt.assert_almost_equal(self.PO.mul(V[np.newaxis, :, 0, ...], A), A[1:2] + 2 * A[2:3])
        npt.assert_almost_equal(self.PO.mul(A, 2), 2 * A)
        npt.assert_almost_equal(self.PO.mul(3, A), 3 * A)
        npt.assert_almost_equal(self.PO.mul(A, B), (A[:, :, 0, 0, 0] @ B[:, :, 0, 0, 0])[:, :, np.newaxis, np.newaxis, np.newaxis])

    def test_mat_ldivide(self):
        A = np.arange(9).reshape(3, 3, 1, 1, 1) ** 2
        npt.assert_almost_equal(self.PO.ldivide(2, A), 0.5 * A)
        B = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        X = self.PO.ldivide(A, B)
        npt.assert_almost_equal(self.PO.mul(A, X), B)

    def test_mat_inv(self):
        A = np.arange(9).reshape(3, 3, 1, 1, 1) ** 2
        A_inv = self.PO.ldivide(A, 1.0)
        npt.assert_almost_equal(self.PO.mul(A_inv, A), self.PO.eye)
        B = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        B_inv = self.PO.ldivide(B, 1.0)
        npt.assert_almost_equal(self.PO.mul(B_inv, B), self.PO.eye)

    # def test_curl(self):
    #     # Test not implemented yet
    #     self.fail()

    def test_curl_ft(self):
        A = np.zeros([3, 1, *self.shape])
        A[:, 0, 0, 0, 5] = np.array([3.0, 0, 0])
        B = self.PO.curl_ft(A)
        C = np.zeros([3, 1, *self.shape], dtype=np.complex128)
        C[:, 0, 0, 0, 5] = np.array([0, 2*np.pi*0.25j * 3.0, 0])
        npt.assert_almost_equal(B, C)

    # def test_transversal_projection(self):
    #     # Test not implemented yet
    #     self.fail()

    # def test_longitudinal_projection(self):
    #     # Test not implemented yet
    #     self.fail()

    def test_transversal_projection_ft(self):
        A = np.zeros([3, 1, *self.shape])
        A[:, 0, 0, 0, 10] = np.array([2, 3, 4])
        B = self.PO.transversal_projection_ft(A)
        C = np.zeros([3, 1, *self.shape])
        C[:, 0, 0, 0, 10] = np.array([2, 3, 0])
        npt.assert_almost_equal(B, C)

        A = np.zeros([3, 1, *self.shape])
        A[:, 0, 0, 0, 3] = np.array([2, 3, 4])
        B = self.PO.transversal_projection_ft(A)
        C = np.zeros([3, 1, *self.shape])
        C[:, 0, 0, 0, 3] = np.array([2, 3, 0])
        npt.assert_almost_equal(B, C)

        A = np.zeros([3, 1, *self.shape])
        A[:, 0, 0, 5, 10] = np.array([0, 3, 3])
        B = self.PO.transversal_projection_ft(A)
        C = np.zeros([3, 1, *self.shape])
        C[:, 0, 0, 5, 10] = np.array([0, 0, 0])
        npt.assert_almost_equal(B, C)

    def test_longitudinal_projection_ft(self):
        A = np.zeros([3, 1, *self.shape])
        A[:, 0, 0, 0, 10] = np.array([2, 3, 4])
        B = self.PO.longitudinal_projection_ft(A)
        C = np.zeros([3, 1, *self.shape])
        C[:, 0, 0, 0, 10] = np.array([0, 0, 4])
        npt.assert_almost_equal(B, C)

        A = np.zeros([3, 1, *self.shape])
        A[:, 0, 0, 0, 2] = np.array([2, 3, 4])
        B = self.PO.longitudinal_projection_ft(A)
        C = np.zeros([3, 1, *self.shape])
        C[:, 0, 0, 0, 2] = np.array([0, 0, 4])
        npt.assert_almost_equal(B, C)

        A = np.zeros([3, 1, *self.shape])
        A[:, 0, 0, 5, 10] = np.array([4, 5, 6])
        B = self.PO.longitudinal_projection_ft(A)
        C = np.zeros([3, 1, *self.shape])
        C[:, 0, 0, 5, 10] = np.array([0, 5.5, 5.5])
        npt.assert_almost_equal(B, C)

    def test_calc_K2(self):
        A = self.PO.calc_K2()
        npt.assert_almost_equal(A[0, 0, 1, 1, 1] / ((2*np.pi/(self.sample_pitch * self.wavenumber)[0])**2),
                                np.sum(1.0 / (self.shape ** 2)))

    def test_mat3_eig(self):
        D = np.diag([4, 5, 6j])[:, :, np.newaxis, np.newaxis, np.newaxis]
        nI = 2.7*np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis]
        # A = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        # H = A + self.PO.conjugate_transpose(A)
        npt.assert_almost_equal(np.sort(self.PO.mat3_eig(D), axis=0), np.array([6j, 4, 5]).reshape(3, 1, 1, 1))

        Q = scipy.linalg.orth(np.arange(9).reshape([3, 3]) ** 2 + 1j * np.arange(9).reshape([3, 3]))
        QD = (Q @ np.diag([4, 5, 6j]) @ np.conj(Q.transpose()))[:, :, np.newaxis, np.newaxis, np.newaxis]
        npt.assert_almost_equal(np.sort(self.PO.mat3_eig(QD), axis=0), np.array([6j, 4, 5]).reshape(3, 1, 1, 1))

        npt.assert_almost_equal(np.sort(self.PO.mat3_eig(nI), axis=0), 2.7*np.ones((3, 1, 1, 1)), decimal=4)

        def rot_x_d(a):
            return 1j * np.array([[0, 0, 0], [0, 0, -a], [0, a, 0]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        npt.assert_almost_equal(np.sort(self.PO.mat3_eig(rot_x_d(0.01))[:, 0, 0, 0], axis=0),
                                np.sort(np.linalg.eig(rot_x_d(0.01)[:, :, 0, 0, 0])[0], axis=0), decimal=4,
                                err_msg='Three real eigenvalues around zero not detected correctly.')

    def test_calc_roots_of_low_order_polynomial(self):
        C = np.array([0.0, 0.0, 0.100001, -1.0])
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [0.0, 0.0, C[2]], err_msg='numerical accuracy test failed')

        C = np.array([1])
        root = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_equal(root, [], err_msg='failed root finding of constant equation')

        C = np.array([0])
        root = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_equal(root, [], err_msg='failed root finding of constant 0 equation')

        C = np.array([1, 0])  # 1 == 0
        root = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_equal(root, np.nan, err_msg='failed root finding of degenerate linear equation')

        C = np.array([[1, 1], [0, 0]])  # 1 == 0
        root = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_equal(root, np.nan, err_msg='failed root finding of multiple degenerate linear equation')

        C = np.array([0, 1])  # x == 0
        root = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_almost_equal(root, 0.0, err_msg='zero root of linear equation not detected')

        C = np.array([-1, 1])  # (x-1) == 0
        root = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_almost_equal(root, 1.0, err_msg='real root of linear equation not detected')

        C = np.array([-1j, 1])  # (x-1j) == 0
        root = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_almost_equal(root, 1.0j, err_msg='imaginary root of linear equation not detected')

        C = np.array([-1-1j, 1])  # (x-1-1j) == 0
        root = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_almost_equal(root, 1.0+1.0j, err_msg='complex root of linear equation not detected')

        C = np.array([2, 1, 0])  # x + 2 == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(roots, [-2, np.nan],
                                      err_msg='failed root finding of degenerate quadratic equation')

        C = np.array([1, 0, 0])  # 1 == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(roots, [np.nan, np.nan],
                                      err_msg='failed root finding of very degenerate quadratic equation')

        C = np.array([0, 0, 0])  # 0 == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(roots, [np.nan, np.nan],
                                      err_msg='failed root finding of very degenerate quadratic equation')

        C = np.array([2, -3, 1])  # (x-1)(x-2) == x^2 - 3 x + 2 == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [1, 2],
                                      err_msg='two positive real roots of second order polynomial not detected')

        C = np.array([0, -1, 1])  # x(x-1) == x^2 - x == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [0, 1],
                                      err_msg='positive and zero roots of second order polynomial not detected')

        C = np.array([-1, 0, 1])  # (x-1)(x+1) == x^2 - 1 == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [-1, 1],
                              err_msg='positive and negative roots of second order polynomial not detected')

        C = np.array([2, 3, 1])  # (x+2)(x+1) == x^2 + 3x + 2 == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [-2, -1],
                                      err_msg='negative roots of second order polynomial not detected')

        C = np.array([-1, 0, 1, 0])  # x**2 == 1
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(roots, [-1, 1, np.nan],
                                      err_msg='failed root finding of degenerate cubic polynomial')

        C = np.array([1, 0, 1, 0])  # x**2 == -1
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(roots, [-1j, 1j, np.nan],
                                      err_msg='failed root finding of imaginary degenerate cubic polynomials')

        C = np.array([-1, 0, 0, 1])  # x**3 == 1
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [np.exp(-2j*np.pi/3), np.exp(2j*np.pi/3), 1],
                                  err_msg='failed finding cubic roots centered around zero.')

        C = np.array([[1], [0], [1], [0]])  # x**2 == -1
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(roots, [[-1j], [1j], [np.nan]],
                                      err_msg='failed root finding of imaginary degenerate cubic polynomials with singleton dimension')

        C = np.array([[-1, 1], [0, 0], [1, 1], [0, 0]])  # x**2 == 1, x**2 == -1
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(roots, [[-1, -1j], [1, 1j], [np.nan, np.nan]],
                                      err_msg='failed root finding of two degenerate cubic polynomials')

        C = np.array([-6, 11, -6, 1])  # (x-1)(x-2)(x-3) == x^3 - 6 x^2 + 11 x - 6 == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [1, 2, 3],
                                      err_msg='three positive real roots of third order polynomial not detected')

        C = np.array([1, 0, 1])  # (x-1i)(x+1i) == x^2 + 1 == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [-1j, 1j],
                                      err_msg='two complex roots around zero of second order polynomial not detected')

        C = np.array([-1j, 1-1j, 1])  # (x+1)(x-1i) == x^2 + (1-1j)x -1j == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [-1, 1j],
                                      err_msg='two complex roots of second order polynomial not detected')

        C = np.array([0, -1, 0, 1])  # (x+1)(x-0)(x-1) == x^3 - x == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [-1, 0, 1],
                                      err_msg='three real roots, including zero, of third order polynomial not detected')

        C = np.array([6, 11, 6, 1])  # (x+1)(x+2)(x+3) == x^3 + 6 x^2 + 11 x + 6 == 0
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [-3, -2, -1],
                                      err_msg='three negative real roots of third order polynomial not detected')

        C = np.array([0, 1e-2, 0, -1])  # -x^3 + 1e-2 x == 0  => -0.1, 0.0, 0.1
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [-0.1, 0.0, 0.1],
                                      err_msg='third order polynomial real roots around zero not detected')

        C = np.array([1, 2, 3, -4])
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        for root_idx in range(3):
            npt.assert_almost_equal(self.PO.evaluate_polynomial(C, roots[root_idx]), 0.0,
                                    err_msg='third order polynomial root %d not found' % root_idx)

        C = np.array([0.0, 0.0, 0.100001, -1.0])
        roots = self.PO.calc_roots_of_low_order_polynomial(C)
        npt.assert_array_almost_equal(np.sort(roots), [0.0, 0.0, C[2]], err_msg='numerical accuracy test failed')

    def test_evaluate_polynomial(self):
        C = np.array([1, 2, 3, -4])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        result = self.PO.evaluate_polynomial(C, np.array([0, 1])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        npt.assert_almost_equal(result, np.array([1, 2])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])

    def test_cross(self):
        A = np.array([1, 2, 3])
        B = np.array([3, 4, 5])
        npt.assert_almost_equal(self.PO.cross(A, B), np.array([-2, 4, -2]))

        A = A.reshape([3, 1, 1])
        B = B.reshape([3, 1, 1])
        npt.assert_almost_equal(self.PO.cross(A, B), np.array([-2, 4, -2]).reshape(3, 1, 1))

    def test_outer(self):
        A = np.array([1, 2]).reshape([2, 1])
        B = np.array([3, 4, 5]).reshape([3, 1])
        npt.assert_almost_equal(self.PO.outer(A, B), np.array([[3, 4, 5], [6, 8, 10]]))

        A = A.reshape([2, 1, 1])
        B = B.reshape([3, 1, 1])
        npt.assert_almost_equal(self.PO.outer(A, B), np.array([[3, 4, 5], [6, 8, 10]]).reshape([2, 3, 1]))

    def test_div_ft(self):
        # TODO: Add more in depth tests
        V = np.zeros([3, 1, *self.shape], dtype=float)
        V[:, 0, 0, 0, 0] = np.array([1.0, 2.0, 3.0])
        npt.assert_almost_equal(self.PO.div_ft(V), np.zeros([1, 1, *self.shape]))

        M = np.zeros([3, 3, *self.shape], dtype=float)
        M[:, :, 0, 0, 0] = np.arange(9).reshape([3, 3])
        npt.assert_almost_equal(self.PO.div_ft(M), np.zeros([3, 1, *self.shape]))

    def test_div(self):
        # TODO: Add more in depth tests
        V = np.zeros([3, 1, *self.shape], dtype=float)
        V[0, ...] = 1.0
        V[1, ...] = 2.0
        V[2, ...] = 3.0
        npt.assert_almost_equal(self.PO.div(V), np.zeros([1, 1, *self.shape]))

        M = np.tile(np.arange(9).reshape([3, 3, 1, 1, 1]), [1, 1, *self.shape])
        npt.assert_almost_equal(self.PO.div(M), np.zeros([3, 1, *self.shape]))


if __name__ == '__main__':
    unittest.main()