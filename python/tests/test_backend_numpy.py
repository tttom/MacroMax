import unittest
import numpy.testing as npt

from macromax.backend import BackEnd
from macromax.backend.numpy import BackEndNumpy
from macromax import Grid
import numpy as np
import scipy


class TestBackEnd(unittest.TestCase):
    def __init__(self, *args, dtype=np.complex64, internal_dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__nb_pol_dims: int = 3
        self.__grid: Grid = Grid([5, 10, 20], 100e-9)
        self.__wavenumber: float = 10e6
        if internal_dtype is None:
            internal_dtype = dtype
        self.__dtype = dtype
        self.__internal_dtype = internal_dtype

        self.__backend = None

    @property
    def nb_pol_dims(self) -> int:
        return self.__nb_pol_dims

    @property
    def grid(self) -> Grid:
        return self.__grid

    @property
    def wavenumber(self) -> float:
        return self.__wavenumber

    @property
    def dtype(self) -> float:
        return self.__dtype

    @property
    def internal_dtype(self) -> float:
        return self.__internal_dtype

    @property
    def BE(self) -> BackEnd:
        if self.__backend is None:
            self.__backend = BackEndNumpy(self.nb_pol_dims, self.grid * self.wavenumber, dtype=self.dtype)
        return self.__backend

    def setUp(self):
        self.__backend = None  # reset

    def test_ft_axes(self):
        npt.assert_equal(self.BE.ft_axes, [2, 3, 4])

    def test_sort(self):
        sample_tensor = [[[4, 10, 26],
                          [11,  1,  6],
                          [20,  8,  0]],
                         [[9,  2, 12],
                          [13, 18,  7],
                          [15, 16,  3]],
                         [[5, 23, 19],
                          [24, 17, 22],
                          [21, 25, 14]]]
        reference = np.sort(sample_tensor, axis=0)
        result = self.BE.asnumpy(self.BE.sort(self.BE.astype(sample_tensor)))
        npt.assert_array_equal(result, reference, err_msg='tensor sorted incorrectly')

    def test_dtype(self):
        npt.assert_equal(self.BE.dtype == self.internal_dtype, True,
                         f'BackEnd dtype not incorrectly reported ({self.BE.dtype} instead of {self.dtype}).')

    def test_astype(self):
        arr = self.BE.astype([1, 2, 3])
        npt.assert_equal(self.BE.asnumpy(arr), np.array([1, 2, 3], dtype=self.dtype), 'BackEnd astype did not work correctly.')
        arr = self.BE.astype([[1, 2, 3]])
        npt.assert_equal(self.BE.asnumpy(arr), np.array([[1, 2, 3]], dtype=self.dtype), 'BackEnd astype did not work correctly.')

    def test_asnumpy(self):
        arr = self.BE.astype([1, 2, 3])
        npt.assert_equal(self.BE.asnumpy(arr), np.array([1, 2, 3], dtype=self.dtype), 'BackEnd asnumpy did not work correctly.')
        arr = self.BE.astype([[1, 2, 3]])
        npt.assert_equal(self.BE.asnumpy(arr), np.array([[1, 2, 3]], dtype=self.dtype), 'BackEnd asnumpy did not work correctly.')

    def test_eps(self):
        npt.assert_equal(self.BE.eps, np.finfo(self.dtype).eps, 'Machine precision not reported correctly')

    def test_allocate_array(self):
        arr = self.BE.allocate_array()
        npt.assert_equal(arr.shape, self.BE.array_ft_input.shape, 'Allocated array incorrect.')

    def test_eye(self):
        result = self.BE.eye
        npt.assert_almost_equal(result.shape, np.array([3, 3, 1, 1, 1]), err_msg='eye shape not correct')
        npt.assert_array_equal(self.BE.asnumpy(result[:, :, 0, 0, 0]), np.eye(3), err_msg='eye property not identity matrix')

    def test_vector_length(self):
        npt.assert_almost_equal(self.BE.vector_length, 3, err_msg='returned vector length is incorrect')

    def test_grid(self):
        npt.assert_equal(self.BE.grid == self.grid * self.wavenumber, True, err_msg='grid not set correctly')

    def test_sample_pitch(self):
        npt.assert_almost_equal(self.BE.grid.step, (1, 1, 1), err_msg='returned sample pitch incorrect')

    def test_vectorial(self):
        npt.assert_equal(self.BE.vectorial, True, err_msg='vectorial property incorrectly set to False')

    def test_isscalar(self):
        npt.assert_equal(self.BE.is_scalar(np.pi), True, err_msg='is_scalar did not detect pi as a scalar')
        npt.assert_equal(self.BE.is_scalar(np.exp(1j)), True, err_msg='is_scalar did not detect complex number as a scalar')
        npt.assert_equal(self.BE.is_scalar(np.array(1.0)), True, err_msg='is_scalar did not detect singleton matrix as a scalar')
        npt.assert_equal(self.BE.is_scalar(np.array([1.0])), True, err_msg='is_scalar did not detect np.array([1.0]) as a scalar')
        npt.assert_equal(self.BE.is_scalar(np.array([[1.0]])), True, err_msg='is_scalar did not detect np.array([[1.0]]) as a scalar')
        npt.assert_equal(self.BE.is_scalar(np.array([[[1.0]]])), True, err_msg='is_scalar did not detect np.array([[[1.0]]]) as a scalar')
        npt.assert_equal(self.BE.is_scalar(np.array([1.0, 1.0])), False, err_msg='is_scalar detected np.array([1.0, 1.0]) as a scalar')

    def test_to_matrix_field(self):
        npt.assert_equal(self.BE.asnumpy(self.BE.to_matrix_field(2)), 2)
        npt.assert_equal(self.BE.asnumpy(self.BE.to_matrix_field(np.zeros([3, 3, *self.grid.shape]))), 0.0)
        npt.assert_equal(self.BE.asnumpy(self.BE.to_matrix_field(np.pi * np.ones([3, 3, *self.grid.shape]))), np.pi)
        npt.assert_equal(self.BE.asnumpy(self.BE.to_matrix_field(np.pi * np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis])),
                         np.pi * self.BE.asnumpy(self.BE.eye))
        npt.assert_equal(self.BE.to_matrix_field(np.arange(3)).shape, (1, 1, 1, 1, 3))
        A = self.BE.to_matrix_field(np.tile(np.arange(3)[:, np.newaxis, np.newaxis, np.newaxis], [1, *self.grid.shape]))
        npt.assert_equal(A.ndim, 5)
        npt.assert_equal(A.shape[:2], (3, 1))

    def test_ft(self):
        A = np.zeros([3, 3, *self.grid.shape], dtype=float)
        A[:, :, 0, 0, 0] = 1
        npt.assert_equal(self.BE.asnumpy(self.BE.ft(A)), np.ones([3, 3, *self.grid.shape]))
        B = np.arange(9 * np.prod(self.grid.shape)).reshape((3, 3, *self.grid.shape)).astype(self.dtype)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.ift(self.BE.ft(B))), B, decimal=3,
                                      err_msg='Fourier Transform did not work as expected.')

    def test_ift(self):
        A = np.zeros([3, 3, *self.grid.shape], dtype=float)
        A[:, :, 0, 0, 0] = 1
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.ift(A)), np.ones([3, 3, *self.grid.shape]) / np.prod(self.grid.shape))
        B = np.arange(9 * np.prod(self.grid.shape)).reshape((3, 3, *self.grid.shape)).astype(self.dtype)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.ft(self.BE.ift(B))), B, decimal=3,
                                      err_msg='Inverse Fourier Transform did not work as expected.')

    def test_conjugate_transpose(self):
        npt.assert_equal(self.BE.asnumpy(self.BE.adjoint(np.array([[[[[1.0]]]], [[[[1.0j]]]], [[[[0.0]]]]]))),
                         np.array([[[[[1.0]]], [[[-1.0j]]], [[[0.0]]]]]))

    def test_mat_subtract(self):
        npt.assert_equal(self.BE.asnumpy(self.BE.subtract(3 * self.BE.eye, 2)), self.BE.asnumpy(self.BE.eye))
        A = np.arange(9).reshape((3, 3, 1, 1, 1))
        B = np.arange(8, -1, -1).reshape((3, 3, 1, 1, 1))
        C = 2 * np.arange(9).reshape((3, 3, 1, 1, 1)) - 8
        npt.assert_equal(self.BE.subtract(A, B), C)

    def test_mat_mul(self):
        V = np.arange(3).reshape((3, 1, 1, 1, 1))
        A = np.arange(9).reshape((3, 3, 1, 1, 1))
        AF = np.tile(A, [1, 1, *self.BE.grid.shape])
        B = np.arange(8, -1, -1).reshape((3, 3, 1, 1, 1))
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(A, V)), A[:, 1:2, ...] + 2 * A[:, 2:3, ...])
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(AF, V)), A[:, 1:2, ...] + 2 * AF[:, 2:3, ...])
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(V[np.newaxis, :, 0, ...], A)), A[1:2] + 2 * A[2:3])
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(A, 2)), 2 * A)
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(3, A)), 3 * A)
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(A, B)), (A[:, :, 0, 0, 0] @ B[:, :, 0, 0, 0])[:, :, np.newaxis, np.newaxis, np.newaxis])

    def test_mat_ldivide(self):
        A = np.arange(9).reshape((3, 3, 1, 1, 1))**2
        npt.assert_equal(self.BE.asnumpy(self.BE.ldivide(2, A)), 0.5 * A)
        B = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        X = self.BE.ldivide(A, B)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.mul(A, X)), B, decimal=5)  # TODO: Can this be more accurate?

    def test_mat_inv(self):
        A = np.arange(9).reshape((3, 3, 1, 1, 1))**2
        A_inv = self.BE.ldivide(A, 1.0)
        npt.assert_almost_equal(self.BE.mul(A_inv, A), self.BE.eye, decimal=6)
        B = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        B_inv = self.BE.ldivide(B, 1.0)
        npt.assert_almost_equal(self.BE.mul(B_inv, B), self.BE.eye)

    # def test_curl(self):
    #     # Test not implemented yet
    #     self.fail()

    def test_curl_ft(self):
        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 0, 5] = np.array([3.0, 0, 0])
        B = self.BE.curl_ft(A)
        C = np.zeros([3, 1, *self.grid.shape], dtype=np.complex128)
        C[:, 0, 0, 0, 5] = np.array([0, 2*np.pi*0.25j * 3.0, 0])
        npt.assert_almost_equal(self.BE.asnumpy(B), C)

    # def test_transversal_projection(self):
    #     # Test not implemented yet
    #     self.fail()

    # def test_longitudinal_projection(self):
    #     # Test not implemented yet
    #     self.fail()

    def test_transversal_projection_ft(self):
        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 0, 10] = np.array([2, 3, 4])
        B = self.BE.transversal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 0, 10] = np.array([2, 3, 0])
        npt.assert_almost_equal(self.BE.asnumpy(B), C, decimal=6)

        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 0, 3] = np.array([2, 3, 4])
        B = self.BE.transversal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 0, 3] = np.array([2, 3, 0])
        npt.assert_almost_equal(self.BE.asnumpy(B), C, decimal=6)

        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 5, 10] = np.array([0, 3, 3])
        B = self.BE.transversal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 5, 10] = np.array([0, 0, 0])
        npt.assert_almost_equal(self.BE.asnumpy(B), C, decimal=6)

    def test_longitudinal_projection_ft(self):
        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 0, 10] = np.array([2, 3, 4])
        B = self.BE.longitudinal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 0, 10] = np.array([0, 0, 4])
        npt.assert_almost_equal(self.BE.asnumpy(B), C, decimal=6)

        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 0, 2] = np.array([2, 3, 4])
        B = self.BE.longitudinal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 0, 2] = np.array([0, 0, 4])
        npt.assert_array_almost_equal(self.BE.asnumpy(B), C)

        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 5, 10] = np.array([4, 5, 6])
        B = self.BE.longitudinal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 5, 10] = np.array([0, 5.5, 5.5])
        npt.assert_array_almost_equal(self.BE.asnumpy(B), C)

    def test_k(self):
        k = self.grid.k / self.wavenumber
        for _ in range(self.BE.grid.ndim):
            npt.assert_almost_equal(self.BE.asnumpy(self.BE.k[_]), k[_])

    def test_k2(self):
        A = self.BE.asnumpy(self.BE.k2)
        npt.assert_almost_equal(A[1, 1, 1] / ((2*np.pi/(self.grid.step * self.wavenumber)[0])**2),
                                np.sum(1.0 / (self.grid.shape ** 2)))

    def test_mat3_eig(self):
        D = np.diag([4, 5, 6j])[:, :, np.newaxis, np.newaxis, np.newaxis]
        nI = 2.7*np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis]
        # A = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        # H = A + self.PO.adjoint(A)
        npt.assert_almost_equal(self.BE.asnumpy(self.BE.mat3_eig(D)), np.array([6j, 4, 5]).reshape(3, 1, 1, 1),
                                decimal=5)

        Q = scipy.linalg.orth(np.arange(9).reshape([3, 3]) ** 2 + 1j * np.arange(9).reshape([3, 3]))
        QD = (Q @ np.diag([4, 5, 6j]) @ np.conj(Q.transpose()))[:, :, np.newaxis, np.newaxis, np.newaxis]
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.mat3_eig(QD)), np.array([6j, 4, 5]).reshape(3, 1, 1, 1),
                                      decimal=5)  # TODO: Can this be more accurate?

        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.mat3_eig(nI)), 2.7*np.ones((3, 1, 1, 1)), decimal=1)  # TODO: We need an iterative algorithm to calculate eigenvalues.

        def rot_x_d(a):
            return 1j * np.array([[0, 0, 0], [0, 0, -a], [0, a, 0]])[:, :, np.newaxis, np.newaxis, np.newaxis]

        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.mat3_eig(rot_x_d(0.1))[:, 0, 0, 0]),
                                      np.sort(np.linalg.eig(rot_x_d(0.1)[:, :, 0, 0, 0])[0], axis=0), decimal=6,
                                      err_msg='Three small real eigenvalues not detected correctly.')

        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.mat3_eig(rot_x_d(0.01))[:, 0, 0, 0]),
                                      np.sort(np.linalg.eig(rot_x_d(0.01)[:, :, 0, 0, 0])[0], axis=0), decimal=6,
                                      err_msg='Three real eigenvalues around zero not detected correctly.')

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

    def test_cross(self):
        A = np.array([1, 2, 3])
        B = np.array([3, 4, 5])
        npt.assert_almost_equal(self.BE.cross(A, B), np.array([-2, 4, -2]))

        A = A.reshape([3, 1, 1])
        B = B.reshape([3, 1, 1])
        npt.assert_almost_equal(self.BE.cross(A, B), np.array([-2, 4, -2]).reshape(3, 1, 1))

    def test_outer(self):
        A = np.array([1, 2]).reshape([2, 1])
        B = np.array([3, 4, 5]).reshape([3, 1])
        npt.assert_almost_equal(self.BE.outer(A, B), np.array([[3, 4, 5], [6, 8, 10]]))

        A = A.reshape([2, 1, 1])
        B = B.reshape([3, 1, 1])
        npt.assert_almost_equal(self.BE.outer(A, B), np.array([[3, 4, 5], [6, 8, 10]]).reshape([2, 3, 1]))

    def test_div_ft(self):
        # TODO: Add more in depth tests
        V = np.zeros([3, 1, *self.grid.shape], dtype=float)
        V[:, 0, 0, 0, 0] = np.array([1.0, 2.0, 3.0])
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.div_ft(V)), np.zeros([1, 1, *self.grid.shape]))

        M = np.zeros([3, 3, *self.grid.shape], dtype=float)
        M[:, :, 0, 0, 0] = np.arange(9).reshape([3, 3])
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.div_ft(M)), np.zeros([3, 1, *self.grid.shape]))

    def test_div(self):
        # TODO: Add more in depth tests
        V = np.zeros([3, 1, *self.grid.shape], dtype=float)
        V[0, ...] = 1.0
        V[1, ...] = 2.0
        V[2, ...] = 3.0
        npt.assert_almost_equal(self.BE.asnumpy(self.BE.div(V)), np.zeros([1, 1, *self.grid.shape]), decimal=6)

        M = np.tile(np.arange(9).reshape([3, 3, 1, 1, 1]), [1, 1, *self.grid.shape])
        npt.assert_almost_equal(self.BE.asnumpy(self.BE.div(M)), np.zeros([3, 1, *self.grid.shape]), decimal=6)


if __name__ == '__main__':
    unittest.main()
