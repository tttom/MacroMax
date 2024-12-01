import unittest
import numpy.testing as npt

from macromax.backend.numpy import BackEndNumpy
from macromax import Grid
import numpy as np
import scipy


class BaseTestBackEnd(unittest.TestCase):
    def __init__(self, *args, dtype=np.complex64, hardware_dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__nb_pol_dims: int = 3
        self.__grid: Grid = Grid([5, 10, 20], 100e-9)
        self.__wavenumber: float = 10e6
        if hardware_dtype is None:
            hardware_dtype = dtype
        self.__dtype = dtype
        self.__hardware_dtype = hardware_dtype
        self.BE = BackEndNumpy(self.nb_pol_dims, self.grid * self.wavenumber, hardware_dtype=self.dtype)

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
    def dtype(self):
        return self.__dtype

    @property
    def hardware_dtype(self):
        return self.__hardware_dtype

    def setUp(self):
        self.__backend = None  # reset

    def test_ft_axes(self):
        npt.assert_equal(self.BE.ft_axes, [2, 3, 4])

    def test_assign_exact(self):
        x = self.BE.allocate_array()
        x_np = self.BE.astype(np.arange(np.prod(x.shape)).reshape(x.shape))
        x = self.BE.assign_exact(x_np, x)
        npt.assert_array_equal(self.BE.asnumpy(x), self.BE.asnumpy(x_np))

    def test_assign(self):
        x = self.BE.allocate_array()
        x_np = np.arange(np.prod(x.shape)).reshape(x.shape).astype(self.BE.numpy_dtype)
        x = self.BE.assign(x_np, x)
        npt.assert_array_equal(self.BE.asnumpy(x), x_np)
        x = self.BE.assign(0, x)
        npt.assert_array_equal(self.BE.asnumpy(x), 0)
        x = self.BE.assign(1, x)
        npt.assert_array_equal(self.BE.asnumpy(x), 1)

    def test_allclose(self):
        arr = self.BE.allocate_array()
        other = self.BE.allocate_array()
        arr = self.BE.assign(0.0, arr)
        other = self.BE.assign(0, other)
        npt.assert_equal(self.BE.allclose(arr, other), True)
        other = self.BE.assign(1e-9, other)
        npt.assert_equal(self.BE.allclose(arr, other), True)
        other = self.BE.assign(0.1, other)
        npt.assert_equal(self.BE.allclose(arr, other), False)

    def test_amax(self):
        arr = self.BE.allocate_array()  # TODO complex64
        arr = self.BE.real(arr)
        arr = self.BE.assign(1.0, arr)  # TODO assign an array of 1s to arr which is complex
        npt.assert_array_equal(self.BE.amax(arr), 1.0)
        sh = [1 + 2 * self.BE.vectorial, *self.BE.grid.shape]
        arr = self.BE.assign(np.arange(np.prod(sh)).reshape(sh), arr)
        npt.assert_array_equal(self.BE.amax(arr), np.prod(sh) - 1)

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
        npt.assert_array_equal(result, reference, err_msg=f'tensor {reference} sorted incorrectly as {result}')

    def test_dtype(self):
        npt.assert_equal(self.BE.hardware_dtype is self.hardware_dtype, True,
                         f'BackEnd dtype not incorrectly reported ({self.BE.hardware_dtype} instead of {self.dtype}).')

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
        npt.assert_array_equal(arr.shape, self.BE.array_ft_input.shape, 'Allocated array incorrect.')

    def test_eye(self):
        result = self.BE.eye
        npt.assert_array_almost_equal(result.shape, np.array([3, 3, 1, 1, 1]), err_msg='eye shape not correct')
        npt.assert_array_equal(self.BE.asnumpy(result[:, :, 0, 0, 0]), np.eye(3), err_msg='eye property not identity matrix')

    def test_vector_length(self):
        npt.assert_array_almost_equal(self.BE.vector_length, 3, err_msg='returned vector length is incorrect')

    def test_grid(self):
        npt.assert_equal(self.BE.grid == self.grid * self.wavenumber, True, err_msg='grid not set correctly')

    def test_sample_pitch(self):
        npt.assert_array_almost_equal(self.BE.grid.step, (1, 1, 1), err_msg='returned sample pitch incorrect')

    def test_vectorial(self):
        npt.assert_equal(self.BE.vectorial, True, err_msg='vectorial property incorrectly set to False')

    def test_isscalar(self):
        npt.assert_equal(self.BE.is_scalar(self.BE.astype(np.pi)), True, err_msg='is_scalar did not detect pi as a scalar')
        npt.assert_equal(self.BE.is_scalar(self.BE.astype(np.exp(1j))), True, err_msg='is_scalar did not detect complex number as a scalar')
        npt.assert_equal(self.BE.is_scalar(self.BE.astype(np.array(1.0))), True, err_msg='is_scalar did not detect singleton matrix as a scalar')
        npt.assert_equal(self.BE.is_scalar(self.BE.astype(np.array([1.0]))), True, err_msg='is_scalar did not detect np.array([1.0]) as a scalar')
        npt.assert_equal(self.BE.is_scalar(self.BE.astype(np.array([[1.0]]))), True, err_msg='is_scalar did not detect np.array([[1.0]]) as a scalar')
        npt.assert_equal(self.BE.is_scalar(self.BE.astype(np.array([[[1.0]]]))), True, err_msg='is_scalar did not detect np.array([[[1.0]]]) as a scalar')
        npt.assert_equal(self.BE.is_scalar(self.BE.astype(np.array([1.0, 1.0]))), False, err_msg='is_scalar detected np.array([1.0, 1.0]) as a scalar')

    def test_to_matrix_field(self):
        npt.assert_equal(self.BE.asnumpy(self.BE.to_matrix_field(2)), 2)
        npt.assert_equal(self.BE.asnumpy(self.BE.to_matrix_field(np.zeros([3, 3, *self.grid.shape]))), 0.0)
        npt.assert_array_almost_equal(
            self.BE.asnumpy(self.BE.to_matrix_field(np.pi * np.ones([3, 3, *self.grid.shape]))),
            np.pi
        )
        npt.assert_equal(self.BE.asnumpy(self.BE.to_matrix_field(np.pi * np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis])),
                         np.pi * self.BE.asnumpy(self.BE.eye))
        npt.assert_array_equal(self.BE.to_matrix_field(np.arange(3)).shape, (1, 1, 1, 1, 3))
        A = self.BE.to_matrix_field(np.tile(np.arange(3)[:, np.newaxis, np.newaxis, np.newaxis], [1, *self.grid.shape]))
        npt.assert_equal(A.ndim, 5)
        npt.assert_array_equal(A.shape[:2], (3, 1))

    def test_ft(self):
        A = np.zeros([3, 3, *self.grid.shape], dtype=float)
        A[:, :, 0, 0, 0] = 1
        A = self.BE.astype(A)
        npt.assert_almost_equal(self.BE.asnumpy(self.BE.ft(A)), np.ones([3, 3, *self.grid.shape]), decimal=14)
        B_np = np.arange(9 * np.prod(self.grid.shape)).reshape((3, 3, *self.grid.shape))
        B = self.BE.astype(B_np.copy())
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.ift(self.BE.ft(B))), B_np, decimal=3,
                                      err_msg='Fourier Transform did not work as expected.')

    def test_ift(self):
        A = np.zeros([3, 3, *self.grid.shape], dtype=float)
        A[:, :, 0, 0, 0] = 1
        A = self.BE.astype(A.copy())
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.ift(A)), np.ones([3, 3, *self.grid.shape]) / np.prod(self.grid.shape))
        B = np.arange(9 * np.prod(self.grid.shape)).reshape((3, 3, *self.grid.shape)).astype(self.dtype)
        B = self.BE.copy(B)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.ft(self.BE.ift(B))), self.BE.asnumpy(B), decimal=3,
                                      err_msg='Inverse Fourier Transform did not work as expected.')

    def test_conjugate_transpose(self):
        A = np.array([[[[[1.0]]]], [[[[1.0j]]]], [[[[0.0]]]]])
        A = self.BE.astype(A)
        A_adj = np.array([[[[[1.0]]], [[[-1.0j]]], [[[0.0]]]]])
        npt.assert_equal(self.BE.asnumpy(self.BE.adjoint(A)), A_adj)

    def test_mat_subtract(self):
        npt.assert_equal(self.BE.asnumpy(self.BE.subtract(3 * self.BE.eye, 2)), self.BE.asnumpy(self.BE.eye))
        A = np.arange(9).reshape((3, 3, 1, 1, 1))
        B = np.arange(8, -1, -1).reshape((3, 3, 1, 1, 1))
        C = 2 * np.arange(9).reshape((3, 3, 1, 1, 1)) - 8
        A = self.BE.astype(A)
        B = self.BE.astype(B)
        npt.assert_array_equal(self.BE.asnumpy(self.BE.subtract(A, B)), C)

    def test_mat_mul(self):
        V = np.arange(3).reshape((3, 1, 1, 1, 1))
        A = np.arange(9).reshape((3, 3, 1, 1, 1))
        A = self.BE.astype(A)
        AF = np.tile(self.BE.asnumpy(A), [1, 1, *self.BE.grid.shape])
        AF = self.BE.astype(AF)
        B = np.arange(8, -1, -1).reshape((3, 3, 1, 1, 1))
        B = self.BE.astype(B)
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(A, V)), self.BE.asnumpy(A[:, 1:2, ...] + 2 * A[:, 2:3, ...]))
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(AF, V)), self.BE.asnumpy(A[:, 1:2, ...] + 2 * AF[:, 2:3, ...]))
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(V[np.newaxis, :, 0, ...], A)), self.BE.asnumpy(A[1:2] + 2 * A[2:3]))
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(A, 2)), self.BE.asnumpy(2 * A))
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(3, A)), self.BE.asnumpy(3 * A))
        npt.assert_equal(self.BE.asnumpy(self.BE.mul(A, B)), self.BE.asnumpy((A[:, :, 0, 0, 0] @ B[:, :, 0, 0, 0])[:, :, np.newaxis, np.newaxis, np.newaxis]))

    def test_mat_ldivide(self):
        A = np.arange(9).reshape((3, 3, 1, 1, 1))**2
        A = self.BE.astype(A)
        npt.assert_equal(self.BE.asnumpy(self.BE.ldivide(2, A)), self.BE.asnumpy(0.5 * A))
        B = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        X = self.BE.ldivide(A, B)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.mul(A, X)), B, decimal=5)  # TODO: Can this be more accurate?

    def test_mat_inv(self):
        A = np.arange(9).reshape((3, 3, 1, 1, 1))**2
        A = self.BE.astype(A)
        A_inv = self.BE.ldivide(A, 1.0)
        npt.assert_array_almost_equal(self.BE.mul(A_inv, A), self.BE.eye, decimal=6)
        B = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        B = self.BE.astype(B)
        B_inv = self.BE.ldivide(B, 1.0)
        npt.assert_array_almost_equal(self.BE.mul(B_inv, B), self.BE.eye)

    # def test_curl(self):
    #     # Test not implemented yet
    #     self.fail()

    def test_curl_ft(self):
        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 0, 5] = np.array([3.0, 0, 0])
        A = self.BE.astype(A)
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
        A = self.BE.astype(A)
        B = self.BE.transversal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 0, 10] = np.array([2, 3, 0])
        npt.assert_array_almost_equal(self.BE.asnumpy(B), C, decimal=6)

        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 0, 3] = np.array([2, 3, 4])
        A = self.BE.astype(A)
        B = self.BE.transversal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 0, 3] = np.array([2, 3, 0])
        npt.assert_array_almost_equal(self.BE.asnumpy(B), C, decimal=6)

        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 5, 10] = np.array([0, 3, 3])
        A = self.BE.astype(A)
        B = self.BE.transversal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 5, 10] = np.array([0, 0, 0])
        npt.assert_array_almost_equal(self.BE.asnumpy(B), C, decimal=6)

    def test_longitudinal_projection_ft(self):
        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 0, 10] = np.array([2, 3, 4])
        A = self.BE.astype(A)
        B = self.BE.longitudinal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 0, 10] = np.array([0, 0, 4])
        npt.assert_array_almost_equal(self.BE.asnumpy(B), C, decimal=6)

        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 0, 2] = np.array([2, 3, 4])
        A = self.BE.astype(A)
        B = self.BE.longitudinal_projection_ft(A)
        C = np.zeros([3, 1, *self.grid.shape])
        C[:, 0, 0, 0, 2] = np.array([0, 0, 4])
        npt.assert_array_almost_equal(self.BE.asnumpy(B), C)

        A = np.zeros([3, 1, *self.grid.shape])
        A[:, 0, 0, 5, 10] = np.array([4, 5, 6])
        A = self.BE.astype(A)
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

    def test_mat3_eigh(self):
        # This test ignores the sorting order of the eigenvalues
        D = np.diag([4, 5, -6])[:, :, np.newaxis, np.newaxis, np.newaxis]
        nI = 2.7*np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis]
        # A = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
        # H = A + self.PO.adjoint(A)
        def sort(arr):
            sh = arr.shape
            arr = sorted(sorted(arr.ravel(), key=np.imag), key=np.real)
            return np.asarray(arr).reshape(sh)
        D = self.BE.astype(D)
        npt.assert_almost_equal(sort(self.BE.asnumpy(self.BE.mat3_eigh(D))), np.array([-6, 4, 5]).reshape(3, 1, 1, 1),
                                decimal=5)

        Q = scipy.linalg.orth(np.arange(9).reshape([3, 3]) ** 2 + 1j * np.arange(9).reshape([3, 3]))
        QD = (Q @ np.diag([4, 5, -6]) @ np.conj(Q.transpose()))[:, :, np.newaxis, np.newaxis, np.newaxis]
        QD = self.BE.astype(QD)
        npt.assert_array_almost_equal(sort(self.BE.asnumpy(self.BE.mat3_eigh(QD))), np.array([-6, 4, 5]).reshape(3, 1, 1, 1),
                                      decimal=5)  # TODO: Can this be more accurate?

        nI = self.BE.astype(nI)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.mat3_eigh(nI)), 2.7*np.ones((3, 1, 1, 1)), decimal=1)  # TODO: We need an iterative algorithm to calculate eigenvalues.

        def rot_x_d(a):
            return 1j * np.array([[0, 0, 0], [0, 0, -a], [0, a, 0]])[:, :, np.newaxis, np.newaxis, np.newaxis]

        npt.assert_array_almost_equal(np.sort(self.BE.asnumpy(self.BE.mat3_eigh(self.BE.astype(rot_x_d(0.1)))[:, 0, 0, 0])),
                                      np.sort(np.linalg.eig(rot_x_d(0.1)[:, :, 0, 0, 0])[0], axis=0), decimal=6,
                                      err_msg='Three small real eigenvalues not detected correctly.')

        npt.assert_array_almost_equal(np.sort(self.BE.asnumpy(self.BE.mat3_eigh(self.BE.astype(rot_x_d(0.01)))[:, 0, 0, 0])),
                                      np.sort(np.linalg.eig(rot_x_d(0.01)[:, :, 0, 0, 0])[0], axis=0), decimal=6,
                                      err_msg='Three real eigenvalues around zero not detected correctly.')

    def test_cross(self):
        A = np.array([1, 2, 3])
        B = np.array([3, 4, 5])
        A = self.BE.astype(A)
        B = self.BE.astype(B)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.cross(A, B)), np.array([-2, 4, -2]))

        A = A.reshape([3, 1, 1])
        B = B.reshape([3, 1, 1])
        A = self.BE.astype(A)
        B = self.BE.astype(B)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.cross(A, B)), np.array([-2, 4, -2]).reshape(3, 1, 1))

    def test_outer(self):
        A = np.array([1, 2]).reshape([2, 1])
        B = np.array([3, 4, 5]).reshape([3, 1])
        A = self.BE.astype(A)
        B = self.BE.astype(B)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.outer(A, B)), np.array([[3, 4, 5], [6, 8, 10]]))

        A = A.reshape([2, 1, 1])
        B = B.reshape([3, 1, 1])
        A = self.BE.astype(A)
        B = self.BE.astype(B)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.outer(A, B)), np.array([[3, 4, 5], [6, 8, 10]]).reshape([2, 3, 1]))

    def test_div_ft(self):
        # TODO: Add more in depth tests
        V = np.zeros([3, 1, *self.grid.shape], dtype=float)
        V[:, 0, 0, 0, 0] = np.array([1.0, 2.0, 3.0])
        V = self.BE.astype(V)
        npt.assert_array_almost_equal(self.BE.asnumpy(self.BE.div_ft(V)), np.zeros([1, 1, *self.grid.shape]))

        M = np.zeros([3, 3, *self.grid.shape], dtype=float)
        M[:, :, 0, 0, 0] = np.arange(9).reshape([3, 3])
        M = self.BE.astype(M)
        div_ft_M = self.BE.div_ft(M)
        npt.assert_array_almost_equal(self.BE.asnumpy(div_ft_M), np.zeros([3, 1, *self.grid.shape]))

    def test_div(self):
        # TODO: Add more in depth tests
        vector_field = np.zeros([3, 1, *self.grid.shape], dtype=float)
        vector_field[0] = 1.0
        vector_field[1] = 2.0
        vector_field[2] = 3.0
        vector_field = self.BE.astype(vector_field)
        npt.assert_almost_equal(self.BE.asnumpy(self.BE.div(vector_field)), np.zeros([1, 1, *self.grid.shape]), decimal=6)

        M = np.tile(np.arange(9).reshape([3, 3, 1, 1, 1]), [1, 1, *self.grid.shape])
        M = self.BE.astype(M)
        div_M = self.BE.div(M)
        npt.assert_almost_equal(self.BE.asnumpy(div_M), np.zeros([3, 1, *self.grid.shape]), decimal=6)
