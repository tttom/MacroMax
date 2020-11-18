import unittest

from macromax.backend import BackEnd
from macromax.backend.torch import BackEndTorch

from tests.test_backend_numpy import TestBackEnd

import numpy as np
import torch


class TestBackEndTorch(TestBackEnd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=np.complex64, internal_dtype=torch.complex64, **kwargs)
        self.__backend = None

    @property
    def BE(self) -> BackEnd:
        if self.__backend is None:
            self.__backend = BackEndTorch(self.nb_pol_dims, self.grid * self.wavenumber, dtype=self.internal_dtype)
        return self.__backend

    def test_calc_complex_roots(self):
        pass
        # TODO: Complex sorting not implemented on PyTorch
        # C = np.array([-1, 0, 0, 1])  # x**3 == 1
        # roots = self.BE.calc_roots_of_low_order_polynomial(C)
        # npt.assert_array_almost_equal(np.sort(roots), [np.exp(-2j*np.pi/3), np.exp(2j*np.pi/3), 1],
        #                               err_msg='failed finding cubic roots centered around zero.')
        #
        # C = np.array([1, 2, 3, -4])
        # roots = self.BE.calc_roots_of_low_order_polynomial(C)
        # for root_idx in range(3):
        #     npt.assert_almost_equal(self.BE.asnumpy(self.BE.evaluate_polynomial(C, roots[root_idx])), 0.0,
        #                             err_msg='third order polynomial root %d not found' % root_idx)

    def test_mat_inv(self):
        pass  # TODO: inverse not implemented with ComplexDouble on pytorch *not critical*
    #     A = np.arange(9).reshape((3, 3, 1, 1, 1))**2
    #     A_inv = self.BE.ldivide(A, 1.0)
    #     npt.assert_almost_equal(self.BE.mul(A_inv, A), self.BE.eye)
    #     B = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
    #     B_inv = self.BE.ldivide(B, 1.0)
    #     npt.assert_almost_equal(self.BE.mul(B_inv, B), self.BE.eye)


if __name__ == '__main__':
    unittest.main()
