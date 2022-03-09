import unittest

from macromax.backend import BackEnd
from macromax.backend.tensorflow import BackEndTensorFlow

from tests.test_backend import BaseTestBackEnd

import numpy as np
import tensorflow as tf


class TestBackEndTensorFlow(BaseTestBackEnd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=np.complex128, hardware_dtype=tf.complex128, **kwargs)
        self.__backend = None

    @property
    def BE(self) -> BackEnd:
        if self.__backend is None:
            self.__backend = BackEndTensorFlow(self.nb_pol_dims, self.grid * self.wavenumber, hardware_dtype=self.hardware_dtype)
        return self.__backend

    def test_mat_inv(self):
        pass  # TODO: inverse not implemented with ComplexDouble on tensorflow *not critical*
    #     A = np.arange(9).reshape((3, 3, 1, 1, 1))**2
    #     A_inv = self.BE.ldivide(A, 1.0)
    #     npt.assert_almost_equal(self.BE.mul(A_inv, A), self.BE.eye)
    #     B = np.array([[1.0, 2j, 3], [-4j, 5, 6], [7, 8, 9]])[:, :, np.newaxis, np.newaxis, np.newaxis]
    #     B_inv = self.BE.ldivide(B, 1.0)
    #     npt.assert_almost_equal(self.BE.mul(B_inv, B), self.BE.eye)


if __name__ == '__main__':
    unittest.main()
