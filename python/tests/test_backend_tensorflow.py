import unittest

from macromax.backend.tensorflow import BackEndTensorFlow

from tests.test_backend import BaseTestBackEnd

import numpy as np
import tensorflow as tf


class TestBackEndTensorFlow(BaseTestBackEnd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=np.complex128, hardware_dtype=tf.complex128, **kwargs)
        self.BE = BackEndTensorFlow(self.nb_pol_dims, self.grid * self.wavenumber, hardware_dtype=self.hardware_dtype)


if __name__ == '__main__':
    unittest.main()
