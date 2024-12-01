import unittest

from macromax.backend.torch import BackEndTorch

from tests.test_backend import BaseTestBackEnd

import numpy as np
import torch


class TestBackEndTorch(BaseTestBackEnd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=np.complex64, hardware_dtype=torch.complex64, **kwargs)
        self.BE = BackEndTorch(self.nb_pol_dims, self.grid * self.wavenumber, hardware_dtype=self.hardware_dtype)


if __name__ == '__main__':
    unittest.main()
