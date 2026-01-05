import unittest
import numpy as np
from quantize_helper import quantize_weights_int8

class TestQuantize(unittest.TestCase):
    def test_quantization_dtype(self):
        w = np.random.randn(32, 32).astype(np.float32)
        q = quantize_weights_int8(w)
        self.assertEqual(q.dtype, np.int8)

if __name__ == '__main__':
    unittest.main()
