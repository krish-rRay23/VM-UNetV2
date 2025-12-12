import unittest
import numpy as np

def validate_normalized_bounds(arr, min_val=0.0, max_val=1.0):
    return bool(np.all(arr >= min_val - 1e-5) and np.all(arr <= max_val + 1e-5))

class TestNorm(unittest.TestCase):
    def test_bounds(self):
        data = np.array([0.0, 0.5, 1.0])
        self.assertTrue(validate_normalized_bounds(data))
