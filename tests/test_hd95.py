import unittest
import numpy as np
from hd95 import approximate_hd95

class TestHD95(unittest.TestCase):
    def test_identical_masks(self):
        m = np.zeros((32, 32))
        m[5:15, 5:15] = 1
        self.assertAlmostEqual(approximate_hd95(m, m), 0.0, places=2)

if __name__ == '__main__':
    unittest.main()
