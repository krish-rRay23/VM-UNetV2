import unittest
import numpy as np
from calibration import compute_ece

class TestCalibration(unittest.TestCase):
    def test_perfectly_calibrated(self):
        probs = np.array([0.9, 0.9, 0.1, 0.1])
        labels = np.array([1, 1, 0, 0])
        ece = compute_ece(probs, labels, n_bins=5)
        self.assertLess(ece, 0.2)

if __name__ == '__main__':
    unittest.main()
