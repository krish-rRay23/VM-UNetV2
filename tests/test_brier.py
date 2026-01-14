import unittest
import numpy as np

def compute_brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))

class TestBrier(unittest.TestCase):
    def test_brier(self):
        y = np.array([1, 0])
        p = np.array([1.0, 0.0])
        self.assertEqual(compute_brier_score(y, p), 0.0)
