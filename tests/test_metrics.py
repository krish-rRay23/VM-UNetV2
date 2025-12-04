import unittest
import numpy as np
from eval_metrics import calculate_dice, calculate_iou

class TestMetrics(unittest.TestCase):
    def test_perfect_match(self):
        m1 = np.ones((10, 10))
        m2 = np.ones((10, 10))
        self.assertAlmostEqual(calculate_dice(m1, m2), 1.0, places=4)
        self.assertAlmostEqual(calculate_iou(m1, m2), 1.0, places=4)

    def test_zero_overlap(self):
        m1 = np.zeros((10, 10))
        m1[:5, :] = 1
        m2 = np.zeros((10, 10))
        m2[5:, :] = 1
        self.assertAlmostEqual(calculate_dice(m1, m2), 0.0, places=3)
        self.assertAlmostEqual(calculate_iou(m1, m2), 0.0, places=3)

if __name__ == '__main__':
    unittest.main()
