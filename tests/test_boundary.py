import unittest
import numpy as np
from boundary_analysis import calculate_confusion_matrix

class TestBoundary(unittest.TestCase):
    def test_confusion_matrix(self):
        p = np.array([1, 1, 0, 0])
        t = np.array([1, 0, 1, 0])
        cm = calculate_confusion_matrix(p, t)
        self.assertEqual(cm["TP"], 1)
        self.assertEqual(cm["FP"], 1)
        self.assertEqual(cm["FN"], 1)
        self.assertEqual(cm["TN"], 1)

if __name__ == '__main__':
    unittest.main()
