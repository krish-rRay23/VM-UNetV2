import unittest
import numpy as np

def check_boundary_threshold(dist_matrix, max_allowed_mm=5.0):
    return bool(np.max(dist_matrix) <= max_allowed_mm)

class TestSurfaceDist(unittest.TestCase):
    def test_thresh(self):
        m = np.array([1.2, 3.4, 4.8])
        self.assertTrue(check_boundary_threshold(m, 5.0))
