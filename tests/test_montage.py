import unittest
import numpy as np

def make_montage(slices):
    n = len(slices)
    return np.concatenate(slices, axis=1)

class TestMontage(unittest.TestCase):
    def test_concat(self):
        s1 = np.ones((32, 32))
        s2 = np.ones((32, 32))
        m = make_montage([s1, s2])
        self.assertEqual(m.shape, (32, 64))
