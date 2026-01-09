import unittest
import numpy as np
from ensemble import soft_voting_ensemble

class TestEnsemble(unittest.TestCase):
    def test_ensemble_averaging(self):
        p1 = np.ones((10, 10)) * 0.4
        p2 = np.ones((10, 10)) * 0.8
        res = soft_voting_ensemble([p1, p2])
        self.assertTrue(np.allclose(res, 0.6))

if __name__ == '__main__':
    unittest.main()
