import unittest
import numpy as np
from pruner import prune_by_magnitude

class TestPruner(unittest.TestCase):
    def test_pruning_ratio(self):
        w = np.linspace(-10, 10, 100)
        p = prune_by_magnitude(w, prune_ratio=0.3)
        zero_count = np.sum(p == 0.0)
        self.assertGreaterEqual(zero_count, 28)

if __name__ == '__main__':
    unittest.main()
