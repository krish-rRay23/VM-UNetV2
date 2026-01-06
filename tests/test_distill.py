import unittest
import numpy as np
from distill_loss import compute_distillation_loss

class TestDistill(unittest.TestCase):
    def test_zero_loss_identical_feats(self):
        f = np.ones((8, 16, 16))
        loss = compute_distillation_loss(f, f)
        self.assertAlmostEqual(loss, 0.0)

if __name__ == '__main__':
    unittest.main()
