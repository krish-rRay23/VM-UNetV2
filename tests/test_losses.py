import unittest
import numpy as np
from losses import binary_dice_loss, combo_loss

class TestLosses(unittest.TestCase):
    def test_loss_bounds(self):
        target = np.array([1, 0, 1, 1], dtype=float)
        pred = np.array([0.9, 0.1, 0.8, 0.95], dtype=float)
        loss = combo_loss(pred, target)
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 1.0)

if __name__ == '__main__':
    unittest.main()
