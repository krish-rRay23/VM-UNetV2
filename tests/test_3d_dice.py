import unittest
import numpy as np

def aggregate_3d_dice(slice_dices):
    return float(np.mean(slice_dices))

class Test3DDice(unittest.TestCase):
    def test_mean(self):
        d = aggregate_3d_dice([0.9, 0.95, 0.85])
        self.assertAlmostEqual(d, 0.90)
