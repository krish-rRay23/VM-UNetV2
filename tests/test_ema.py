import unittest
import numpy as np

class WeightEMA:
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}

    def update(self, name, param):
        if name not in self.shadow:
            self.shadow[name] = param.copy()
        else:
            self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param

class TestEMA(unittest.TestCase):
    def test_ema_step(self):
        ema = WeightEMA(decay=0.9)
        ema.update("w", np.array([10.0]))
        ema.update("w", np.array([0.0]))
        self.assertAlmostEqual(ema.shadow["w"][0], 9.0)
