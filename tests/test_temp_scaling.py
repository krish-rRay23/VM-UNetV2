import unittest
import numpy as np

def apply_temperature_scaling(logits, temperature=1.5):
    return logits / temperature

class TestTempScale(unittest.TestCase):
    def test_scale(self):
        logits = np.array([2.0, 4.0])
        scaled = apply_temperature_scaling(logits, 2.0)
        self.assertEqual(scaled[0], 1.0)
        self.assertEqual(scaled[1], 2.0)
