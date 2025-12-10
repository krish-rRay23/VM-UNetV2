import unittest
import numpy as np
from tta import ensemble_tta_predictions

class TestTTA(unittest.TestCase):
    def test_identity_model_tta(self):
        x = np.ones((64, 64))
        avg = ensemble_tta_predictions(lambda img: img * 2.0, x)
        self.assertTrue(np.allclose(avg, x * 2.0))

if __name__ == '__main__':
    unittest.main()
