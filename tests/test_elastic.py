import unittest
import numpy as np
from elastic_aug import apply_random_brightness_contrast

class TestElasticAug(unittest.TestCase):
    def test_brightness_adjustment(self):
        img = np.ones((32, 32), dtype=np.uint8) * 100
        out = apply_random_brightness_contrast(img, alpha=1.2, beta=5)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(out[0, 0], 100)

if __name__ == '__main__':
    unittest.main()
