import unittest
import numpy as np
from data_validator import validate_image_pair

class TestDataValidator(unittest.TestCase):
    def test_valid_pair(self):
        img = np.zeros((256, 256, 3))
        mask = np.zeros((256, 256))
        res = validate_image_pair(img, mask)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["errors"]), 0)

    def test_mismatched_pair(self):
        img = np.zeros((256, 256, 3))
        mask = np.zeros((128, 128))
        res = validate_image_pair(img, mask)
        self.assertFalse(res["valid"])

if __name__ == '__main__':
    unittest.main()
