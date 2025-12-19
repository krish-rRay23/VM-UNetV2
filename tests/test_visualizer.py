import unittest
import numpy as np
from visualizer import create_segmentation_overlay

class TestVisualizer(unittest.TestCase):
    def test_overlay_shape(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64))
        mask[10:30, 10:30] = 1
        res = create_segmentation_overlay(img, mask)
        self.assertEqual(res.shape, (64, 64, 3))
        self.assertEqual(res.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()
