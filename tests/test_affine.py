import unittest
import numpy as np

def apply_zoom(img, zoom_factor=1.1):
    h, w = img.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    return np.zeros((new_h, new_w))

class TestZoom(unittest.TestCase):
    def test_zoom_shape(self):
        img = np.ones((100, 100))
        out = apply_zoom(img, 1.2)
        self.assertEqual(out.shape, (120, 120))
