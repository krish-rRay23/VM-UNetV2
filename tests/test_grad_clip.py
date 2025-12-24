import unittest
import numpy as np

def clip_gradients(grads, max_norm=1.0):
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        return [g * scale for g in grads], total_norm
    return grads, total_norm

class TestGradClip(unittest.TestCase):
    def test_clipping(self):
        g = [np.array([3.0, 4.0])]
        clipped, norm = clip_gradients(g, max_norm=2.5)
        self.assertAlmostEqual(norm, 5.0)
        new_norm = np.sqrt(np.sum(clipped[0] ** 2))
        self.assertAlmostEqual(new_norm, 2.5)
