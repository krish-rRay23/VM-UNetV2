import unittest
import numpy as np

def attention_transfer_loss(teacher_maps, student_maps):
    t_norm = teacher_maps / (np.linalg.norm(teacher_maps) + 1e-6)
    s_norm = student_maps / (np.linalg.norm(student_maps) + 1e-6)
    return float(np.mean((t_norm - s_norm) ** 2))

class TestATLoss(unittest.TestCase):
    def test_identical(self):
        m = np.ones((4, 4))
        loss = attention_transfer_loss(m, m)
        self.assertAlmostEqual(loss, 0.0)
