import unittest
import numpy as np

def compute_diag_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"sensitivity": sens, "specificity": spec}

class TestDiagMetrics(unittest.TestCase):
    def test_perfect(self):
        y = np.array([1, 0, 1, 0])
        m = compute_diag_metrics(y, y)
        self.assertEqual(m["sensitivity"], 1.0)
        self.assertEqual(m["specificity"], 1.0)
