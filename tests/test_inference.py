import unittest
import numpy as np
from inference_runner import run_single_inference

class TestInference(unittest.TestCase):
    def test_single_inference_output(self):
        img = np.ones((64, 64)) * 200
        res = run_single_inference(img, threshold=0.5)
        self.assertEqual(res["mask"].shape, (64, 64))
        self.assertGreater(res["mean_confidence"], 0.7)

if __name__ == '__main__':
    unittest.main()
