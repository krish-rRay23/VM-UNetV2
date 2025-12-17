import unittest
from profiler import count_parameters_summary

class TestProfiler(unittest.TestCase):
    def test_param_counting(self):
        shapes = [(64, 3, 3, 3), (128, 64, 3, 3)]
        summary = count_parameters_summary(shapes)
        self.assertGreater(summary["total_parameters"], 0)
        self.assertTrue(summary["is_lightweight"])

if __name__ == '__main__':
    unittest.main()
