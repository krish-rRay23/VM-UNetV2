import unittest
from lr_scheduler import calculate_cosine_lr

class TestScheduler(unittest.TestCase):
    def test_start_and_end_lr(self):
        base_lr = 1e-3
        min_lr = 1e-6
        self.assertAlmostEqual(calculate_cosine_lr(0, 100, base_lr, min_lr), base_lr)
        self.assertAlmostEqual(calculate_cosine_lr(100, 100, base_lr, min_lr), min_lr)

if __name__ == '__main__':
    unittest.main()
