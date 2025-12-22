import unittest
from checkpoint_manager import CheckpointTracker

class TestCheckpoint(unittest.TestCase):
    def test_best_tracking(self):
        tracker = CheckpointTracker(mode="max")
        self.assertTrue(tracker.update(0.85, epoch=1))
        self.assertFalse(tracker.update(0.82, epoch=2))
        self.assertTrue(tracker.update(0.89, epoch=3))
        self.assertEqual(tracker.best_epoch, 3)

if __name__ == '__main__':
    unittest.main()
