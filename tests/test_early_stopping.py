import unittest
from early_stopping import EarlyStopping

class TestEarlyStopping(unittest.TestCase):
    def test_patience_trigger(self):
        es = EarlyStopping(patience=3, min_delta=0.01)
        es.step(0.50)
        es.step(0.51)
        es.step(0.52)
        should_stop = es.step(0.53)
        self.assertTrue(should_stop)

if __name__ == '__main__':
    unittest.main()
