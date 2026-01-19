import unittest
from logger_helper import format_epoch_metric_row

class TestLogger(unittest.TestCase):
    def test_metric_format(self):
        row = format_epoch_metric_row(10, 0.123456, 0.91234, 0.84567)
        self.assertEqual(row, "10,0.12346,0.9123,0.8457")

if __name__ == '__main__':
    unittest.main()
