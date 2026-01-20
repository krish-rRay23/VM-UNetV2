import unittest

def format_epoch_perf(epoch, imgs_per_sec, vram_gb):
    return f"Epoch {epoch:03d} | Throughput: {imgs_per_sec:.1f} img/s | VRAM: {vram_gb:.2f} GB"

class TestGPULogger(unittest.TestCase):
    def test_str(self):
        s = format_epoch_perf(1, 45.2, 7.84)
        self.assertIn("45.2 img/s", s)
