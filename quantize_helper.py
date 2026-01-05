import numpy as np

def quantize_weights_int8(weights: np.ndarray) -> np.ndarray:
    """Quantizes float32 weights into int8 range [-128, 127] with symmetric scale."""
    max_val = np.max(np.abs(weights)) + 1e-8
    scale = 127.0 / max_val
    quantized = np.clip(np.round(weights * scale), -128, 127).astype(np.int8)
    return quantized
