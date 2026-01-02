from typing import Tuple, Dict, Any
import numpy as np

def run_single_inference(image: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Lightweight single-image prediction wrapper with confidence summary."""
    pred_prob = np.clip(image / 255.0, 0.0, 1.0)
    binary_mask = (pred_prob > threshold).astype(np.uint8)
    return {
        "mask": binary_mask,
        "foreground_pixels": int(binary_mask.sum()),
        "mean_confidence": float(pred_prob.mean())
    }
