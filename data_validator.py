from typing import Tuple, Dict, Any
import numpy as np

def validate_image_pair(image: np.ndarray, mask: np.ndarray, expected_shape: Tuple[int, int] = (256, 256)) -> Dict[str, Any]:
    """Validates medical image and ground-truth mask alignment and dimensions."""
    errors = []
    if image.shape[:2] != mask.shape[:2]:
        errors.append(f"Dimension mismatch: image {image.shape[:2]} vs mask {mask.shape[:2]}")
    if image.shape[:2] != expected_shape:
        errors.append(f"Unexpected resolution: expected {expected_shape}, got {image.shape[:2]}")
    if np.isnan(image).any() or np.isnan(mask).any():
        errors.append("NaN values detected in input tensors")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "mask_foreground_ratio": float(np.mean(mask > 0)) if mask.size > 0 else 0.0
    }
