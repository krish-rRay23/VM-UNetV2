import numpy as np

def apply_random_brightness_contrast(image: np.ndarray, alpha: float = 1.1, beta: float = 10.0) -> np.ndarray:
    """Applies linear brightness and contrast scaling for medical radiography."""
    adjusted = image.astype(float) * alpha + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)
