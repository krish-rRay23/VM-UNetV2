import numpy as np

def create_segmentation_overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.4) -> np.ndarray:
    """Overlays binary segmentation mask over RGB image with transparency alpha."""
    overlay = image_rgb.copy().astype(float)
    color_mask = np.zeros_like(overlay)
    for c in range(3):
        color_mask[:, :, c] = (mask > 0.5) * color[c]

    mask_bool = mask > 0.5
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_mask[mask_bool]
    return np.clip(overlay, 0, 255).astype(np.uint8)
