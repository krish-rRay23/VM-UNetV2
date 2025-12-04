import numpy as np

def calculate_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Computes Dice similarity coefficient between boolean/binary segmentation masks."""
    p_flat = pred.flatten() > 0.5
    t_flat = target.flatten() > 0.5
    intersection = np.logical_and(p_flat, t_flat).sum()
    return float((2.0 * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth))

def calculate_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Computes Intersection over Union (Jaccard Index) for binary masks."""
    p_flat = pred.flatten() > 0.5
    t_flat = target.flatten() > 0.5
    intersection = np.logical_and(p_flat, t_flat).sum()
    union = np.logical_or(p_flat, t_flat).sum()
    return float((intersection + smooth) / (union + smooth))
