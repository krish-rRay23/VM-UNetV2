import numpy as np

def binary_dice_loss(pred_probs: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Computes soft Dice loss for probabilistic prediction maps."""
    intersection = (pred_probs * target).sum()
    dice = (2.0 * intersection + smooth) / (pred_probs.sum() + target.sum() + smooth)
    return float(1.0 - dice)

def combo_loss(pred_probs: np.ndarray, target: np.ndarray, alpha: float = 0.5) -> float:
    """Weighted combination of Dice loss and binary cross entropy."""
    d_loss = binary_dice_loss(pred_probs, target)
    eps = 1e-7
    p_clipped = np.clip(pred_probs, eps, 1.0 - eps)
    bce = -np.mean(target * np.log(p_clipped) + (1.0 - target) * np.log(1.0 - p_clipped))
    return float(alpha * d_loss + (1.0 - alpha) * bce)
