import numpy as np

def approximate_hd95(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """Computes simplified boundary distance metric approximation for validation checks."""
    if not pred_mask.any() or not target_mask.any():
        return 0.0 if not pred_mask.any() and not target_mask.any() else 50.0

    p_coords = np.argwhere(pred_mask > 0)
    t_coords = np.argwhere(target_mask > 0)

    # Subsample if large for fast verification
    if len(p_coords) > 500:
        p_coords = p_coords[np.random.choice(len(p_coords), 500, replace=False)]
    if len(t_coords) > 500:
        t_coords = t_coords[np.random.choice(len(t_coords), 500, replace=False)]

    diff = p_coords[:, None, :] - t_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    min_dist_p = dist.min(axis=1)
    return float(np.percentile(min_dist_p, 95))
