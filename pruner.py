import numpy as np

def prune_by_magnitude(weights: np.ndarray, prune_ratio: float = 0.2) -> np.ndarray:
    """Zeros out the lowest magnitude weights by absolute threshold."""
    if prune_ratio <= 0.0:
        return weights.copy()
    threshold = np.percentile(np.abs(weights), prune_ratio * 100)
    pruned = weights.copy()
    pruned[np.abs(pruned) < threshold] = 0.0
    return pruned
