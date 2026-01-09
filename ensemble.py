import numpy as np
from typing import List

def soft_voting_ensemble(prob_maps: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
    """Aggregates multiple model prediction probability maps using weighted soft voting."""
    if not prob_maps:
        raise ValueError("At least one probability map required")
    if weights is None:
        weights = [1.0 / len(prob_maps)] * len(prob_maps)

    norm_weights = np.array(weights) / sum(weights)
    ensemble = np.zeros_like(prob_maps[0], dtype=float)
    for p, w in zip(prob_maps, norm_weights):
        ensemble += p * w
    return ensemble
