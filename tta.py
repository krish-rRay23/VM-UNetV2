import numpy as np
from typing import List, Callable

def apply_horizontal_flip(x: np.ndarray) -> np.ndarray:
    return np.flip(x, axis=1)

def apply_vertical_flip(x: np.ndarray) -> np.ndarray:
    return np.flip(x, axis=0)

def ensemble_tta_predictions(model_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Computes average inference output across 4-fold dihedral augmentations (identity, h-flip, v-flip, hv-flip)."""
    p1 = model_fn(x)
    p2 = apply_horizontal_flip(model_fn(apply_horizontal_flip(x)))
    p3 = apply_vertical_flip(model_fn(apply_vertical_flip(x)))
    p4 = apply_vertical_flip(apply_horizontal_flip(model_fn(apply_horizontal_flip(apply_vertical_flip(x)))))
    return (p1 + p2 + p3 + p4) / 4.0
