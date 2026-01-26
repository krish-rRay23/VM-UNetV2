from typing import Dict, Any
import numpy as np

def calculate_confusion_matrix(pred: np.ndarray, target: np.ndarray) -> Dict[str, int]:
    """Generates TP, FP, TN, FN pixel counts for segmentation analysis."""
    p = pred.flatten() > 0.5
    t = target.flatten() > 0.5
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
