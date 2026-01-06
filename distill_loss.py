import numpy as np

def compute_distillation_loss(student_feats: np.ndarray, teacher_feats: np.ndarray) -> float:
    """Mean Squared Error alignment loss between student and teacher intermediate feature representations."""
    return float(np.mean((student_feats - teacher_feats) ** 2))
