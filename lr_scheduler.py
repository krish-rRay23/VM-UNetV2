import math

def calculate_cosine_lr(current_epoch: int, total_epochs: int, base_lr: float, min_lr: float = 1e-6) -> float:
    """Calculates cosine annealed learning rate for a given epoch."""
    if current_epoch >= total_epochs:
        return min_lr
    ratio = current_epoch / float(total_epochs)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + (base_lr - min_lr) * cosine_decay
