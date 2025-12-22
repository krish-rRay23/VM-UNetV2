from typing import Dict, Any

class CheckpointTracker:
    def __init__(self, mode: str = "max"):
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1

    def update(self, current_metric: float, epoch: int) -> bool:
        """Returns True if current metric improves over previous best."""
        is_best = False
        if self.mode == "max" and current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = epoch
            is_best = True
        elif self.mode == "min" and current_metric < self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = epoch
            is_best = True
        return is_best
