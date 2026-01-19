from typing import Dict, Any

def format_epoch_metric_row(epoch: int, train_loss: float, val_dice: float, val_iou: float) -> str:
    """Formats training iteration record as RFC CSV row."""
    return f"{epoch},{train_loss:.5f},{val_dice:.4f},{val_iou:.4f}"
