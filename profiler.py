from typing import Dict, Any

def count_parameters_summary(layer_shapes: list) -> Dict[str, Any]:
    """Computes total parameter count and estimated memory footprint in MB."""
    total_params = sum(int(np.prod(s)) for s in layer_shapes) if layer_shapes else 0
    # 4 bytes per float32 parameter
    param_memory_mb = (total_params * 4) / (1024 * 1024)
    return {
        "total_parameters": total_params,
        "memory_mb": round(param_memory_mb, 2),
        "is_lightweight": total_params < 30_000_000 # Under 30M params for edge deployment
    }
import numpy as np
