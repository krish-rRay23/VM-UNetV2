# Reproducibility Guidelines

To ensure deterministic evaluation across GPU platforms:
1. Fix seed globally across Python `random`, `numpy`, and PyTorch:
   ```python
   import random, os, numpy as np, torch
   random.seed(42)
   np.random.seed(42)
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```
2. Set `PYTHONHASHSEED=42` in execution environment.
