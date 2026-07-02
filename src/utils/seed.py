"""
seed.py — Global seed setter for full reproducibility.
Call set_seed(cfg.seed) at the start of every script.
"""

import os
import random
import numpy as np

# torch is only needed for the LSTM/IWOA stages. The GIS/HHI analysis stages
# must remain runnable without the heavy deep-learning dependency installed,
# so import it lazily and degrade gracefully if it is unavailable.
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _TORCH_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and (if installed) PyTorch.

    Args:
        seed: Integer seed value. Use cfg.seed from config.yaml.

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[seed] Global seed set to {seed}"
          f"{'' if _TORCH_AVAILABLE else ' (torch not installed — NumPy/Python only)'}")


if __name__ == "__main__":
    set_seed(42)
    print("Seed test passed.")
