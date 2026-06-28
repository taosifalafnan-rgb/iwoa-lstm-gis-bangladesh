"""
seed.py — Global seed setter for full reproducibility.
Call set_seed(cfg.seed) at the start of every script.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + GPU).

    Args:
        seed: Integer seed value. Use cfg.seed from config.yaml.

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] Global seed set to {seed}")


if __name__ == "__main__":
    set_seed(42)
    print("Seed test passed.")
