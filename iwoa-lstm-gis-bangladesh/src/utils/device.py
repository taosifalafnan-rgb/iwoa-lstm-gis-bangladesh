"""
device.py — Auto GPU/CPU device detection.
Never hardcode 'cuda' or 'cpu' anywhere else in the codebase.
"""

import torch
from src.utils.logger import get_logger

log = get_logger(__name__)


def get_device() -> torch.device:
    """
    Auto-detect and return the best available compute device.

    Returns:
        device: torch.device — either 'cuda', 'mps' (Apple Silicon), or 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Device: CUDA GPU — {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        log.info("Device: CPU — consider using Google Colab for IWOA/LSTM training")

    return device


if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")
