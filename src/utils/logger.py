"""
logger.py — Dual logger: console + file
All pipeline steps use this. Never use bare print() in source files.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_file: str = "outputs/results/run_log.txt",
               level: str = "INFO") -> logging.Logger:
    """
    Create a logger that writes to both console and log file.

    Args:
        name:     Module name — use __name__ when calling.
        log_file: Path to log file. Created if it doesn't exist.
        level:    Logging level string: DEBUG, INFO, WARNING, ERROR.

    Returns:
        logger: Configured logging.Logger instance.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger  # Already configured

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Logger initialized successfully.")
    log.debug("Debug message — only visible at DEBUG level.")
    log.warning("Warning message test.")
