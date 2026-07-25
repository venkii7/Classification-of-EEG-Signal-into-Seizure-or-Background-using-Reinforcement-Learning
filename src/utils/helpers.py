"""
helpers.py — Utility functions for reproducibility, logging, and device management.
"""

import os
import random
import logging
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Return a torch device. Falls back to CPU if CUDA is unavailable.

    Args:
        preferred: "cuda" or "cpu"
    Returns:
        torch.device
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using device: CPU")
    return device


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Configure logging to console and file.

    Args:
        log_dir: Directory to store log files.
        level:   Logging level.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
    )


@contextmanager
def timer(description: str = "Operation"):
    """Context manager to time a block of code."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logging.info(f"{description} completed in {elapsed:.2f}s")
