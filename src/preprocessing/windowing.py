"""
windowing.py — Sliding-window segmentation and label alignment for EEG data.

Segments a continuous multi-channel EEG signal into fixed-length overlapping
windows and assigns a label to each window based on the annotation overlap
(majority-vote strategy).
"""

import logging
from typing import List, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def create_windows(
    data: np.ndarray,
    sfreq: float,
    annotations: List[Tuple[float, float, int]],
    window_sec: float = config.WINDOW_SEC,
    stride_sec: float = config.STRIDE_SEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment EEG data into overlapping windows and align labels.

    For each window, label assignment works by majority vote:
      - Compute what fraction of the window overlaps with seizure annotations
      - If ≥50% overlap → label = 1 (seizure), else label = 0 (background)

    Args:
        data:        EEG data, shape (n_channels, n_total_samples).
        sfreq:       Sampling frequency (Hz).
        annotations: List of (start_sec, end_sec, label_int) tuples.
        window_sec:  Window length in seconds.
        stride_sec:  Stride in seconds.

    Returns:
        windows: np.ndarray of shape (n_windows, n_channels, window_samples)
        labels:  np.ndarray of shape (n_windows,) with integer labels
    """
    n_channels, n_total = data.shape
    window_samples = int(window_sec * sfreq)
    stride_samples = int(stride_sec * sfreq)

    if n_total < window_samples:
        logging.warning(
            f"Signal length ({n_total}) < window size ({window_samples}). "
            "Padding with zeros."
        )
        pad = np.zeros((n_channels, window_samples - n_total), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=1)
        n_total = data.shape[1]

    # ── Compute window start positions ────────────────────────────────────
    starts = list(range(0, n_total - window_samples + 1, stride_samples))
    n_windows = len(starts)

    if n_windows == 0:
        return np.empty((0, n_channels, window_samples), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)

    windows = np.empty((n_windows, n_channels, window_samples), dtype=np.float32)
    labels = np.zeros(n_windows, dtype=np.int64)

    # ── Pre-compute seizure intervals in sample units ─────────────────────
    seiz_intervals = []
    for start_sec, end_sec, label in annotations:
        if label == 1:  # seizure
            s_start = int(start_sec * sfreq)
            s_end = int(end_sec * sfreq)
            seiz_intervals.append((s_start, s_end))

    # ── Extract windows and assign labels ─────────────────────────────────
    for i, win_start in enumerate(starts):
        win_end = win_start + window_samples
        windows[i] = data[:, win_start:win_end]

        # Majority-vote label assignment
        if seiz_intervals:
            seiz_overlap = 0
            for s_start, s_end in seiz_intervals:
                overlap_start = max(win_start, s_start)
                overlap_end = min(win_end, s_end)
                if overlap_end > overlap_start:
                    seiz_overlap += overlap_end - overlap_start

            # If ≥50% of window is seizure, label as seizure
            if seiz_overlap >= window_samples * 0.5:
                labels[i] = 1

    n_seiz = int(labels.sum())
    n_bckg = n_windows - n_seiz
    logging.debug(
        f"Created {n_windows} windows: {n_seiz} seizure, {n_bckg} background "
        f"(ratio: {n_seiz / max(n_windows, 1):.4f})"
    )

    return windows, labels
