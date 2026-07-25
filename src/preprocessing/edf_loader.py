"""
edf_loader.py — Load EDF files from the TUH EEG Seizure Corpus.

Uses MNE-Python to:
  1. Read raw EDF data
  2. Select standard 10-20 channels
  3. Resample to a target frequency
  4. Apply bandpass filtering
  5. Return a NumPy array of shape (n_channels, n_samples)
"""

import logging
from typing import Optional, List, Tuple

import numpy as np
import mne

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_edf(
    edf_path: str,
    channels: Optional[List[str]] = None,
    target_sfreq: float = config.TARGET_SFREQ,
    bandpass_low: float = config.BANDPASS_LOW,
    bandpass_high: float = config.BANDPASS_HIGH,
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load and preprocess an EDF file.

    Args:
        edf_path:     Path to the .edf file.
        channels:     List of channel names to select. If None, uses config defaults.
        target_sfreq: Target sampling frequency after resampling.
        bandpass_low:  Low cutoff for bandpass filter (Hz).
        bandpass_high: High cutoff for bandpass filter (Hz).

    Returns:
        data:       NumPy array of shape (n_channels, n_samples), filtered & resampled.
        sfreq:      Sampling frequency of the returned data.
        ch_names:   List of channel names in the returned data.
    """
    if channels is None:
        channels = config.EEG_CHANNELS

    # ── Load raw EDF ──────────────────────────────────────────────────────
    # Suppress MNE logging noise
    mne.set_log_level("WARNING")

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        logging.error(f"Failed to load EDF file: {edf_path} — {e}")
        raise

    # ── Channel selection (Robust Matching) ───────────────────────────────
    # TUH files may have varied channel naming (e.g., -REF, -LE, -AVG)
    available = raw.ch_names
    selected = []
    
    # Helper to normalize channel names for matching
    def normalize_ch(name):
        # Remove 'EEG ' prefix, remove common reference suffixes
        n = name.upper().replace('EEG ', '').replace(' ', '')
        for suffix in ['-REF', '-LE', '-AVG', '-AR']:
            if n.endswith(suffix):
                n = n[:-len(suffix)]
        return n

    # Create map of normalized names -> original names in file
    norm_map = {normalize_ch(ch): ch for ch in available}
    
    # Try to find each target channel
    missing = []
    for target in channels:
        # 1. Exact match
        if target in available:
            selected.append(target)
            continue
            
        # 2. Normalized match
        norm_target = normalize_ch(target)
        if norm_target in norm_map:
            selected.append(norm_map[norm_target])
        else:
            missing.append(target)
            
    # Fallback only if critical failure
    if len(selected) < len(channels):
        logging.warning(
            f"Channel Mismatch in {os.path.basename(edf_path)}! "
            f"Found {len(selected)}/{len(channels)}.\n"
            f"Missing: {missing}\n"
            f"Available (first 10): {available[:10]}"
        )
        if len(selected) == 0:
             # Last resort: just take first N (risky but better than crashing)
             logging.error("CRITICAL: No matching channels found. Using first N channels (Data may be invalid).")
             selected = available[: min(len(channels), len(available))]

    # Reorder to match config order (crucial for CNN spatial filters)
    # We must pick them in the order of 'channels', not 'selected'
    # But 'selected' is already built in loop order of 'channels'.
    
    try:
        raw.pick_channels(selected, ordered=True)
    except Exception as e:
        logging.error(f"Error picking channels: {e}")
        raise

    # ── Resample ──────────────────────────────────────────────────────────
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)

    # ── Bandpass filter ───────────────────────────────────────────────────
    raw.filter(
        l_freq=bandpass_low,
        h_freq=bandpass_high,
        method="fir",
        verbose=False,
    )

    # ── Extract data ──────────────────────────────────────────────────────
    data = raw.get_data()  # shape: (n_channels, n_samples)

    # ── Z-score normalization per channel ─────────────────────────────────
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero
    data = (data - mean) / std

    return data.astype(np.float32), float(raw.info["sfreq"]), list(raw.ch_names)
