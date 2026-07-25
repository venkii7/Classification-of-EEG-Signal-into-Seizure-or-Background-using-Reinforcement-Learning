"""
preprocess.py — Orchestrate the full preprocessing pipeline.

Walks the TUH directory tree, discovers EDF + annotation pairs, processes
them through the pipeline (load → filter → window → label), and saves
the results as cached .npz files for fast reuse.
"""

import os
import glob
import logging
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from .edf_loader import load_edf
from .annotations import parse_annotations, find_annotation_file
from .windowing import create_windows


def discover_edf_files(data_dir: str) -> List[str]:
    """
    Recursively discover all EDF files under a directory.

    Args:
        data_dir: Root directory to search (e.g., data/edf/train).

    Returns:
        Sorted list of absolute paths to .edf files.
    """
    pattern = os.path.join(data_dir, "**", "*.edf")
    files = glob.glob(pattern, recursive=True)
    files.sort()
    logging.info(f"Found {len(files)} EDF files in {data_dir}")
    return files


def process_single_file(
    edf_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process a single EDF file: load → filter → window → label.

    Args:
        edf_path: Path to the .edf file.

    Returns:
        (windows, labels) or None if processing fails.
        windows: shape (n_windows, n_channels, window_samples)
        labels:  shape (n_windows,)
    """
    try:
        # Load and preprocess the EDF
        data, sfreq, ch_names = load_edf(edf_path)

        # Find and parse annotations
        ann_path = find_annotation_file(edf_path)
        if not ann_path:
            logging.warning(f"Skipping {edf_path}: no annotation file found")
            return None

        annotations = parse_annotations(ann_path)
        if len(annotations) == 0:
            logging.warning(f"Skipping {edf_path}: no valid annotations")
            return None

        # Create sliding windows with aligned labels
        windows, labels = create_windows(data, sfreq, annotations)

        if windows.shape[0] == 0:
            logging.warning(f"Skipping {edf_path}: no windows created")
            return None

        return windows, labels

    except Exception as e:
        logging.error(f"Error processing {edf_path}: {e}")
        return None


def preprocess_split(
    split_dir: str,
    cache_path: Optional[str] = None,
    max_files: Optional[int] = None,
    skip_files: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess an entire data split (train/dev/eval).

    If a cache file exists, loads from cache. Otherwise, processes all EDF files
    and saves to cache.

    Args:
        split_dir:  Directory for the split (e.g. data/edf/train).
        cache_path: Path to save/load cached .npz file. If None, auto-generated.
        max_files:  Maximum number of EDF files to process (for debugging).
        skip_files: Number of EDF files to skip from the beginning (for incremental training).

    Returns:
        all_windows: shape (N, n_channels, window_samples)
        all_labels:  shape (N,)
    """
    # Auto-generate cache path
    if cache_path is None:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        split_name = os.path.basename(split_dir.rstrip("/\\"))
        cache_path = os.path.join(config.CACHE_DIR, f"{split_name}_data.npz")

    # Load from cache if exists
    if os.path.exists(cache_path):
        logging.info(f"Loading cached data from {cache_path}")
        cached = np.load(cache_path)
        return cached["windows"], cached["labels"]

    # Discover and process EDF files
    edf_files = discover_edf_files(split_dir)
    if skip_files > 0:
        logging.info(f"Skipping first {skip_files} files")
        edf_files = edf_files[skip_files:]
    if max_files is not None:
        edf_files = edf_files[:max_files]

    all_windows = []
    all_labels = []

    skipped = 0
    for edf_path in tqdm(edf_files, desc=f"Processing {os.path.basename(split_dir)}"):
        result = process_single_file(edf_path)
        if result is not None:
            windows, labels = result
            # Validate channel count matches expected config
            if windows.shape[1] != config.NUM_CHANNELS:
                logging.warning(
                    f"Skipping {os.path.basename(edf_path)}: "
                    f"has {windows.shape[1]} channels, expected {config.NUM_CHANNELS}"
                )
                skipped += 1
                continue
            all_windows.append(windows)
            all_labels.append(labels)
    
    if skipped > 0:
        logging.info(f"Skipped {skipped} files due to channel count mismatch")

    if len(all_windows) == 0:
        logging.error(f"No data processed from {split_dir}")
        return np.empty((0, config.NUM_CHANNELS, config.WINDOW_SAMPLES), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Save to cache
    logging.info(f"Saving cache: {all_windows.shape[0]} windows -> {cache_path}")
    np.savez(cache_path, windows=all_windows, labels=all_labels)

    # Log class distribution
    n_seiz = int(all_labels.sum())
    n_bckg = len(all_labels) - n_seiz
    logging.info(
        f"Split '{os.path.basename(split_dir)}': "
        f"{all_windows.shape[0]} windows, "
        f"seizure={n_seiz} ({100 * n_seiz / len(all_labels):.1f}%), "
        f"background={n_bckg} ({100 * n_bckg / len(all_labels):.1f}%)"
    )

    return all_windows, all_labels


def preprocess_all(
    data_dir: str = config.DATA_DIR,
    max_files: Optional[int] = None,
) -> dict:
    """
    Preprocess all splits (train, dev, eval).

    Args:
        data_dir:   Root data directory containing edf/train, edf/dev, edf/eval.
        max_files:  Max files per split (for debugging).

    Returns:
        Dictionary: {"train": (windows, labels), "dev": ..., "eval": ...}
    """
    splits = {}
    for split_name in ["train", "dev", "eval"]:
        split_dir = os.path.join(data_dir, "edf", split_name)
        if os.path.isdir(split_dir):
            splits[split_name] = preprocess_split(split_dir, max_files=max_files)
        else:
            logging.warning(f"Split directory not found: {split_dir}")
    return splits


def generate_synthetic_data(
    n_files: int = 5,
    n_channels: int = config.NUM_CHANNELS,
    duration_sec: float = 60.0,
    sfreq: float = config.TARGET_SFREQ,
    seizure_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG data for dry-run testing.

    Creates random EEG-like data with random seizure segments.

    Args:
        n_files:       Number of synthetic "recordings" to generate.
        n_channels:    Number of EEG channels.
        duration_sec:  Duration of each recording in seconds.
        sfreq:         Sampling frequency.
        seizure_ratio: Fraction of each recording that is seizure.

    Returns:
        windows: shape (N, n_channels, window_samples)
        labels:  shape (N,)
    """
    all_windows = []
    all_labels = []
    window_samples = int(config.WINDOW_SEC * sfreq)
    stride_samples = int(config.STRIDE_SEC * sfreq)
    n_samples = int(duration_sec * sfreq)

    for _ in range(n_files):
        # Generate random signal (z-score normalized already by construction)
        data = np.random.randn(n_channels, n_samples).astype(np.float32)

        # Create random seizure annotations
        annotations = []
        # Place a seizure segment of `seizure_ratio` fraction
        seiz_len = int(duration_sec * seizure_ratio)
        seiz_start = np.random.randint(0, int(duration_sec) - seiz_len)
        annotations.append((float(seiz_start), float(seiz_start + seiz_len), 1))

        # Create windows
        windows, labels = create_windows(data, sfreq, annotations)
        all_windows.append(windows)
        all_labels.append(labels)

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    logging.info(
        f"Generated synthetic data: {all_windows.shape[0]} windows, "
        f"seizure={int(all_labels.sum())}, "
        f"background={len(all_labels) - int(all_labels.sum())}"
    )

    return all_windows, all_labels
