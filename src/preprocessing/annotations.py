"""
annotations.py — Parse TUH EEG Seizure Corpus annotation files.

TUH provides annotations in CSV files (both regular CSV and CSV_BI formats).
The _bi.csv format has columns:
    channel, start_time, stop_time, label, probability

We extract (start_time, stop_time, label) and map labels to integers.
"""

import os
import logging
from typing import List, Tuple

import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def parse_annotations(
    csv_path: str,
    label_map: dict = None,
) -> List[Tuple[float, float, int]]:
    """
    Parse a TUH CSV annotation file and return time-stamped labels.

    The function tries multiple CSV formats:
    1. _bi.csv format: channel, start_time, stop_time, label, probability
    2. Regular CSV:    start_time, stop_time, label, ...

    Args:
        csv_path:   Path to the annotation CSV file.
        label_map:  Mapping from string labels to integers.
                    Default: {"bckg": 0, "seiz": 1}

    Returns:
        List of (start_time, end_time, label_int) tuples, sorted by start_time.
    """
    if label_map is None:
        label_map = config.LABEL_MAP

    if not os.path.exists(csv_path):
        logging.warning(f"Annotation file not found: {csv_path}")
        return []

    try:
        # ── Try reading with different separators ─────────────────────────
        # TUH CSV files may use comma or space as separator
        for sep in [",", r"\s+"]:
            try:
                df = pd.read_csv(
                    csv_path,
                    sep=sep,
                    comment="#",
                    header=None,
                    engine="python",
                )
                if df.shape[1] >= 3:
                    break
            except Exception:
                continue
        else:
            logging.warning(f"Could not parse annotation file: {csv_path}")
            return []

        annotations = []

        if df.shape[1] >= 5:
            # _bi.csv format: channel, start, stop, label, probability
            for _, row in df.iterrows():
                try:
                    start = float(row.iloc[1])
                    stop = float(row.iloc[2])
                    label_str = str(row.iloc[3]).strip().lower()
                    if label_str in label_map:
                        annotations.append((start, stop, label_map[label_str]))
                except ValueError:
                    continue  # Skip header or malformed rows
        elif df.shape[1] >= 3:
            # Regular format: start, stop, label, ...
            for _, row in df.iterrows():
                try:
                    start = float(row.iloc[0])
                    stop = float(row.iloc[1])
                    label_str = str(row.iloc[2]).strip().lower()
                    if label_str in label_map:
                        annotations.append((start, stop, label_map[label_str]))
                except ValueError:
                    continue

        # Remove duplicates and sort by start time
        annotations = list(set(annotations))
        annotations.sort(key=lambda x: x[0])

        return annotations

    except Exception as e:
        logging.error(f"Error parsing annotations from {csv_path}: {e}")
        return []


def find_annotation_file(edf_path: str) -> str:
    """
    Given an EDF file path, find the corresponding annotation CSV file.

    TUH convention: annotation files are in a parallel directory structure
    or in the same directory with matching name and .csv / .csv_bi extension.

    Args:
        edf_path: Path to the .edf file.

    Returns:
        Path to the annotation file, or empty string if not found.
    """
    base = os.path.splitext(edf_path)[0]

    # Try common annotation file patterns
    candidates = [
        base + ".csv_bi",
        base + ".csv",
        base + "_bi.csv",
        # Also try replacing 'edf' directory with 'csv' or 'csv_bi'
        edf_path.replace(".edf", ".csv_bi"),
        edf_path.replace(".edf", ".csv"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Search in parent/sibling directories
    edf_dir = os.path.dirname(edf_path)
    edf_name = os.path.splitext(os.path.basename(edf_path))[0]

    for root, _, files in os.walk(edf_dir):
        for f in files:
            if f.startswith(edf_name) and f.endswith((".csv", ".csv_bi")):
                return os.path.join(root, f)

    logging.warning(f"No annotation file found for: {edf_path}")
    return ""
