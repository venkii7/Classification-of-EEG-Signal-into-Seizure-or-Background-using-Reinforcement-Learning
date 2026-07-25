"""
TUSZ v2.0.3 - Seizure File Finder
==================================
Scans all .csv_bi files in the /train directory and identifies
files that contain at least one 'seiz' label.

Dataset structure:
    edf/train/<patient_id>/<session_id_date>/<montage_file>.csv_bi

Usage:
    python find_seiz_files.py --root /path/to/edf/train
    python find_seiz_files.py --root /path/to/edf/train --output results.csv
"""

import os
import csv
import argparse
from collections import defaultdict
from datetime import datetime


# ─────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────

def find_seiz_files(train_root: str) -> list[dict]:
    """
    Recursively walk the train directory, parse every .csv_bi file,
    and return a list of records for files that contain 'seiz' labels.

    Each record contains:
        - patient_id     : top-level subfolder under train/
        - session_id     : second-level subfolder (session + date)
        - montage_folder : third-level subfolder (montage name)
        - filename       : the .csv_bi filename
        - full_path      : absolute path to the file
        - folder_path    : directory containing the file
        - seiz_count     : number of seiz-labelled rows in the file
        - total_seiz_dur : total seizure duration in seconds
    """

    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train root not found: '{train_root}'")

    results = []

    for dirpath, dirnames, filenames in os.walk(train_root):
        dirnames.sort()   # deterministic traversal order

        csv_bi_files = sorted(f for f in filenames if f.endswith(".csv_bi"))
        if not csv_bi_files:
            continue

        # ── Derive dataset path components ──────────────────────────────
        rel_dir = os.path.relpath(dirpath, train_root)
        parts   = rel_dir.split(os.sep)

        patient_id     = parts[0] if len(parts) > 0 else "unknown"
        session_id     = parts[1] if len(parts) > 1 else "unknown"
        montage_folder = parts[2] if len(parts) > 2 else "unknown"

        for fname in csv_bi_files:
            fpath = os.path.join(dirpath, fname)

            seiz_rows      = []
            total_seiz_dur = 0.0
            has_seiz       = False

            try:
                with open(fpath, "r") as fh:
                    for line in fh:
                        line = line.strip()

                        # Skip header / comment lines
                        if not line or line.startswith("#") or line.startswith("version"):
                            continue

                        # csv_bi format:
                        #   channel, start_time, stop_time, label, confidence
                        cols = [c.strip() for c in line.split(",")]
                        if len(cols) < 4:
                            continue

                        label = cols[3].strip().lower()

                        if label == "seiz":
                            has_seiz = True
                            try:
                                start = float(cols[1])
                                stop  = float(cols[2])
                                dur   = stop - start
                                total_seiz_dur += dur
                                seiz_rows.append({
                                    "start": start,
                                    "stop":  stop,
                                    "dur":   dur,
                                })
                            except ValueError:
                                pass  # malformed time — still flag the file

            except (OSError, UnicodeDecodeError) as e:
                print(f"  [WARNING] Could not read '{fpath}': {e}")
                continue

            if has_seiz:
                results.append({
                    "patient_id":     patient_id,
                    "session_id":     session_id,
                    "montage_folder": montage_folder,
                    "filename":       fname,
                    "folder_path":    dirpath,
                    "full_path":      fpath,
                    "seiz_count":     len(seiz_rows),
                    "total_seiz_dur": round(total_seiz_dur, 3),
                })

    return results


# ─────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────

def print_summary(results: list[dict], train_root: str) -> None:
    """Pretty-print results to stdout."""

    sep = "─" * 90

    print(f"\n{'═' * 90}")
    print(f"  TUSZ  ▸  Seizure File Finder")
    print(f"  Train root : {train_root}")
    print(f"  Scanned at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 90}\n")

    if not results:
        print("  ✗  No files with 'seiz' labels were found.\n")
        return

    # ── Per-file table ──────────────────────────────────────────────────
    print(f"  {'#':<6} {'Patient':<14} {'Session':<28} {'Montage':<18} {'File':<40} {'Seiz Events':>11} {'Seiz Dur(s)':>12}")
    print(f"  {sep}")

    patient_set  = set()
    session_set  = set()
    total_events = 0
    total_dur    = 0.0

    for idx, r in enumerate(results, 1):
        patient_set.add(r["patient_id"])
        session_set.add((r["patient_id"], r["session_id"]))
        total_events += r["seiz_count"]
        total_dur    += r["total_seiz_dur"]

        print(
            f"  {idx:<6} "
            f"{r['patient_id']:<14} "
            f"{r['session_id']:<28} "
            f"{r['montage_folder']:<18} "
            f"{r['filename']:<40} "
            f"{r['seiz_count']:>11} "
            f"{r['total_seiz_dur']:>12.2f}"
        )

    # ── Aggregate summary ───────────────────────────────────────────────
    print(f"\n  {sep}")
    print(f"  SUMMARY")
    print(f"  {sep}")
    print(f"  {'Files with seiz labels':<35}: {len(results)}")
    print(f"  {'Unique patients':<35}: {len(patient_set)}")
    print(f"  {'Unique sessions':<35}: {len(session_set)}")
    print(f"  {'Total seizure events':<35}: {total_events}")
    print(f"  {'Total seizure duration':<35}: {total_dur:.2f} seconds  ({total_dur/3600:.2f} hours)")
    print(f"\n{'═' * 90}\n")


def save_csv(results: list[dict], output_path: str) -> None:
    """Save results to a CSV file."""
    fieldnames = [
        "patient_id", "session_id", "montage_folder",
        "filename", "folder_path", "full_path",
        "seiz_count", "total_seiz_dur"
    ]
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  ✔  Results saved to: {output_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find all .csv_bi files in TUSZ /train that contain 'seiz' labels."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./edf/train",
        help="Path to the train root directory (default: ./edf/train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save results to this CSV file path"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n  Scanning: {args.root} ...")
    results = find_seiz_files(train_root=args.root)

    print_summary(results, train_root=args.root)

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()