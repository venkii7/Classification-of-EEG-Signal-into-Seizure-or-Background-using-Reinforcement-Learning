"""
evaluate.py — Evaluation pipeline for RL-based EEG seizure detection.

Metrics computed:
  - Sensitivity (Recall for seizure class) = TP / (TP + FN)
  - Specificity (Recall for background class) = TN / (TN + FP)
  - F1-Score (for seizure class)
  - False Alarm Rate = FP / (FP + TN)
  - Accuracy
  - AUC-ROC (if probabilities available)
  - Confusion matrix
"""

import logging
from typing import Dict, Optional, Union, List

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def compute_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics for seizure detection."""
    if len(true_labels) == 0:
        return {
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "false_alarm_rate": 0.0,
        }

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        if true_labels[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn = fp = fn = tp = 0

    accuracy = accuracy_score(true_labels, predictions)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(true_labels, predictions, pos_label=1, zero_division=0)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "false_alarm_rate": false_alarm_rate,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": len(true_labels),
        "total_seizure": int(np.sum(true_labels == 1)),
        "total_background": int(np.sum(true_labels == 0)),
    }
    return metrics


def evaluate_agent(
    agent,
    windows: np.ndarray,
    labels: np.ndarray,
    agent_type: str = "dqn",
    device: Optional[torch.device] = None,
) -> Dict:
    """Evaluate a trained RL agent on a dataset."""
    if device is None:
        device = torch.device("cpu")

    predictions = []

    for i in range(len(windows)):
        state = windows[i]
        if agent_type == "dqn" or agent_type == "cnn":
            action = agent.select_action(state, training=False)
        elif agent_type == "ppo":
            action, _, _ = agent.select_action(state, training=False)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")
        predictions.append(action)

    predictions = np.array(predictions)
    metrics = compute_metrics(labels, predictions)
    metrics["predictions"] = predictions  # Return raw preds for plotting

    return metrics


def print_evaluation_report(metrics: Dict[str, float], split_name: str = "Eval"):
    """Print a formatted evaluation report."""
    print(f"\n{'═' * 60}")
    print(f"  EVALUATION REPORT — {split_name.upper()} SET")
    print(f"{'═' * 60}")
    print(f"  Total Samples:    {metrics.get('total_samples', 'N/A')}")
    print(f"  Total Seizure:    {metrics.get('total_seizure', 'N/A')}")
    print(f"  Total Background: {metrics.get('total_background', 'N/A')}")
    print(f"{'─' * 60}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Sensitivity:      {metrics['sensitivity']:.4f}  (Seizure Recall)")
    print(f"  Specificity:      {metrics['specificity']:.4f}  (Background Recall)")
    print(f"  Precision:        {metrics['precision']:.4f}  (Seizure Precision)")
    print(f"  F1-Score:         {metrics['f1_score']:.4f}  (Seizure F1)")
    print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
    print(f"{'─' * 60}")
    print(f"  Confusion Matrix:")
    print(f"    TP={metrics.get('true_positives', 'N/A'):>6}  "
          f"FP={metrics.get('false_positives', 'N/A'):>6}")
    print(f"    FN={metrics.get('false_negatives', 'N/A'):>6}  "
          f"TN={metrics.get('true_negatives', 'N/A'):>6}")
    print(f"{'═' * 60}\n")


def plot_eeg_overlay(
    windows: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
    edf_path: Optional[str] = None,
):
    """
    Generate a clinical-style EEG visualization with stacked channels and prediction overlay.
    """
    # Fallback if no EDF path provided
    if edf_path is None or not os.path.exists(edf_path):
        _plot_simple_overlay(windows, labels, predictions, save_path)
        return

    try:
        import mne
    except ImportError:
        logging.warning("MNE not installed. Falling back to simple plot.")
        _plot_simple_overlay(windows, labels, predictions, save_path)
        return

    # 1. Load Data
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        logging.error(f"Failed to load EDF for plotting: {e}")
        _plot_simple_overlay(windows, labels, predictions, save_path)
        return

    # Select channels
    available_channels = raw.ch_names
    # Try to find standard channels
    picks = []
    plot_channels = []
    
    # Clean channel names in EDF (remove spaces if needed)
    cleaned_ch_names = [ch.upper().replace(' ', '') for ch in available_channels]
    
    # Map from config channels to index
    target_channels = config.EEG_CHANNELS
    
    for target in target_channels:
        # Try exact match
        if target in available_channels:
            picks.append(available_channels.index(target))
            plot_channels.append(target.replace('EEG ', '').replace('-REF', ''))
        else:
            # Try approximate match (e.g. without 'EEG ' or '-REF')
            simple_target = target.replace('EEG ', '').replace('-REF', '').upper()
            found = False
            for idx, ch in enumerate(cleaned_ch_names):
                if simple_target in ch:
                    picks.append(idx)
                    plot_channels.append(simple_target)
                    found = True
                    break
    
    if not picks:
        logging.warning("No matching channels found for clinical plot.")
        _plot_simple_overlay(windows, labels, predictions, save_path)
        return

    raw.pick(picks)
    
    # Filter
    raw.filter(l_freq=0.5, h_freq=50.0, verbose=False)
    
    # Get data: (n_channels, n_times)
    data, times = raw.get_data(return_times=True)
    n_channels = len(plot_channels)
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Scale and Offset
    # Clinical EEG is often ~50uV/cm. Here we just stack them.
    # Estimate robust scaling
    scale = np.nanpercentile(np.abs(data), 95) * 3  # reasonable scale
    offset = scale * 1.5
    
    # Plot traces
    for i in range(n_channels):
        # Invert i so top channel is at top
        y_pos = (n_channels - 1 - i) * offset
        trace = data[i, :]
        ax.plot(times, trace + y_pos, color='black', linewidth=0.6)
        
    # Set Y-ticks
    yticks = [(n_channels - 1 - i) * offset for i in range(n_channels)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(plot_channels)
    
    # Create prediction masks
    # "predictions" is per window. We need to map to time.
    # Create a boolean mask for the full timeline
    pred_mask = np.zeros_like(times, dtype=bool)
    
    # Map windows to sample indices
    sfreq = raw.info['sfreq']
    stride_samples = int(config.STRIDE_SEC * sfreq)
    window_samples = int(config.WINDOW_SEC * sfreq)
    
    for i, pred in enumerate(predictions):
        if pred == 1:
            start_idx = int(i * config.STRIDE_SEC * sfreq)
            end_idx = start_idx + window_samples
            if start_idx < len(pred_mask):
                pred_mask[start_idx:min(end_idx, len(pred_mask))] = True
                
    # Fill prediction regions
    y_min = -offset
    y_max = n_channels * offset
    
    # Use fill_between for the mask
    ax.fill_between(times, y_min, y_max, where=pred_mask, 
                    color='red', alpha=0.15, label='Detected Seizure')

    # Zoom Logic: REMOVED (Show full recording)
    # seizure_indices = np.where(pred_mask)[0]
    # if len(seizure_indices) > 0:
    #     ax.text(times[seizure_indices[0]], y_max, "DETECTED SEIZURE", color='red', fontsize=12, fontweight='bold')

    # Styling
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Clinical EEG Visualization with Seizure Detections")
    
    plt.tight_layout()
    
    # Save first
    plt.savefig(save_path, dpi=150)
    print(f"Saved clinical EEG plot to {save_path}")
    
    # Then show
    print("Displaying plot... (Close window to continue)")
    plt.show()
    plt.close()


def _plot_simple_overlay(
    windows: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
):
    """Fallback simple plot."""
    n_windows = len(windows)
    time_axis = np.arange(n_windows) * config.STRIDE_SEC
    
    plt.figure(figsize=(15, 6))
    plt.fill_between(time_axis, 0, 1, where=(predictions==1), 
                     color='red', alpha=0.5, label='Predicted Seizure', step='mid')
    plt.title("Seizure Detection (Simple Overlay)")
    plt.xlabel("Time (seconds)")
    plt.yticks([])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved simple overlay plot to {save_path}")


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Seizure'],
                yticklabels=['Background', 'Seizure'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def export_predictions_excel(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
):
    """
    Export predictions to Excel with start/end times.
    Groups consecutive windows into events.
    """
    events = []
    
    # Helper to convert events
    def get_events(seq, name):
        events_list = []
        if len(seq) == 0: return []
        
        is_active = False
        start_idx = 0
        
        for i, val in enumerate(seq):
            if val == 1 and not is_active:
                is_active = True
                start_idx = i
            elif val == 0 and is_active:
                is_active = False
                events_list.append({
                    "Type": name,
                    "Start (s)": start_idx * config.STRIDE_SEC,
                    "End (s)": (i * config.STRIDE_SEC) + config.WINDOW_SEC,
                    "Duration (s)": ((i - start_idx) * config.STRIDE_SEC) + config.WINDOW_SEC
                })
        
        # Handle event ending at last window
        if is_active:
            events_list.append({
                "Type": name,
                "Start (s)": start_idx * config.STRIDE_SEC,
                "End (s)": (len(seq) * config.STRIDE_SEC) + config.WINDOW_SEC,
                "Duration (s)": ((len(seq) - start_idx) * config.STRIDE_SEC) + config.WINDOW_SEC
            })
            
        return events_list

    pred_events = get_events(predictions, "Predicted Seizure")
    true_events = get_events(labels, "True Seizure")
    
    all_events = sorted(pred_events + true_events, key=lambda x: x["Start (s)"])
    
    df = pd.DataFrame(all_events)
    df.to_excel(save_path, index=False)
    print(f"Saved predictions to {save_path}")
