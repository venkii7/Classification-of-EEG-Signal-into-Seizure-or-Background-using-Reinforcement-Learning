"""
train_cnn.py — Supervised training loop for CNN-based EEG seizure detection.

Trains the EEGCNNClassifier using standard supervised learning:
  - CrossEntropyLoss with optional inverse-frequency class weighting
  - Adam optimizer with gradient clipping
  - Early stopping based on dev set F1-score
  - Saves best model (by F1) and final model checkpoints
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from src.models.cnn_supervised import EEGCNNClassifier, CNNSupervisedAgent
from src.evaluation.evaluate import evaluate_agent


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for handling class imbalance.

    Args:
        labels: array of integer labels (0 or 1)
    Returns:
        Tensor of shape (n_classes,) with inverse-frequency weights
    """
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(classes) * counts)
    weight_tensor = torch.ones(config.NUM_CLASSES, dtype=torch.float32)
    for cls, w in zip(classes, weights):
        weight_tensor[int(cls)] = w
    logging.info(
        f"Class weights: background={weight_tensor[0]:.3f}, seizure={weight_tensor[1]:.3f}"
    )
    return weight_tensor


def train_cnn(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    dev_windows: Optional[np.ndarray] = None,
    dev_labels: Optional[np.ndarray] = None,
    num_epochs: int = config.CNN_EPOCHS,
    batch_size: int = config.CNN_BATCH_SIZE,
    learning_rate: float = config.CNN_LR,
    patience: int = config.CNN_PATIENCE,
    device: Optional[torch.device] = None,
    resume_path: Optional[str] = None,
) -> CNNSupervisedAgent:
    """
    Train a supervised CNN classifier on preprocessed EEG data.

    Args:
        train_windows: Training windows, shape (N, n_channels, window_samples).
        train_labels:  Training labels, shape (N,).
        dev_windows:   Dev set windows for evaluation.
        dev_labels:    Dev set labels for evaluation.
        num_epochs:    Maximum number of training epochs.
        batch_size:    Mini-batch size.
        learning_rate: Learning rate for Adam optimizer.
        patience:      Early stopping patience (epochs without dev F1 improvement).
        device:        Torch device.
        resume_path:   Path to checkpoint to resume training from.

    Returns:
        Trained CNNSupervisedAgent.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Create agent ──────────────────────────────────────────────────────
    n_channels = train_windows.shape[1]
    window_samples = train_windows.shape[2]
    agent = CNNSupervisedAgent(
        n_channels=n_channels,
        window_samples=window_samples,
        device=device,
    )
    model = agent.model

    # ── Resume from checkpoint ────────────────────────────────────────────
    if resume_path is not None:
        logging.info(f"Resuming CNN training from: {resume_path}")
        agent.load(resume_path)

    # ── Prepare data ──────────────────────────────────────────────────────
    # If no dev set provided, split training data into 90/10 train/val
    if dev_windows is None:
        logging.info("No dev set provided — splitting train data 90/10 for internal validation")
        n_total = len(train_windows)
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        split_point = int(n_total * 0.9)

        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

        val_windows = train_windows[val_indices]
        val_labels = train_labels[val_indices]

        # Use only train portion for training
        actual_train_windows = train_windows[train_indices]
        actual_train_labels = train_labels[train_indices]

        val_seiz = int(np.sum(val_labels == 1))
        val_bckg = int(np.sum(val_labels == 0))
        logging.info(
            f"Internal val set: {len(val_labels)} windows "
            f"(seizure={val_seiz}, background={val_bckg})"
        )

        dev_windows = val_windows
        dev_labels = val_labels
    else:
        actual_train_windows = train_windows
        actual_train_labels = train_labels

    # ── Oversample Seizure Windows (Training Set Only) ────────────────────
    if getattr(config, "CNN_OVERSAMPLE", True):
        logging.info("Oversampling seizure windows to balance training classes...")
        bckg_idx = np.where(actual_train_labels == 0)[0]
        seiz_idx = np.where(actual_train_labels == 1)[0]
        
        n_bckg = len(bckg_idx)
        n_seiz = len(seiz_idx)
        
        if n_seiz > 0 and n_seiz < n_bckg:
            # Randomly duplicate seizure indices to match background count
            extra_seiz_idx = np.random.choice(seiz_idx, size=(n_bckg - n_seiz), replace=True)
            balanced_indices = np.concatenate([bckg_idx, seiz_idx, extra_seiz_idx])
            np.random.shuffle(balanced_indices)
            
            actual_train_windows = actual_train_windows[balanced_indices]
            actual_train_labels = actual_train_labels[balanced_indices]
            
            logging.info(f"Oversampled training set: {len(actual_train_labels)} windows (seizure={n_bckg}, background={n_bckg})")

    train_x = torch.FloatTensor(np.array(actual_train_windows))
    train_y = torch.LongTensor(np.array(actual_train_labels))
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    # ── Loss function with optional class weighting ───────────────────────
    if config.CNN_CLASS_WEIGHT:
        class_weights = compute_class_weights(np.array(actual_train_labels)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, eps=config.CNN_ADAM_EPS
    )

    # ── Tracking ──────────────────────────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0

    # If resuming, evaluate first to establish baseline F1
    if resume_path is not None and dev_windows is not None:
        logging.info("Evaluating resumed model to establish baseline F1...")
        baseline_metrics = evaluate_agent(
            agent, dev_windows, dev_labels,
            agent_type="cnn", device=device
        )
        best_f1 = baseline_metrics["f1_score"]
        logging.info(f"Baseline F1 from resumed model: {best_f1:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────
    logging.info(
        f"Starting CNN training: {num_epochs} epochs, "
        f"{len(train_loader)} batches/epoch, "
        f"batch_size={batch_size}, lr={learning_rate}"
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.CNN_MAX_GRAD_NORM
            )

            optimizer.step()

            # Track metrics
            epoch_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_total += batch_x.size(0)

        # ── Epoch summary ─────────────────────────────────────────────────
        avg_loss = epoch_loss / epoch_total
        avg_acc = epoch_correct / epoch_total

        # ── Dev set evaluation ────────────────────────────────────────────
        if dev_windows is not None:
            metrics = evaluate_agent(
                agent, dev_windows, dev_labels,
                agent_type="cnn", device=device
            )
            dev_f1 = metrics["f1_score"]
            dev_sens = metrics["sensitivity"]
            dev_spec = metrics["specificity"]

            logging.info(
                f"Epoch {epoch:>3}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {avg_acc:.4f} | "
                f"Dev F1: {dev_f1:.4f} | "
                f"Sens: {dev_sens:.4f} | "
                f"Spec: {dev_spec:.4f}"
            )

            # ── Model selection (best F1) ─────────────────────────────────
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                patience_counter = 0
                save_path = os.path.join(config.CHECKPOINT_DIR, "cnn_best.pt")
                agent.save(save_path)
                logging.info(
                    f"  ★ New best F1: {best_f1:.4f} — saved to {save_path}"
                )
            else:
                patience_counter += 1

            # ── Early stopping ────────────────────────────────────────────
            if patience_counter >= patience:
                logging.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs). "
                    f"Best F1: {best_f1:.4f}"
                )
                break
        else:
            logging.info(
                f"Epoch {epoch:>3}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {avg_acc:.4f}"
            )

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(config.CHECKPOINT_DIR, "cnn_final.pt")
    agent.save(final_path)
    logging.info(f"Training complete. Final model saved to {final_path}")
    logging.info(f"Best dev F1: {best_f1:.4f}")

    return agent
