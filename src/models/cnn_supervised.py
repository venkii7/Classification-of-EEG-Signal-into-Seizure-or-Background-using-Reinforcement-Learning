"""
cnn_supervised.py — Standalone supervised CNN classifier for EEG seizure detection.

Provides:
  - EEGCNNClassifier: PyTorch nn.Module with 3 Conv1d blocks + classification head
  - CNNSupervisedAgent: Wrapper with select_action/save/load interface matching PPO/DQN agents

Architecture (same backbone as PPO/DQN feature extractor):
  Input:  (batch, 19, 512)
  Block 1: Conv1d(19→32, k=7, pad=3) → BN → ReLU → MaxPool(2) → (batch, 32, 256)
  Block 2: Conv1d(32→64, k=5, pad=2) → BN → ReLU → MaxPool(2) → (batch, 64, 128)
  Block 3: Conv1d(64→128, k=3, pad=1) → BN → ReLU → MaxPool(2) → (batch, 128, 64)
  AdaptiveAvgPool1d(4) → Flatten → (batch, 512)
  Classification Head: Linear(512→256) → ReLU → Dropout → Linear(256→128) → ReLU → Dropout → Linear(128→2)
"""

import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class EEGCNNClassifier(nn.Module):
    """
    Supervised CNN classifier for binary EEG seizure detection.

    Uses the same 3-block Conv1d backbone as PPO/DQN, followed by a
    classification head that outputs logits for 2 classes.

    Input:  (batch, n_channels, window_samples)  e.g. (batch, 19, 512)
    Output: (batch, 2)  — logits for [background, seizure]
    """

    def __init__(
        self,
        n_channels: int = 19,
        window_samples: int = 512,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.window_samples = window_samples
        self.n_classes = n_classes

        # ── 3-block Conv1d backbone (identical to feature_extractor.py) ───
        self.conv_blocks = nn.Sequential(
            # Block 1: n_channels → 32
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 2: 32 → 64
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 3: 64 → 128
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(4)

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, n_channels, window_samples)
        Returns:
            logits: (batch, n_classes)
        """
        x = self.conv_blocks(x)   # (batch, 128, T')
        x = self.pool(x)          # (batch, 128, 4)
        x = x.flatten(1)          # (batch, 512)
        x = self.classifier(x)    # (batch, n_classes)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities instead of logits."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class CNNSupervisedAgent:
    """
    Wrapper around EEGCNNClassifier that provides the same interface
    as PPOAgent and DQNAgent (select_action, save, load).

    This allows seamless use with the existing evaluation pipeline.
    """

    def __init__(
        self,
        n_channels: int = config.NUM_CHANNELS,
        window_samples: int = config.WINDOW_SAMPLES,
        n_classes: int = config.NUM_CLASSES,
        dropout: float = config.CNN_DROPOUT,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = EEGCNNClassifier(
            n_channels=n_channels,
            window_samples=window_samples,
            n_classes=n_classes,
            dropout=dropout,
        ).to(device)

        self.n_channels = n_channels
        self.window_samples = window_samples

    def select_action(
        self,
        state: np.ndarray,
        training: bool = False,
    ) -> int:
        """
        Select an action (0=background, 1=seizure) for a single EEG window.

        Uses greedy (argmax) selection — consistent with how PPO/DQN
        are evaluated in inference mode.

        Args:
            state: EEG window array of shape (n_channels, window_samples)
            training: ignored (kept for interface compatibility)
        Returns:
            action: 0 or 1
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.model(x)
            action = int(torch.argmax(logits, dim=-1).item())
        return action

    def save(self, path: str) -> None:
        """Save model weights to a checkpoint file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "n_channels": self.n_channels,
            "window_samples": self.window_samples,
        }, path)
        logging.info(f"CNN model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        logging.info(f"CNN model loaded from {path}")
