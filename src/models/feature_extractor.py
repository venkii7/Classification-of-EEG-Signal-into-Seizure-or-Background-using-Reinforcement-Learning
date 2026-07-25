"""
feature_extractor.py — CNN and CNN-LSTM feature extractors for multi-channel EEG.

These networks transform raw EEG windows of shape (n_channels, window_samples)
into compact feature vectors used by the RL agents (DQN / PPO).
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    1D-CNN feature extractor for multi-channel EEG.

    Architecture:
        Conv1d → BatchNorm → ReLU → MaxPool  (×3 blocks)
        → AdaptiveAvgPool1d → Flatten → Linear → ReLU → feature_dim

    Input:  (batch, n_channels, window_samples)
    Output: (batch, feature_dim)
    """

    def __init__(
        self,
        n_channels: int = 19,
        window_samples: int = 512,
        feature_dim: int = 128,
    ):
        super().__init__()

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

        # Adaptive pooling to fixed size regardless of input length
        self.pool = nn.AdaptiveAvgPool1d(4)

        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, window_samples)
        Returns:
            features: (batch, feature_dim)
        """
        x = self.conv_blocks(x)     # (batch, 128, T')
        x = self.pool(x)            # (batch, 128, 4)
        x = x.flatten(1)            # (batch, 512)
        x = self.fc(x)              # (batch, feature_dim)
        return x


class CNNLSTMFeatureExtractor(nn.Module):
    """
    CNN-LSTM feature extractor for capturing temporal dependencies across
    consecutive EEG windows within an episode.

    The CNN backbone extracts per-window features, and the LSTM processes
    a sequence of these features to capture temporal context.

    Usage:
        - For DQN: pass single windows with hidden state carried forward
        - For PPO: can process sequences of windows in one pass

    Input:  (batch, n_channels, window_samples)  — single window mode
            OR (batch, seq_len, n_channels, window_samples) — sequence mode
    Output: (batch, feature_dim), (h, c)
    """

    def __init__(
        self,
        n_channels: int = 19,
        window_samples: int = 512,
        feature_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
    ):
        super().__init__()

        self.cnn = CNNFeatureExtractor(
            n_channels=n_channels,
            window_samples=window_samples,
            feature_dim=feature_dim,
        )

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else 0.2,
        )

        self.fc_out = nn.Sequential(
            nn.Linear(lstm_hidden, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state to zeros."""
        h = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        hidden=None,
    ):
        """
        Args:
            x:      (batch, n_channels, window_samples)        — single window
                    OR (batch, seq_len, n_channels, window_samples) — sequence
            hidden: (h, c) tuple or None. If None, initializes to zeros.

        Returns:
            features: (batch, feature_dim)
            hidden:   (h, c) — updated hidden state
        """
        if x.dim() == 3:
            # Single window mode: (batch, n_channels, window_samples)
            batch_size = x.size(0)
            cnn_out = self.cnn(x)                     # (batch, feature_dim)
            cnn_out = cnn_out.unsqueeze(1)             # (batch, 1, feature_dim)
        elif x.dim() == 4:
            # Sequence mode: (batch, seq_len, n_channels, window_samples)
            batch_size, seq_len, n_ch, n_samp = x.shape
            x_flat = x.reshape(batch_size * seq_len, n_ch, n_samp)
            cnn_out = self.cnn(x_flat)                 # (B*T, feature_dim)
            cnn_out = cnn_out.reshape(batch_size, seq_len, -1)  # (B, T, feature_dim)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        lstm_out, hidden = self.lstm(cnn_out, hidden)  # (batch, seq, lstm_hidden)

        # Take the last time step output
        last_out = lstm_out[:, -1, :]                   # (batch, lstm_hidden)
        features = self.fc_out(last_out)                # (batch, feature_dim)

        return features, hidden
