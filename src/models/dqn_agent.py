"""
dqn_agent.py — Deep Q-Network agent for EEG seizure detection.

Components:
  - Q-Network: feature extractor → MLP head → Q-values for 2 actions
  - Target network with periodic hard copy
  - Experience replay buffer
  - ε-greedy exploration with linear decay
"""

import random
from collections import deque, namedtuple
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from .feature_extractor import CNNFeatureExtractor


# ──────────────────────────────────────────────────────────────────────────────
# Transition tuple for replay buffer
# ──────────────────────────────────────────────────────────────────────────────
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size circular replay buffer for DQN experience replay."""

    def __init__(self, capacity: int = config.DQN_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a random mini-batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q-Network: CNN feature extractor + MLP head outputting Q-values.

    Input:  (batch, n_channels, window_samples)
    Output: (batch, n_actions)
    """

    def __init__(
        self,
        n_channels: int = config.NUM_CHANNELS,
        window_samples: int = config.WINDOW_SAMPLES,
        feature_dim: int = config.DQN_FEATURE_DIM,
        n_actions: int = config.NUM_CLASSES,
    ):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(
            n_channels=n_channels,
            window_samples=window_samples,
            feature_dim=feature_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.head(features)


class DQNAgent:
    """
    DQN Agent for EEG seizure detection.

    Features:
      - Double DQN (target network)
      - ε-greedy with linear decay
      - Experience replay
    """

    def __init__(
        self,
        n_channels: int = config.NUM_CHANNELS,
        window_samples: int = config.WINDOW_SAMPLES,
        feature_dim: int = config.DQN_FEATURE_DIM,
        n_actions: int = config.NUM_CLASSES,
        lr: float = config.DQN_LR,
        gamma: float = config.DQN_GAMMA,
        epsilon_start: float = config.DQN_EPSILON_START,
        epsilon_end: float = config.DQN_EPSILON_END,
        epsilon_decay: int = config.DQN_EPSILON_DECAY,
        buffer_size: int = config.DQN_BUFFER_SIZE,
        batch_size: int = config.DQN_BATCH_SIZE,
        target_update: int = config.DQN_TARGET_UPDATE,
        device: Optional[torch.device] = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device or torch.device("cpu")

        # ── Networks ──────────────────────────────────────────────────────
        self.q_network = QNetwork(n_channels, window_samples, feature_dim, n_actions).to(self.device)
        self.target_network = QNetwork(n_channels, window_samples, feature_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # ── Optimizer ─────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # ── Replay buffer ─────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(buffer_size)

        # ── Counters ──────────────────────────────────────────────────────
        self.steps_done = 0
        self.updates_done = 0

    def get_epsilon(self) -> float:
        """Compute current epsilon using linear decay."""
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(
            0, 1 - self.steps_done / self.epsilon_decay
        )
        return eps

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state:    EEG window, shape (n_channels, window_samples).
            training: If False, always use greedy action.

        Returns:
            action: 0 (background) or 1 (seizure).
        """
        if training:
            eps = self.get_epsilon()
            if random.random() < eps:
                return random.randint(0, self.n_actions - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1

    def update(self) -> Optional[float]:
        """
        Perform one gradient descent step on a mini-batch from the replay buffer.

        Returns:
            loss value, or None if buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # ── Current Q-values ──────────────────────────────────────────────
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ── Target Q-values (Double DQN) ──────────────────────────────────
        with torch.no_grad():
            # Use online network to select actions
            next_actions = self.q_network(next_states_t).argmax(dim=1)
            # Use target network to evaluate those actions
            next_q = self.target_network(next_states_t)
            next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # ── Loss & update ─────────────────────────────────────────────────
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.updates_done += 1

        # ── Sync target network ───────────────────────────────────────────
        if self.updates_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "updates_done": self.updates_done,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.updates_done = checkpoint["updates_done"]
