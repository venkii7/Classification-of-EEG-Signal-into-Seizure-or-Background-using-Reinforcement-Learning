"""
ppo_agent.py — Proximal Policy Optimization agent for EEG seizure detection.

Components:
  - Actor-Critic network: shared CNN backbone → policy head + value head
  - PPO-Clip objective
  - Generalized Advantage Estimation (GAE)
  - Rollout buffer for collecting trajectories
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from .feature_extractor import CNNFeatureExtractor


class RolloutBuffer:
    """
    Buffer to store rollout trajectories for PPO.

    Stores states, actions, log-probs, rewards, dones, and values.
    Computes returns and advantages using GAE.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):
        """Store one transition."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = config.PPO_GAMMA,
        gae_lambda: float = config.PPO_GAE_LAMBDA,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation and returns.

        Args:
            last_value:  V(s_T+1) — value of the next state after the last stored step.
            gamma:       Discount factor.
            gae_lambda:  GAE smoothing parameter.

        Returns:
            returns:     Array of shape (T,) — discounted returns.
            advantages:  Array of shape (T,) — GAE advantages.
        """
        values = self.values + [last_value]
        rewards = self.rewards
        dones = self.dones

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages

    def get_tensors(self, device: torch.device):
        """Convert stored data to tensors."""
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        return states, actions, log_probs

    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared CNN feature backbone.

    Actor:  features → policy (softmax over 2 actions)
    Critic: features → scalar value estimate
    """

    def __init__(
        self,
        n_channels: int = config.NUM_CHANNELS,
        window_samples: int = config.WINDOW_SAMPLES,
        feature_dim: int = config.PPO_FEATURE_DIM,
        n_actions: int = config.NUM_CLASSES,
    ):
        super().__init__()

        self.feature_extractor = CNNFeatureExtractor(
            n_channels=n_channels,
            window_samples=window_samples,
            feature_dim=feature_dim,
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, n_channels, window_samples)

        Returns:
            action_probs: (batch, n_actions) — probability distribution
            value:        (batch, 1) — state value estimate
        """
        features = self.feature_extractor(x)
        logits = self.actor(features)
        action_probs = torch.softmax(logits, dim=-1)
        value = self.critic(features)
        return action_probs, value

    def get_action_and_value(self, x: torch.Tensor, action=None):
        """
        Sample action and compute log-prob + value + entropy.

        Args:
            x:      (batch, n_channels, window_samples)
            action: If provided, compute log_prob of this action instead of sampling.

        Returns:
            action:    sampled or provided action
            log_prob:  log probability of the action
            entropy:   entropy of the policy distribution
            value:     value estimate
        """
        action_probs, value = self.forward(x)
        dist = Categorical(action_probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class PPOAgent:
    """
    PPO Agent for EEG seizure detection.

    Features:
      - Actor-Critic with shared CNN backbone
      - PPO-Clip objective
      - GAE for advantage estimation
      - Entropy bonus for exploration
    """

    def __init__(
        self,
        n_channels: int = config.NUM_CHANNELS,
        window_samples: int = config.WINDOW_SAMPLES,
        feature_dim: int = config.PPO_FEATURE_DIM,
        n_actions: int = config.NUM_CLASSES,
        lr: float = config.PPO_LR,
        gamma: float = config.PPO_GAMMA,
        gae_lambda: float = config.PPO_GAE_LAMBDA,
        clip_eps: float = config.PPO_CLIP_EPS,
        entropy_coef: float = config.PPO_ENTROPY_COEF,
        value_coef: float = config.PPO_VALUE_COEF,
        max_grad_norm: float = config.PPO_MAX_GRAD_NORM,
        num_epochs: int = config.PPO_NUM_EPOCHS,
        batch_size: int = config.PPO_BATCH_SIZE,
        device: Optional[torch.device] = None,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        # ── Network ───────────────────────────────────────────────────────
        self.network = ActorCritic(
            n_channels, window_samples, feature_dim, n_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # ── Rollout buffer ────────────────────────────────────────────────
        self.rollout_buffer = RolloutBuffer()

        self.total_steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Select action using the current policy.

        Args:
            state:    EEG window, shape (n_channels, window_samples).
            training: If True, sample from policy; if False, use argmax.

        Returns:
            action:   Selected action.
            log_prob: Log probability of the selected action.
            value:    State value estimate.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if training:
                action, log_prob, _, value = self.network.get_action_and_value(state_t)
                return action.item(), log_prob.item(), value.item()
            else:
                action_probs, value = self.network(state_t)
                action = action_probs.argmax(dim=1)
                return action.item(), 0.0, value.squeeze(-1).item()

    def store_transition(self, state, action, log_prob, reward, done, value):
        """Store a transition in the rollout buffer."""
        self.rollout_buffer.store(state, action, log_prob, reward, done, value)
        self.total_steps += 1

    def update(self) -> dict:
        """
        Perform PPO update using collected rollout data.

        Returns:
            Dictionary with loss components.
        """
        if len(self.rollout_buffer) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        # ── Compute returns and advantages ────────────────────────────────
        # Get value of last state for GAE
        last_state = self.rollout_buffer.states[-1]
        last_state_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.network(last_state_t)
            last_value = last_value.item()

        returns, advantages = self.rollout_buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )

        # ── Convert to tensors ────────────────────────────────────────────
        states, actions, old_log_probs = self.rollout_buffer.get_tensors(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        if advantages_t.std() > 0:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ── PPO update epochs ─────────────────────────────────────────────
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        T = len(states)
        indices = np.arange(T)

        for _ in range(self.num_epochs):
            np.random.shuffle(indices)

            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]

                # Get current policy values
                _, new_log_probs, entropy, new_values = \
                    self.network.get_action_and_value(batch_states, batch_actions)

                # ── Policy loss (PPO-Clip) ────────────────────────────────
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value loss ────────────────────────────────────────────
                value_loss = nn.functional.mse_loss(new_values, batch_returns)

                # ── Entropy bonus ─────────────────────────────────────────
                entropy_loss = -entropy.mean()

                # ── Total loss ────────────────────────────────────────────
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()
                n_updates += 1

        # Clear buffer
        self.rollout_buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
