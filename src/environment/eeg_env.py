"""
eeg_env.py — Custom Gymnasium environment for EEG seizure detection.

Models EEG seizure detection as a sequential decision-making problem:
  - Each episode = one EEG recording (sequence of sliding windows)
  - State  = feature vector of the current EEG window (n_channels × window_samples)
  - Action = 0 (background) or 1 (seizure)
  - Reward = asymmetric to penalize missed seizures more heavily

The environment iterates through windows sequentially, simulating a real-time
streaming EEG scenario.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class EEGSeizureEnv(gym.Env):
    """
    Gymnasium environment for EEG seizure detection via RL.

    The agent steps through consecutive EEG windows of a recording and
    classifies each window as seizure (1) or background (0).

    Attributes:
        windows_list:  List of arrays, each shape (n_windows_i, n_channels, window_samples).
        labels_list:   List of arrays, each shape (n_windows_i,).
        reward_scheme: Dict with keys 'correct_seiz', 'correct_bckg', 'miss_seiz', 'false_alarm'.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        file_boundaries: Optional[np.ndarray] = None,
        reward_scheme: Optional[Dict[str, float]] = None,
        shuffle_files: bool = True,
    ):
        """
        Initialize the EEG environment.

        Args:
            windows:         All windows, shape (N, n_channels, window_samples).
            labels:          All labels, shape (N,).
            file_boundaries: Array of indices where new files start. If None,
                             treats all windows as one episode.
            reward_scheme:   Custom reward values. Defaults to config values.
            shuffle_files:   Whether to randomize file order across episodes.
        """
        super().__init__()

        self.all_windows = windows
        self.all_labels = labels
        self.shuffle_files = shuffle_files

        # ── Build per-file episode data ───────────────────────────────────
        if file_boundaries is not None and len(file_boundaries) > 0:
            self.episodes = []
            boundaries = list(file_boundaries) + [len(windows)]
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                if end > start:
                    self.episodes.append((start, end))
        else:
            # Treat all data as one big episode — split into chunks
            # of ~200 windows each to keep episodes manageable
            chunk_size = 200
            self.episodes = []
            for i in range(0, len(windows), chunk_size):
                end = min(i + chunk_size, len(windows))
                self.episodes.append((i, end))

        # ── Class-balanced episode indices ────────────────────────────────
        # Classify episodes: "seizure" if they contain ≥1 seizure window,
        # "background" if they contain only background windows.
        self.seizure_episode_indices = []
        self.background_episode_indices = []
        for idx, (start, end) in enumerate(self.episodes):
            episode_labels = labels[start:end]
            if np.any(episode_labels == 1):
                self.seizure_episode_indices.append(idx)
            else:
                self.background_episode_indices.append(idx)

        logging.info(
            f"Episode balance: {len(self.seizure_episode_indices)} seizure episodes, "
            f"{len(self.background_episode_indices)} background-only episodes "
            f"(total: {len(self.episodes)})"
        )
        self._balance_counter = 0  # alternates between seizure/background

        # ── Reward scheme ─────────────────────────────────────────────────
        if reward_scheme is None:
            self.reward_scheme = {
                "correct_seiz": config.REWARD_CORRECT_SEIZ,
                "correct_bckg": config.REWARD_CORRECT_BCKG,
                "miss_seiz":    config.REWARD_MISS_SEIZ,
                "false_alarm":  config.REWARD_FALSE_ALARM,
            }
        else:
            self.reward_scheme = reward_scheme

        # ── Gym spaces ────────────────────────────────────────────────────
        n_channels = windows.shape[1]
        window_samples = windows.shape[2]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_channels, window_samples),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(config.NUM_CLASSES)  # 0 or 1

        # ── Episode state ─────────────────────────────────────────────────
        self.current_episode_idx = 0
        self.current_step = 0
        self.episode_start = 0
        self.episode_end = 0

        # ── Tracking for metrics ──────────────────────────────────────────
        self.episode_predictions = []
        self.episode_true_labels = []
        self.episode_reward = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment: pick an episode and return the first observation.

        Returns:
            observation: shape (n_channels, window_samples)
            info:        dict with episode metadata
        """
        super().reset(seed=seed)

        # Select episode — class-balanced sampling during training
        if self.shuffle_files:
            # Alternate between seizure and background episodes for balance
            if (len(self.seizure_episode_indices) > 0 and
                len(self.background_episode_indices) > 0):
                if self._balance_counter % 2 == 0:
                    # Pick a seizure episode
                    pool = self.seizure_episode_indices
                else:
                    # Pick a background episode
                    pool = self.background_episode_indices
                self.current_episode_idx = pool[
                    self.np_random.integers(0, len(pool))
                ]
                self._balance_counter += 1
            else:
                # Fallback: all episodes are one class
                self.current_episode_idx = self.np_random.integers(0, len(self.episodes))
        else:
            self.current_episode_idx = (
                (self.current_episode_idx + 1) % len(self.episodes)
                if hasattr(self, "_has_reset") else 0
            )
        self._has_reset = True

        self.episode_start, self.episode_end = self.episodes[self.current_episode_idx]
        self.current_step = 0

        # Reset tracking
        self.episode_predictions = []
        self.episode_true_labels = []
        self.episode_reward = 0.0

        obs = self.all_windows[self.episode_start].astype(np.float32)
        info = {
            "episode_idx": self.current_episode_idx,
            "episode_length": self.episode_end - self.episode_start,
            "true_label": int(self.all_labels[self.episode_start]),
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step: classify the current window and move to the next.

        Args:
            action: 0 (background) or 1 (seizure)

        Returns:
            observation: Next window (or current if done)
            reward:      Scalar reward for this step
            terminated:  True if episode is over
            truncated:   Always False (no time limit beyond episode length)
            info:        Metadata dict
        """
        global_idx = self.episode_start + self.current_step
        true_label = int(self.all_labels[global_idx])
        action = int(action)

        # ── Compute reward ────────────────────────────────────────────────
        if action == true_label:
            if true_label == 1:
                reward = self.reward_scheme["correct_seiz"]
            else:
                reward = self.reward_scheme["correct_bckg"]
        else:
            if true_label == 1 and action == 0:
                reward = self.reward_scheme["miss_seiz"]     # Missed seizure (FN)
            else:
                reward = self.reward_scheme["false_alarm"]    # False alarm (FP)

        # ── Track predictions ─────────────────────────────────────────────
        self.episode_predictions.append(action)
        self.episode_true_labels.append(true_label)
        self.episode_reward += reward

        # ── Advance step ──────────────────────────────────────────────────
        self.current_step += 1
        terminated = (self.episode_start + self.current_step) >= self.episode_end

        if not terminated:
            next_idx = self.episode_start + self.current_step
            obs = self.all_windows[next_idx].astype(np.float32)
        else:
            obs = self.all_windows[global_idx].astype(np.float32)  # Dummy, won't be used

        info = {
            "true_label": true_label,
            "action": action,
            "correct": action == true_label,
            "step_in_episode": self.current_step,
            "episode_reward": self.episode_reward,
        }

        if terminated:
            # Add summary statistics for the completed episode
            preds = np.array(self.episode_predictions)
            trues = np.array(self.episode_true_labels)
            info["episode_accuracy"] = float(np.mean(preds == trues))
            info["episode_total_reward"] = self.episode_reward
            info["episode_n_seizure"] = int(trues.sum())
            info["episode_n_background"] = int(len(trues) - trues.sum())

        return obs, reward, terminated, False, info

    def get_episode_count(self) -> int:
        """Return number of episodes (files) available."""
        return len(self.episodes)

    def get_class_distribution(self) -> Dict[str, int]:
        """Return overall class distribution in the dataset."""
        n_seiz = int(self.all_labels.sum())
        return {"seizure": n_seiz, "background": len(self.all_labels) - n_seiz}
