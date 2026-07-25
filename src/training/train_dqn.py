"""
train_dqn.py — DQN training loop for EEG seizure detection.

Trains the DQN agent by interacting with the EEGSeizureEnv:
  - For each episode: reset → step through all windows → train on replay buffer
  - Logs episode rewards, loss, accuracy
  - Periodically evaluates on dev set and checkpoints best model
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from src.environment.eeg_env import EEGSeizureEnv
from src.models.dqn_agent import DQNAgent
from src.evaluation.evaluate import evaluate_agent


def train_dqn(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    dev_windows: Optional[np.ndarray] = None,
    dev_labels: Optional[np.ndarray] = None,
    num_episodes: int = config.NUM_EPISODES,
    eval_interval: int = config.EVAL_INTERVAL,
    device: Optional[torch.device] = None,
    resume_path: Optional[str] = None,
) -> DQNAgent:
    """
    Train a DQN agent on preprocessed EEG data.

    Args:
        train_windows: Training windows, shape (N, n_channels, window_samples).
        train_labels:  Training labels, shape (N,).
        dev_windows:   Dev set windows for evaluation.
        dev_labels:    Dev set labels for evaluation.
        num_episodes:  Number of training episodes.
        eval_interval: Evaluate every N episodes.
        device:        Torch device.
        resume_path:   Path to checkpoint to resume training from.

    Returns:
        Trained DQNAgent.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Create environment ────────────────────────────────────────────────
    train_env = EEGSeizureEnv(train_windows, train_labels, shuffle_files=True)
    logging.info(f"Training env: {train_env.get_episode_count()} episodes, "
                 f"class dist: {train_env.get_class_distribution()}")

    # ── Create agent ──────────────────────────────────────────────────────
    n_channels = train_windows.shape[1]
    window_samples = train_windows.shape[2]
    agent = DQNAgent(
        n_channels=n_channels,
        window_samples=window_samples,
        device=device,
    )

    # ── Resume from checkpoint if provided ────────────────────────────────
    if resume_path is not None:
        logging.info(f"Resuming DQN training from: {resume_path}")
        agent.load(resume_path)

    # ── Tracking ──────────────────────────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_f1 = 0.0
    episode_rewards = []
    episode_losses = []

    # ── Training loop ─────────────────────────────────────────────────────
    for episode in tqdm(range(1, num_episodes + 1), desc="DQN Training"):
        state, info = train_env.reset()
        episode_reward = 0.0
        episode_loss_sum = 0.0
        n_updates = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Step environment
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated

            # Store transition and update
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()

            if loss is not None:
                episode_loss_sum += loss
                n_updates += 1

            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)
        avg_loss = episode_loss_sum / max(n_updates, 1)
        episode_losses.append(avg_loss)

        # ── Logging ───────────────────────────────────────────────────────
        eps = agent.get_epsilon()
        acc = info.get("episode_accuracy", 0.0)
        if episode % 5 == 0:
            logging.info(
                f"Episode {episode}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.4f} | "
                f"Eps: {eps:.4f} | "
                f"Steps: {agent.steps_done}"
            )

        # ── Evaluation on dev set ─────────────────────────────────────────
        if dev_windows is not None and episode % eval_interval == 0:
            metrics = evaluate_agent(
                agent, dev_windows, dev_labels,
                agent_type="dqn", device=device
            )
            f1 = metrics["f1_score"]
            logging.info(
                f"  [DEV] F1: {f1:.4f} | "
                f"Sens: {metrics['sensitivity']:.4f} | "
                f"Spec: {metrics['specificity']:.4f} | "
                f"FAR: {metrics['false_alarm_rate']:.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                save_path = os.path.join(config.CHECKPOINT_DIR, "dqn_best.pt")
                agent.save(save_path)
                logging.info(f"  New best F1: {best_f1:.4f} — saved to {save_path}")

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(config.CHECKPOINT_DIR, "dqn_final.pt")
    agent.save(final_path)
    logging.info(f"Training complete. Final model saved to {final_path}")
    logging.info(f"Best dev F1: {best_f1:.4f}")

    return agent
