"""
train_ppo.py — PPO training loop for EEG seizure detection.

Trains the PPO agent by collecting rollouts from the environment:
  - Collect N steps of experience → compute GAE → PPO update for K epochs
  - Logs rewards, losses, and policy statistics
  - Periodically evaluates on dev set and checkpoints best model
"""

import os
import logging
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from src.environment.eeg_env import EEGSeizureEnv
from src.models.ppo_agent import PPOAgent
from src.evaluation.evaluate import evaluate_agent


def train_ppo(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    dev_windows: Optional[np.ndarray] = None,
    dev_labels: Optional[np.ndarray] = None,
    num_episodes: int = config.NUM_EPISODES,
    eval_interval: int = config.EVAL_INTERVAL,
    rollout_steps: int = config.PPO_ROLLOUT_STEPS,
    device: Optional[torch.device] = None,
    resume_path: Optional[str] = None,
) -> PPOAgent:
    """
    Train a PPO agent on preprocessed EEG data.

    Args:
        train_windows: Training windows, shape (N, n_channels, window_samples).
        train_labels:  Training labels, shape (N,).
        dev_windows:   Dev set windows for evaluation.
        dev_labels:    Dev set labels for evaluation.
        num_episodes:  Number of training episodes (used as outer loop).
        eval_interval: Evaluate every N episodes.
        rollout_steps: Number of steps per rollout collection.
        device:        Torch device.
        resume_path:   Path to checkpoint to resume training from.

    Returns:
        Trained PPOAgent.
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
    agent = PPOAgent(
        n_channels=n_channels,
        window_samples=window_samples,
        device=device,
    )

    # ── Resume from checkpoint if provided ────────────────────────────────
    if resume_path is not None:
        logging.info(f"Resuming PPO training from: {resume_path}")
        agent.load(resume_path)

    # ── Tracking ──────────────────────────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_f1 = 0.0

    # If resuming, evaluate first to establish baseline F1
    # (so we only save a new "best" if training actually improves)
    if resume_path is not None and dev_windows is not None:
        logging.info("Evaluating resumed model to establish baseline F1...")
        baseline_metrics = evaluate_agent(
            agent, dev_windows, dev_labels,
            agent_type="ppo", device=device
        )
        best_f1 = baseline_metrics["f1_score"]
        logging.info(f"Baseline F1 from resumed model: {best_f1:.4f}")

    # Initialize environment
    state, info = train_env.reset()
    total_episodes_completed = 0
    episode_reward = 0.0

    # ── Training loop ─────────────────────────────────────────────────────
    pbar = tqdm(total=num_episodes, desc="PPO Training (episodes)")

    while total_episodes_completed < num_episodes:
        # ── Collect rollout ───────────────────────────────────────────────
        agent.rollout_buffer.clear()

        for step in range(rollout_steps):
            action, log_prob, value = agent.select_action(state, training=True)

            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, log_prob, reward, float(done), value)
            episode_reward += reward

            if done:
                total_episodes_completed += 1
                pbar.update(1)

                acc = info.get("episode_accuracy", 0.0)
                if total_episodes_completed % 5 == 0:
                    logging.info(
                        f"Episode {total_episodes_completed}/{num_episodes} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Acc: {acc:.4f} | "
                        f"Steps: {agent.total_steps}"
                    )

                # ── Evaluation on dev set ─────────────────────────────────
                if dev_windows is not None and total_episodes_completed % eval_interval == 0:
                    metrics = evaluate_agent(
                        agent, dev_windows, dev_labels,
                        agent_type="ppo", device=device
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
                        save_path = os.path.join(config.CHECKPOINT_DIR, "ppo_best.pt")
                        agent.save(save_path)
                        logging.info(f"  New best F1: {best_f1:.4f} — saved to {save_path}")

                episode_reward = 0.0
                state, info = train_env.reset()

                if total_episodes_completed >= num_episodes:
                    break
            else:
                state = next_state

        # ── PPO update ────────────────────────────────────────────────────
        loss_info = agent.update()
        if total_episodes_completed % 5 == 0:
            logging.info(
                f"  PPO Update | "
                f"Policy Loss: {loss_info['policy_loss']:.4f} | "
                f"Value Loss: {loss_info['value_loss']:.4f} | "
                f"Entropy: {loss_info['entropy']:.4f}"
            )

    pbar.close()

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(config.CHECKPOINT_DIR, "ppo_final.pt")
    agent.save(final_path)
    logging.info(f"Training complete. Final model saved to {final_path}")
    logging.info(f"Best dev F1: {best_f1:.4f}")

    return agent
