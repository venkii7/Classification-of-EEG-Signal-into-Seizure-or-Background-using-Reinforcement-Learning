"""
train.py — Main training entry point for RL-based EEG seizure detection.

Usage:
    python train.py --agent dqn --data_dir ./data --epochs 200
    python train.py --agent ppo --data_dir ./data --epochs 200
    python train.py --agent dqn --dry_run --epochs 5     # Test with synthetic data
    python train.py --agent ppo --resume checkpoints/ppo_best.pt --skip_files 200 --max_files 200 --epochs 12000

This script:
  1. Loads or preprocesses EEG data
  2. Creates or loads cached training/dev data
  3. Trains the selected RL agent (DQN or PPO)
  4. Saves checkpoints to ./checkpoints/
"""

import argparse
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.utils.helpers import set_seed, get_device, setup_logging
from src.preprocessing.preprocess import preprocess_split, generate_synthetic_data
from src.training.train_dqn import train_dqn
from src.training.train_ppo import train_ppo
from src.training.train_cnn import train_cnn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an RL agent for EEG seizure detection."
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["dqn", "ppo", "cnn"],
        default="dqn",
        help="Agent to train: 'dqn', 'ppo', or 'cnn' (default: dqn)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.DATA_DIR,
        help=f"Root data directory (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPISODES,
        help=f"Number of training episodes (default: {config.NUM_EPISODES})",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=config.EVAL_INTERVAL,
        help=f"Evaluate every N episodes (default: {config.EVAL_INTERVAL})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use synthetic data for pipeline testing (no real data needed)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Max EDF files to process per split (for debugging)",
    )
    parser.add_argument(
        "--skip_files",
        type=int,
        default=0,
        help="Number of EDF files to skip from the beginning (for incremental training)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (e.g. checkpoints/ppo_best.pt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help=f"Random seed (default: {config.SEED})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Device: 'cuda' or 'cpu' (default: cuda)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    setup_logging(config.LOG_DIR)
    set_seed(args.seed)
    device = get_device(args.device)

    logging.info("=" * 60)
    logging.info("  RL-EEG Seizure Detection — Training")
    logging.info("=" * 60)
    logging.info(f"Agent:      {args.agent.upper()}")
    logging.info(f"Episodes:   {args.epochs}")
    logging.info(f"Seed:       {args.seed}")
    logging.info(f"Device:     {device}")
    logging.info(f"Dry Run:    {args.dry_run}")
    if args.resume:
        logging.info(f"Resume:     {args.resume}")
    if args.skip_files > 0:
        logging.info(f"Skip Files: {args.skip_files}")

    # ── Load Data ─────────────────────────────────────────────────────────
    if args.dry_run:
        logging.info("Generating synthetic data for dry-run testing...")
        train_windows, train_labels = generate_synthetic_data(
            n_files=10, duration_sec=60.0, seizure_ratio=0.15
        )
        dev_windows, dev_labels = generate_synthetic_data(
            n_files=3, duration_sec=60.0, seizure_ratio=0.15
        )
    else:
        train_dir = os.path.join(args.data_dir, "edf", "train")
        dev_dir = os.path.join(args.data_dir, "edf", "dev")

        if not os.path.isdir(train_dir):
            logging.error(
                f"Training data directory not found: {train_dir}\n"
                f"Please place your TUH dataset under: {args.data_dir}/edf/train/\n"
                f"Or use --dry_run for testing with synthetic data."
            )
            sys.exit(1)

        logging.info(f"Preprocessing training data from: {train_dir}")
        train_windows, train_labels = preprocess_split(
            train_dir, max_files=args.max_files, skip_files=args.skip_files
        )

        if args.agent == "cnn":
            # CNN trains without dev validation — test separately with evaluate.py
            logging.info("CNN mode: skipping dev set preprocessing")
            dev_windows, dev_labels = None, None
        elif os.path.isdir(dev_dir):
            logging.info(f"Preprocessing dev data from: {dev_dir}")
            dev_windows, dev_labels = preprocess_split(
                dev_dir, max_files=args.max_files
            )
        else:
            logging.warning(f"Dev directory not found: {dev_dir}. Skipping dev evaluation.")
            dev_windows, dev_labels = None, None

    logging.info(f"Training data: {train_windows.shape[0]} windows")
    if dev_windows is not None:
        logging.info(f"Dev data:      {dev_windows.shape[0]} windows")

    # ── Train ─────────────────────────────────────────────────────────────
    if args.agent == "dqn":
        agent = train_dqn(
            train_windows, train_labels,
            dev_windows, dev_labels,
            num_episodes=args.epochs,
            eval_interval=args.eval_interval,
            device=device,
            resume_path=args.resume,
        )
    elif args.agent == "ppo":
        agent = train_ppo(
            train_windows, train_labels,
            dev_windows, dev_labels,
            num_episodes=args.epochs,
            eval_interval=args.eval_interval,
            device=device,
            resume_path=args.resume,
        )
    elif args.agent == "cnn":
        agent = train_cnn(
            train_windows, train_labels,
            dev_windows, dev_labels,
            num_epochs=args.epochs,
            device=device,
            resume_path=args.resume,
        )

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
