"""
evaluate.py — Main evaluation entry point for RL-based EEG seizure detection.

Usage:
    python evaluate.py --agent dqn --split dev --checkpoint ./checkpoints/dqn_best.pt
    python evaluate.py --agent ppo --split eval --checkpoint ./checkpoints/ppo_best.pt
    python evaluate.py --agent dqn --dry_run    # Test with synthetic data
    python evaluate.py --agent dqn --split eval --detailed  # Generate plots & Excel

This script:
  1. Loads preprocessed EEG data for the specified split
  2. Loads a trained agent checkpoint
  3. Evaluates the agent and prints comprehensive metrics
  4. (Optional) Generates detailed per-file plots and Excel reports
"""

import argparse
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.utils.helpers import set_seed, get_device, setup_logging
from src.preprocessing.preprocess import (
    preprocess_split, 
    generate_synthetic_data,
    discover_edf_files,
    process_single_file
)
from src.models.dqn_agent import DQNAgent
from src.models.ppo_agent import PPOAgent
from src.models.cnn_supervised import CNNSupervisedAgent
from src.evaluation.evaluate import (
    evaluate_agent, 
    compute_metrics, 
    print_evaluation_report,
    plot_eeg_overlay,
    plot_confusion_matrix,
    export_predictions_excel
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL agent for EEG seizure detection."
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["dqn", "ppo", "cnn"],
        default="dqn",
        help="Agent type: 'dqn', 'ppo', or 'cnn' (default: dqn)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Data split to evaluate on (default: dev)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to agent checkpoint. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.DATA_DIR,
        help=f"Root data directory (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use synthetic data for testing (no real data needed)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed per-file plots and Excel reports",
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
    logging.info("  RL-EEG Seizure Detection — Evaluation")
    logging.info("=" * 60)
    logging.info(f"Agent:      {args.agent.upper()}")
    logging.info(f"Split:      {args.split}")
    logging.info(f"Device:     {device}")
    logging.info(f"Dry Run:    {args.dry_run}")
    logging.info(f"Detailed:   {args.detailed}")

    # ── Load Data ─────────────────────────────────────────────────────────
    if args.dry_run:
        logging.info("Generating synthetic data for dry-run evaluation...")
        eval_windows, eval_labels = generate_synthetic_data(
            n_files=3, duration_sec=60.0, seizure_ratio=0.15
        )
    else:
        split_dir = os.path.join(args.data_dir, "edf", args.split)
        if not os.path.isdir(split_dir):
            logging.error(
                f"Data directory not found: {split_dir}\n"
                f"Please place your TUH dataset under: {args.data_dir}/edf/{args.split}/\n"
                f"Or use --dry_run for testing."
            )
            sys.exit(1)

        logging.info(f"Loading {args.split} data from: {split_dir}")
        eval_windows, eval_labels = preprocess_split(split_dir)

    logging.info(f"Evaluation data: {eval_windows.shape[0]} windows")

    # ── Load Agent ────────────────────────────────────────────────────────
    n_channels = eval_windows.shape[1]
    window_samples = eval_windows.shape[2]

    if args.agent == "dqn":
        agent = DQNAgent(
            n_channels=n_channels,
            window_samples=window_samples,
            device=device,
        )
    elif args.agent == "ppo":
        agent = PPOAgent(
            n_channels=n_channels,
            window_samples=window_samples,
            device=device,
        )
    elif args.agent == "cnn":
        agent = CNNSupervisedAgent(
            n_channels=n_channels,
            window_samples=window_samples,
            device=device,
        )

    # Auto-detect checkpoint
    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            config.CHECKPOINT_DIR, f"{args.agent}_best.pt"
        )
        if not os.path.exists(args.checkpoint):
            args.checkpoint = os.path.join(
                config.CHECKPOINT_DIR, f"{args.agent}_final.pt"
            )

    if os.path.exists(args.checkpoint):
        logging.info(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    else:
        if not args.dry_run:
            logging.warning(
                f"Checkpoint not found: {args.checkpoint}\n"
                "Evaluating with randomly initialized agent."
            )

    # ── Global Evaluation ─────────────────────────────────────────────────
    logging.info("Running global evaluation...")
    metrics = evaluate_agent(
        agent, eval_windows, eval_labels,
        agent_type=args.agent, device=device,
    )

    # Print report
    print_evaluation_report(metrics, split_name=args.split)
    
    # Save global confusion matrix
    if args.detailed or not args.dry_run:
        res_dir = os.path.join("evaluation_results", args.split)
        os.makedirs(res_dir, exist_ok=True)
        cm_path = os.path.join(res_dir, "confusion_matrix.png")
        plot_confusion_matrix(eval_labels, metrics["predictions"], cm_path)

    # ── Detailed Per-File Evaluation ──────────────────────────────────────
    if args.detailed and not args.dry_run:
        logging.info("Generating detailed per-file reports...")
        files = discover_edf_files(split_dir)
        
        for edf_path in files:
            file_name = os.path.splitext(os.path.basename(edf_path))[0]
            logging.info(f"Processing details for: {file_name}")
            
            # Process single file
            result = process_single_file(edf_path)
            if result is None:
                continue
                
            windows, labels = result
            
            # Get predictions
            file_metrics = evaluate_agent(
                agent, windows, labels, 
                agent_type=args.agent, device=device
            )
            preds = file_metrics["predictions"]
            
            # Output paths
            base_out = os.path.join(res_dir, file_name)
            
            # 1. Excel Export (Events)
            export_predictions_excel(labels, preds, f"{base_out}_events.xlsx")
            
            # 2. EEG Overlay Plot
            plot_eeg_overlay(
                windows, labels, preds, 
                f"{base_out}_overlay.png",
                edf_path=edf_path
            )
            
    elif args.detailed and args.dry_run:
        logging.warning("Detailed mode requires real data. Skipping per-file breakdown.")

    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
