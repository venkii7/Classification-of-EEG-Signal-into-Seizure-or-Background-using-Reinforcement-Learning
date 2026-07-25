"""
config.py — Central configuration for the RL-EEG Seizure Detection project.

All hyperparameters, paths, and constants are defined here for easy tuning
and reproducibility across experiments.
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# TUH dataset splits — the user should symlink or copy their data here
TRAIN_DIR = os.path.join(DATA_DIR, "edf", "train")
DEV_DIR = os.path.join(DATA_DIR, "edf", "dev")
EVAL_DIR = os.path.join(DATA_DIR, "edf", "eval")

# ──────────────────────────────────────────────────────────────────────────────
# EEG Signal Parameters
# ──────────────────────────────────────────────────────────────────────────────
# Standard 10-20 montage channels used in TUH corpus (TCP montage, 19 channels)
EEG_CHANNELS = [
    "EEG FP1-REF", "EEG FP2-REF",
    "EEG F3-REF",  "EEG F4-REF",
    "EEG C3-REF",  "EEG C4-REF",
    "EEG P3-REF",  "EEG P4-REF",
    "EEG O1-REF",  "EEG O2-REF",
    "EEG F7-REF",  "EEG F8-REF",
    "EEG T3-REF",  "EEG T4-REF",
    "EEG T5-REF",  "EEG T6-REF",
    "EEG FZ-REF",  "EEG CZ-REF",
    "EEG PZ-REF",
]

NUM_CHANNELS = len(EEG_CHANNELS)        # 19
TARGET_SFREQ = 256                       # Resample to 256 Hz
BANDPASS_LOW = 0.5                       # High-pass cutoff (Hz)
BANDPASS_HIGH = 50.0                     # Low-pass cutoff  (Hz)

# ──────────────────────────────────────────────────────────────────────────────
# Sliding Window Parameters
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SEC = 2.0                         # Window length in seconds
STRIDE_SEC = 1.0                         # Stride in seconds (50% overlap)
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SFREQ)   # 512 samples
STRIDE_SAMPLES = int(STRIDE_SEC * TARGET_SFREQ)   # 256 samples

# ──────────────────────────────────────────────────────────────────────────────
# Label Mapping
# ──────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "bckg": 0,
    "seiz": 1,
}
NUM_CLASSES = 2

# ──────────────────────────────────────────────────────────────────────────────
# RL Environment — Reward Shaping (rebalanced for ~88%/12% class imbalance)
# ──────────────────────────────────────────────────────────────────────────────
REWARD_CORRECT_SEIZ = 5.0                # High reward — seizures are rare, correct detection is critical
REWARD_CORRECT_BCKG = 0.3                # Low reward — background is abundant, avoid trivial "all-bckg" policy
REWARD_MISS_SEIZ    = -5.0               # Severe penalty — missed seizures endanger patient safety
REWARD_FALSE_ALARM  = -1.5               # Moderate penalty — false alarms cause alarm fatigue but less critical

# ──────────────────────────────────────────────────────────────────────────────
# DQN Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
DQN_LR = 1e-4                           # Learning rate
DQN_GAMMA = 0.99                        # Discount factor
DQN_EPSILON_START = 1.0                  # Initial exploration rate
DQN_EPSILON_END = 0.05                   # Final exploration rate
DQN_EPSILON_DECAY = 10000                # Steps over which epsilon decays linearly
DQN_BUFFER_SIZE = 50000                  # Replay buffer capacity
DQN_BATCH_SIZE = 64                      # Mini-batch size for training
DQN_TARGET_UPDATE = 1000                 # Steps between target network syncs
DQN_FEATURE_DIM = 128                    # Feature extractor output dim

# ──────────────────────────────────────────────────────────────────────────────
# PPO Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
PPO_LR = 3e-4                            # Reduced LR for stable learning (was 5e-4)
PPO_GAMMA = 0.99                         # Discount factor
PPO_GAE_LAMBDA = 0.95                    # GAE lambda
PPO_CLIP_EPS = 0.2                       # PPO clipping epsilon
PPO_ENTROPY_COEF = 0.15                  # High entropy to prevent premature collapse (was 0.05)
PPO_VALUE_COEF = 0.5                     # Value loss coefficient
PPO_MAX_GRAD_NORM = 0.5                  # Gradient clipping
PPO_ROLLOUT_STEPS = 2048                 # Steps per rollout
PPO_NUM_EPOCHS = 8                       # More epochs per update (was 4)
PPO_BATCH_SIZE = 64                      # Mini-batch size
PPO_FEATURE_DIM = 128                    # Feature extractor output dim

# ──────────────────────────────────────────────────────────────────────────────
# CNN Supervised Classifier Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
CNN_LR = 1e-4                            # Learning rate for Adam optimizer
CNN_ADAM_EPS = 1e-5                      # Adam epsilon
CNN_EPOCHS = 200                         # Maximum training epochs
CNN_BATCH_SIZE = 64                      # Mini-batch size
CNN_PATIENCE = 20                        # Early stopping patience (dev F1)
CNN_DROPOUT = 0.3                        # Dropout rate in classification head
CNN_MAX_GRAD_NORM = 0.5                  # Gradient clipping max norm
CNN_CLASS_WEIGHT = True                  # Use inverse-frequency class weighting
CNN_OVERSAMPLE = True                    # Duplicate seizure windows to balance classes


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
NUM_EPISODES = 200                       # Total training episodes
EVAL_INTERVAL = 10                       # Evaluate on dev set every N episodes
SEED = 42                                # Random seed for reproducibility
DEVICE = "cuda"                          # "cuda" or "cpu" — auto-detected at runtime
