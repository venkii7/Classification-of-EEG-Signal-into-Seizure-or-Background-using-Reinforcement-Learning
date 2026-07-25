# RL-EEG-TUH — Module Description & Model Architecture

---

## Module Description

### 1. Preprocessing Module (`src/preprocessing/`)

| File | Role |
|------|------|
| `edf_loader.py` | Loads raw EDF files using MNE-Python, selects 19 standard 10-20 channels (with robust name matching across `-REF`, `-LE`, `-AVG` variants), resamples to 256 Hz, applies bandpass filter (0.5–50 Hz), and z-score normalizes per channel |
| `annotations.py` | Parses TUH CSV annotation files (both `_bi.csv` and regular formats), extracts `(start, end, label)` tuples for seizure/background segments |
| `windowing.py` | Segments continuous EEG into 2-second overlapping windows (50% overlap, stride = 1s). Assigns labels via majority-vote: if ≥50% of a window overlaps with a seizure annotation → seizure (1), else background (0) |
| `preprocess.py` | Orchestrates the full pipeline: discover EDF files → process each → validate channel count → concatenate → cache as `.npz` for fast reloading |

---

### 2. Environment Module (`src/environment/`)

| File | Role |
|------|------|
| `eeg_env.py` | Custom Gymnasium environment modeling seizure detection as sequential decision-making. Each episode = a sequence of ~200 EEG windows. Agent receives one window as observation, outputs action (0=background, 1=seizure), receives asymmetric reward (+3.0 for correct seizure, −3.0 for missed seizure, −1.0 for false alarm, +0.1 for correct background) |

---

### 3. Models Module (`src/models/`)

| File | Role |
|------|------|
| `feature_extractor.py` | Shared 1D-CNN backbone (3 conv blocks + adaptive pooling) that converts raw EEG windows `[19 × 512]` into 128-dim feature vectors. Also includes a CNN-LSTM variant for temporal context |
| `dqn_agent.py` | Deep Q-Network agent with experience replay buffer (50K capacity), ε-greedy exploration with linear decay, Double DQN target network, and Huber loss |
| `ppo_agent.py` | Proximal Policy Optimization agent with Actor-Critic architecture (shared CNN backbone), rollout buffer, Generalized Advantage Estimation (GAE), PPO-Clip objective, and entropy bonus for exploration |

---

### 4. Training Module (`src/training/`)

| File | Role |
|------|------|
| `train_dqn.py` | DQN training loop: per-step experience collection → replay buffer sampling → Double DQN loss → periodic dev evaluation → best F1 checkpointing |
| `train_ppo.py` | PPO training loop: collect 2048-step rollouts → compute GAE → multi-epoch mini-batch PPO-Clip updates → periodic dev evaluation → best F1 checkpointing |

---

### 5. Evaluation Module (`src/evaluation/`)

| File | Role |
|------|------|
| `evaluate.py` | Computes metrics (sensitivity, specificity, F1, FAR), generates clinical-style EEG overlay plots with seizure detections highlighted, exports events to Excel, and produces confusion matrix heatmaps |

---

### 6. Utilities Module (`src/utils/`)

| File | Role |
|------|------|
| `helpers.py` | Provides seed-setting for reproducibility (random, numpy, torch, cuda), GPU device detection with CPU fallback, dual-output logging (console + file), and a timer context manager for profiling |

---

## Model Architecture

### Shared CNN Feature Extractor

```
Input: [Batch, 19 channels, 512 samples]
         │
    ┌────▼────────────────────────────────────┐
    │  Conv1d(19→32, kernel=7) → BN → ReLU   │
    │  MaxPool1d(2)              → [B, 32, 256] │
    ├─────────────────────────────────────────┤
    │  Conv1d(32→64, kernel=5) → BN → ReLU   │
    │  MaxPool1d(2)              → [B, 64, 128] │
    ├─────────────────────────────────────────┤
    │  Conv1d(64→128, kernel=3) → BN → ReLU  │
    │  MaxPool1d(2)              → [B, 128, 64] │
    ├─────────────────────────────────────────┤
    │  AdaptiveAvgPool1d(4)      → [B, 128, 4] │
    │  Flatten                   → [B, 512]     │
    │  FC(512→256) → ReLU → Dropout(0.3)      │
    │  FC(256→128) → ReLU       → [B, 128]     │
    └────┬────────────────────────────────────┘
         │
     128-dim Feature Vector
```

---

### DQN Architecture

```
    EEG Window [19, 512]
         │
    CNN Backbone → 128-dim features
         │
    FC(128→64) → ReLU → Dropout(0.2)
         │
    FC(64→2) → Q-values [Q(s,background), Q(s,seizure)]
         │
    Action = argmax(Q)
```

| Component | Detail |
|-----------|--------|
| Exploration | ε-greedy (ε: 1.0 → 0.05 over 10K steps) |
| Target Network | Hard update every 1000 steps (Double DQN) |
| Loss Function | Huber Loss (SmoothL1) |
| Optimizer | Adam (lr = 1×10⁻⁴) |
| Replay Buffer | 50,000 transitions, batch size = 64 |
| Gradient Clipping | max_norm = 1.0 |

---

### PPO Actor-Critic Architecture

```
    EEG Window [19, 512]
         │
    CNN Backbone → 128-dim features
         │
    ┌────┴────────────┐
    │                 │
  ACTOR            CRITIC
    │                 │
  FC(128→64)       FC(128→64)
  Tanh             Tanh
  FC(64→2)         FC(64→1)
  Softmax            │
    │                 │
  π(a|s)           V(s)
  [P(bg), P(sz)]   State Value
```

| Component | Detail |
|-----------|--------|
| Objective | PPO-Clip (ε = 0.2) |
| Advantage Estimation | GAE (γ = 0.99, λ = 0.95) |
| Entropy Bonus | Coefficient = 0.05 |
| Rollout | 2048 steps → 8 update epochs, batch size 64 |
| Optimizer | Adam (lr = 5×10⁻⁴, eps = 1×10⁻⁵) |
| Gradient Clipping | max_norm = 0.5 |

---

### Reward Structure (Both Agents)

| Condition | Reward | Purpose |
|-----------|--------|---------|
| Correct Seizure (TP) | +3.0 | Incentivize seizure detection |
| Correct Background (TN) | +0.1 | Prevent "always seizure" policy |
| Missed Seizure (FN) | −3.0 | Clinical safety — penalize missed events |
| False Alarm (FP) | −1.0 | Reduce false positive rate |

The 30:1 ratio between seizure reward (+3.0) and background reward (+0.1), combined with symmetric ±3.0 for correct/missed seizure, creates an asymmetric incentive structure calibrated for the ~3% seizure prevalence in the TUH dataset.

---

### Evaluation Metrics

| Metric | Formula | Clinical Meaning |
|--------|---------|------------------|
| Sensitivity | TP / (TP + FN) | Proportion of seizures correctly detected |
| Specificity | TN / (TN + FP) | Proportion of backgrounds correctly identified |
| Precision | TP / (TP + FP) | Of all "seizure" predictions, how many are correct |
| F1-Score | 2 × P × R / (P + R) | Harmonic mean of precision and recall |
| False Alarm Rate | FP / (FP + TN) | Rate of false seizure alerts |
