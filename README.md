# Seizure Stage Detection Using Reinforcement Learning from EEG

A complete end-to-end system that models EEG seizure detection as a
**sequential decision-making problem** using reinforcement learning.

Built for the **Temple University Hospital EEG Seizure Corpus (TUH v2.0.3)**.

---

## RL Formulation

| Concept       | Definition                                                    |
|---------------|---------------------------------------------------------------|
| **Environment** | A stream of consecutive EEG sliding windows from one recording |
| **State**       | Multi-channel EEG window (19 channels × 512 samples)          |
| **Action**      | `0` = background, `1` = seizure                               |
| **Reward**      | Asymmetric: +1.0 correct seizure, +0.5 correct background, −2.0 missed seizure, −0.5 false alarm |
| **Episode**     | One complete EEG file processed window-by-window               |

---

## Project Structure

```
RL-EEG-TUH/
├── config.py                   # Hyperparameters & paths
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
├── requirements.txt
├── data/                       # Place TUH dataset here
│   └── edf/
│       ├── train/
│       ├── dev/
│       └── eval/
├── src/
│   ├── preprocessing/
│   │   ├── edf_loader.py       # Load & filter EDF files (MNE)
│   │   ├── annotations.py      # Parse TUH CSV annotations
│   │   ├── windowing.py        # Sliding windows + label alignment
│   │   └── preprocess.py       # Full pipeline + caching + synthetic data
│   ├── environment/
│   │   └── eeg_env.py          # Custom Gymnasium environment
│   ├── models/
│   │   ├── feature_extractor.py  # CNN & CNN-LSTM backbones
│   │   ├── dqn_agent.py         # Double DQN with replay buffer
│   │   └── ppo_agent.py         # PPO with Actor-Critic + GAE
│   ├── training/
│   │   ├── train_dqn.py        # DQN training loop
│   │   └── train_ppo.py        # PPO training loop
│   ├── evaluation/
│   │   └── evaluate.py         # Metrics & evaluation pipeline
│   └── utils/
│       └── helpers.py          # Seeds, logging, device management
├── checkpoints/                # Auto-created for model saves
├── cache/                      # Auto-created for preprocessed data
└── logs/                       # Auto-created for training logs
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test with Synthetic Data (No Dataset Needed)

```bash
# Train DQN agent on synthetic data (pipeline test)
python train.py --agent dqn --dry_run --epochs 5

# Train PPO agent on synthetic data
python train.py --agent ppo --dry_run --epochs 5

# Evaluate on synthetic data
python evaluate.py --agent dqn --dry_run
```

### 3. Train on TUH Dataset

Place the TUH EEG Seizure Corpus under `data/`:

```
data/
└── edf/
    ├── train/   (TUH training split)
    ├── dev/     (TUH dev split)
    └── eval/    (TUH eval split)
```

Then run:

```bash
# Train DQN
python train.py --agent dqn --data_dir ./data --epochs 200

# Train PPO
python train.py --agent ppo --data_dir ./data --epochs 200

# Evaluate on dev set
python evaluate.py --agent dqn --split dev

# Evaluate on eval set
python evaluate.py --agent ppo --split eval --checkpoint ./checkpoints/ppo_best.pt
```

---

## Architecture

### Feature Extractors

- **CNNFeatureExtractor**: 3 × [Conv1d → BatchNorm → ReLU → MaxPool] → AdaptiveAvgPool → MLP → 128-d
- **CNNLSTMFeatureExtractor**: Same CNN + LSTM layer for temporal context across windows

### RL Agents

| Agent | Algorithm | Key Features |
|-------|-----------|-------------|
| **DQN** | Double DQN | Replay buffer, ε-greedy decay, target network, Huber loss |
| **PPO** | PPO-Clip | Actor-Critic, GAE, entropy bonus, gradient clipping |

### Class Imbalance Handling

Seizure events are rare (~5–10% of data). We handle this through:
- **Asymmetric rewards**: Missed seizures penalized 4× more than false alarms
- **Evaluation metrics**: Focus on sensitivity and F1-score rather than just accuracy

---

## Evaluation Metrics

| Metric | Formula | Why It Matters |
|--------|---------|---------------|
| **Sensitivity** | TP / (TP + FN) | How many seizures are detected |
| **Specificity** | TN / (TN + FP) | How many background segments are correctly classified |
| **F1-Score** | 2·P·R / (P+R) | Balance between precision and recall |
| **False Alarm Rate** | FP / (FP + TN) | How often the system raises a false alarm |

---

## Configuration

All hyperparameters are centralized in `config.py`:

- **EEG**: 19 channels, 256 Hz, 0.5–50 Hz bandpass, 2s windows, 1s stride
- **DQN**: LR=1e-4, γ=0.99, ε decay over 10K steps, buffer=50K, batch=64
- **PPO**: LR=3e-4, γ=0.99, λ=0.95, clip=0.2, 4 epochs/update, batch=64

---

## Reproducibility

- Fixed random seeds (`set_seed(42)`) for NumPy, PyTorch, Python, CUDA
- Deterministic CUDNN
- Cached preprocessed data (`.npz` files)
- Full logging to `logs/training.log`

---

## Research Extensions

1. **DRQN**: Replace CNN with CNN-LSTM in DQN for full temporal modeling
2. **Multi-Agent RL**: Separate agents for different brain regions
3. **Curriculum Learning**: Train on easy-to-detect seizures first
4. **Seizure Subtype Classification**: Extend to focal vs. generalized
5. **Online Learning**: Adapt the agent to patient-specific patterns
6. **Attention Mechanisms**: Add spatial attention for channel importance
7. **Transformer Backbone**: Replace LSTM with temporal transformer encoder

---

## Citation

If you use this project, please cite the TUH EEG Corpus:

```
Shah, V., et al. (2018). The Temple University Hospital Seizure Detection Corpus.
Frontiers in Neuroinformatics, 12, 83.
```

---

## License

MIT License — for academic and research use.
