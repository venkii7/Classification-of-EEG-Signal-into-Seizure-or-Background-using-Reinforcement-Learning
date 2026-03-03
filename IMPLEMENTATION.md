# Reinforcement Learning-Based Automated Seizure Detection from Scalp EEG Signals: Implementation Details

---

## 1. Dataset

### 1.1 TUH EEG Seizure Corpus (TUSZ)

The experiments in this work were conducted using the **Temple University Hospital (TUH) EEG Seizure Corpus**, one of the largest publicly available annotated EEG datasets for seizure detection research. The corpus comprises clinical EEG recordings stored in the European Data Format (EDF), collected during routine clinical practice at the Temple University Hospital. Each recording is accompanied by time-stamped annotation files in CSV format that delineate seizure (`seiz`) and background (`bckg`) events, provided by board-certified neurologists.

### 1.2 Data Organization

The dataset is organized into three non-overlapping partitions following the standard TUH split convention:

| Split          | Purpose                              |
|----------------|--------------------------------------|
| **Train**      | Model training and parameter optimization |
| **Dev** (Validation)  | Hyperparameter tuning and model selection |
| **Eval** (Test)       | Final performance evaluation         |

Each split follows a hierarchical directory structure containing EDF recording files with their corresponding annotation CSV files. The annotations utilize the `_bi.csv` binary format with columns: `channel`, `start_time`, `stop_time`, `label`, and `probability`.

### 1.3 Channel Configuration

A standardized set of **19 EEG channels** from the International 10–20 system was used across all recordings, employing the referential (REF) montage:

| Region          | Channels                              |
|-----------------|---------------------------------------|
| Frontal-Polar   | FP1, FP2                             |
| Frontal         | F3, F4, F7, F8, FZ                   |
| Central         | C3, C4, CZ                           |
| Parietal        | P3, P4, PZ                           |
| Occipital       | O1, O2                               |
| Temporal        | T3, T4, T5, T6                       |

A robust channel-matching mechanism was implemented to handle the heterogeneity in channel naming conventions across TUH recordings (e.g., `-REF`, `-LE`, `-AVG`, `-AR` suffixes). Channels are normalized to a canonical form for matching, and recordings with fewer than the required 19 channels are handled gracefully by the system.

---

## 2. Preprocessing

### 2.1 Signal Loading and Filtering

Raw EEG signals were loaded from EDF files using the **MNE-Python** library. The preprocessing pipeline applies the following transformations sequentially:

1. **Channel Selection**: The 19 standard 10–20 channels are extracted from each recording using the robust matching strategy described above. Channels are reordered to maintain consistent spatial arrangement across all recordings—a critical requirement for the convolutional neural network's spatial filters.

2. **Resampling**: All recordings are resampled to a uniform sampling frequency of **256 Hz** to ensure temporal consistency, irrespective of the original recording sampling rate.

3. **Bandpass Filtering**: A finite impulse response (FIR) bandpass filter is applied with cutoff frequencies of **0.5 Hz** (high-pass) and **50.0 Hz** (low-pass). The high-pass filter removes slow baseline drift, while the low-pass filter attenuates high-frequency muscle artifacts and power-line interference.

4. **Z-Score Normalization**: Each channel is independently normalized to zero mean and unit variance using channel-wise z-score normalization:

$$
x_{norm}^{(c)} = \frac{x^{(c)} - \mu^{(c)}}{\sigma^{(c)}}
$$

where $\mu^{(c)}$ and $\sigma^{(c)}$ are the mean and standard deviation of channel $c$ across all time samples in the recording. A guard condition prevents division by zero by setting $\sigma^{(c)} = 1.0$ when the standard deviation equals zero.

### 2.2 Sliding Window Segmentation

The continuous, preprocessed EEG signal is segmented into fixed-length overlapping windows for frame-level classification:

| Parameter       | Value       |
|-----------------|-------------|
| Window Length   | 2.0 seconds (512 samples at 256 Hz) |
| Stride          | 1.0 seconds (256 samples at 256 Hz) |
| Overlap         | 50%         |

Each window produces a tensor of shape **(19 × 512)**, representing 19 channels × 512 time-domain samples.

### 2.3 Label Assignment

Labels are assigned to each window using a **majority-vote strategy**. For each window, the temporal overlap with seizure annotations is computed. A window is labeled as seizure (`1`) if **≥ 50%** of its duration overlaps with a seizure annotation interval; otherwise, it is labeled as background (`0`).

Formally, for a window spanning time interval $[t_s, t_e]$:

$$
\text{label} = 
\begin{cases}
1 \; (\text{seizure}), & \text{if } \sum_{k} \text{overlap}([t_s, t_e], [a_k^s, a_k^e]) \geq 0.5 \times (t_e - t_s) \\
0 \; (\text{background}), & \text{otherwise}
\end{cases}
$$

where $[a_k^s, a_k^e]$ denotes the $k$-th seizure annotation interval.

### 2.4 Data Caching

Preprocessed windows and labels are serialized to compressed NumPy archive format (`.npz`) for each data split. Subsequent training and evaluation runs directly load the cached data, eliminating redundant preprocessing overhead and significantly accelerating the experimental pipeline.

---

## 3. Reinforcement Learning Formulation

### 3.1 Problem Formulation as a Markov Decision Process

Seizure detection is formulated as a **Markov Decision Process (MDP)** where an agent sequentially classifies EEG windows in a streaming fashion, mimicking real-time clinical monitoring. The MDP is defined by the tuple $(S, A, R, P, \gamma)$:

- **State Space ($S$)**: Each state is a preprocessed EEG window of shape $(19, 512)$, representing a 2-second segment of 19-channel EEG data.

- **Action Space ($A$)**: A discrete action space with two actions:
  - $a = 0$: Classify the current window as **background** (non-seizure)
  - $a = 1$: Classify the current window as **seizure**

- **Transition ($P$)**: The environment transitions deterministically to the next chronological window in the recording upon taking an action. Each EEG recording constitutes one episode, during which the agent traverses all windows sequentially.

- **Discount Factor ($\gamma$)**: Set to $0.99$, encouraging the agent to consider long-term consequences of its classification decisions within an episode.

### 3.2 Reward Shaping

An **asymmetric reward function** is carefully designed to address the severe class imbalance inherent in seizure detection (seizure events constitute a small minority of the recording) and to encode clinical safety priorities:

| Event                        | Reward    | Rationale                                  |
|------------------------------|-----------|--------------------------------------------|
| Correct Seizure Detection (TP)    | +1.2  | Moderate positive reinforcement            |
| Correct Background Detection (TN)| +0.8  | Positive reinforcement for normal operation|
| Missed Seizure (FN)              | −2.5  | Heavy penalty — clinical safety concern    |
| False Alarm (FP)                 | −3.0  | Heaviest penalty — alarm fatigue risk      |

The reward structure reflects the competing clinical objectives: **missed seizures** pose a direct threat to patient safety, while **excessive false alarms** lead to alarm fatigue among clinical staff, potentially causing real alarms to be ignored. The penalty magnitudes were calibrated through iterative experimentation to balance sensitivity and specificity.

### 3.3 Episode Structure

When per-file boundaries are unavailable, windows are segmented into episodes of approximately 200 windows each to keep episodes manageable for training. Within each episode, windows are presented sequentially, and the episode terminates when all windows have been classified. File order is randomized across training episodes to prevent overfitting to specific recording sequences.

---

## 4. Feature Extraction Network

### 4.1 CNN Feature Extractor

A **one-dimensional Convolutional Neural Network (1D-CNN)** serves as the shared feature backbone for both RL agents. The network transforms raw EEG windows into compact, discriminative feature vectors.

#### Architecture

The CNN consists of **three convolutional blocks**, each comprising a 1D convolution, batch normalization, ReLU activation, and max-pooling, followed by an adaptive pooling layer and a fully connected head:

```
Input: (batch, 19, 512)
       │
       ├─ Block 1: Conv1d(19 → 32, k=7, pad=3) → BN(32) → ReLU → MaxPool(2)
       │  Output: (batch, 32, 256)
       │
       ├─ Block 2: Conv1d(32 → 64, k=5, pad=2) → BN(64) → ReLU → MaxPool(2)
       │  Output: (batch, 64, 128)
       │
       ├─ Block 3: Conv1d(64 → 128, k=3, pad=1) → BN(128) → ReLU → MaxPool(2)
       │  Output: (batch, 128, 64)
       │
       ├─ AdaptiveAvgPool1d(4)
       │  Output: (batch, 128, 4)
       │
       ├─ Flatten → (batch, 512)
       │
       ├─ FC: Linear(512 → 256) → ReLU → Dropout(0.3) → Linear(256 → 128) → ReLU
       │
       Output: (batch, 128)
```

**Design rationale:**
- **Decreasing kernel sizes** (7 → 5 → 3) across blocks allow the network to capture both broad temporal patterns (e.g., rhythmic activity) and fine-grained morphological features (e.g., spike-wave complexes).
- **Batch normalization** stabilizes training and accelerates convergence.
- **Adaptive average pooling** to a fixed size of 4 decouples the feature extractor from the exact input length, providing flexibility for varying window sizes.
- **Dropout (0.3)** in the fully connected head serves as a regularizer to mitigate overfitting.
- The output feature dimension is **128**, providing a compact yet sufficiently expressive representation for the downstream RL policy heads.

### 4.2 CNN-LSTM Feature Extractor (Auxiliary)

An optional **CNN-LSTM** variant extends the CNN backbone by processing sequences of extracted features through a Long Short-Term Memory (LSTM) network. This architecture captures **temporal dependencies** across consecutive EEG windows within an episode, enabling the model to leverage contextual information from preceding windows for the current classification decision.

| Component       | Configuration                         |
|-----------------|---------------------------------------|
| CNN Backbone    | Shared `CNNFeatureExtractor` (as above) |
| LSTM            | Input: 128, Hidden: 128, Layers: 1   |
| Output FC       | Linear(128 → 128) → ReLU             |

---

## 5. RL Agent Architectures

### 5.1 Proximal Policy Optimization (PPO) Agent

The PPO agent implements the **PPO-Clip** algorithm with an **Actor-Critic** architecture sharing the CNN feature backbone.

#### 5.1.1 Network Architecture

```
Input EEG Window: (19, 512)
       │
       ├─ Shared CNN Feature Extractor → (128,)
       │
       ├── Actor Head (Policy):                    ├── Critic Head (Value):
       │   Linear(128 → 64) → Tanh                │   Linear(128 → 64) → Tanh
       │   Linear(64 → 2) → Softmax               │   Linear(64 → 1)
       │   Output: π(a|s) ∈ ℝ²                     │   Output: V(s) ∈ ℝ
```

#### 5.1.2 Training Algorithm

The PPO agent employs the following training procedure:

1. **Rollout Collection**: The agent collects $N = 2048$ steps of experience by interacting with the environment, storing states, actions, log-probabilities, rewards, dones, and value estimates in a rollout buffer.

2. **Advantage Estimation**: At the end of each rollout, Generalized Advantage Estimation (GAE) is computed:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the temporal difference error, $\gamma = 0.99$ is the discount factor, and $\lambda = 0.95$ is the GAE smoothing parameter. Advantages are normalized to zero mean and unit variance before the update step.

3. **PPO-Clip Objective**: The policy is updated over $K = 8$ epochs using mini-batches of size 64, optimizing:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio and $\epsilon = 0.2$ is the clipping parameter.

4. **Total Loss**: The combined loss function is:

$$
L = L^{CLIP} + c_1 \cdot L^{VF} + c_2 \cdot L^{ENT}
$$

where $L^{VF}$ is the mean squared error value loss ($c_1 = 0.5$), and $L^{ENT}$ is the entropy bonus ($c_2 = 0.05$) to encourage exploration.

#### 5.1.3 Hyperparameters

| Hyperparameter      | Value   |
|---------------------|---------|
| Learning Rate       | 5 × 10⁻⁴ |
| Discount Factor (γ) | 0.99    |
| GAE Lambda (λ)      | 0.95    |
| Clip Epsilon (ε)    | 0.2     |
| Entropy Coefficient | 0.05    |
| Value Coefficient   | 0.5     |
| Max Gradient Norm   | 0.5     |
| Rollout Steps       | 2048    |
| PPO Update Epochs   | 8       |
| Mini-batch Size     | 64      |
| Feature Dimension   | 128     |
| Optimizer           | Adam (ε = 10⁻⁵) |

---

### 5.2 Deep Q-Network (DQN) Agent

The DQN agent implements the **Double DQN** algorithm with experience replay and ε-greedy exploration.

#### 5.2.1 Network Architecture

```
Input EEG Window: (19, 512)
       │
       ├─ CNN Feature Extractor → (128,)
       │
       ├─ Q-Network Head:
       │   Linear(128 → 64) → ReLU → Dropout(0.2) → Linear(64 → 2)
       │   Output: Q(s, a) ∈ ℝ²
```

Two copies of this network are maintained: the **online network** (continuously updated) and the **target network** (periodically synchronized).

#### 5.2.2 Training Algorithm

1. **Experience Replay**: Transitions $(s_t, a_t, r_t, s_{t+1}, done)$ are stored in a circular replay buffer of capacity 50,000. Mini-batches of 64 transitions are uniformly sampled for training.

2. **ε-Greedy Exploration**: Actions are selected greedily with probability $(1 - \varepsilon)$ and randomly with probability $\varepsilon$. The exploration rate decays linearly from $\varepsilon = 1.0$ to $\varepsilon = 0.05$ over the first 10,000 steps.

3. **Double DQN Update**: To mitigate overestimation bias, the online network selects the best action while the target network evaluates its Q-value:

$$
y_t = r_t + \gamma \cdot Q_{\theta^-} \left( s_{t+1}, \; \arg\max_{a'} Q_\theta(s_{t+1}, a') \right) \cdot (1 - done_t)
$$

4. **Loss Function**: The **Huber loss** (Smooth L1) is used for stable optimization:

$$
L = \text{SmoothL1}(Q_\theta(s_t, a_t), \; y_t)
$$

5. **Target Network Sync**: The target network parameters $\theta^-$ are hard-copied from the online network $\theta$ every 1,000 update steps.

#### 5.2.3 Hyperparameters

| Hyperparameter         | Value    |
|------------------------|----------|
| Learning Rate          | 1 × 10⁻⁴ |
| Discount Factor (γ)    | 0.99     |
| Epsilon Start          | 1.0      |
| Epsilon End            | 0.05     |
| Epsilon Decay Steps    | 10,000   |
| Replay Buffer Size     | 50,000   |
| Mini-batch Size        | 64       |
| Target Update Interval | 1,000 steps |
| Feature Dimension      | 128      |
| Gradient Clip Norm     | 1.0      |
| Optimizer              | Adam     |
| Loss Function          | Smooth L1 (Huber) |

---

## 6. Training Procedure

### 6.1 Experimental Setup

| Parameter          | Value                 |
|--------------------|-----------------------|
| Random Seed        | 42                    |
| Framework          | PyTorch               |
| Compute Device     | CUDA (when available) |
| Training Episodes  | Configurable (default: 200; extended runs: 12,000) |
| Evaluation Interval| Every 10 episodes     |

### 6.2 PPO Training Loop

The PPO training loop operates as a rollout-based procedure:

1. **Initialize** the environment with shuffled training data and the Actor-Critic network.
2. **Collect Rollout**: Interact with the environment for 2,048 steps, storing the full trajectory.
3. **Compute Advantages**: At the end of each rollout, compute GAE with the value of the terminal state.
4. **PPO Update**: Perform 8 epochs of mini-batch gradient descent on the collected rollout.
5. **Evaluation**: Every 10 completed episodes, evaluate the agent on the held-out development set.
6. **Model Selection**: The best model (by F1-score on the development set) is checkpointed, along with the final model after training completion.

### 6.3 DQN Training Loop

The DQN training loop operates in an episodic fashion:

1. **Initialize** the environment, Q-network, target network, and replay buffer.
2. **For each episode**:
   - Reset the environment to a new recording
   - Step through all windows, selecting actions with ε-greedy policy
   - Store transitions in the replay buffer
   - Perform gradient descent on sampled mini-batches after each step (when buffer is sufficiently full)
3. **Target Sync**: Synchronize the target network with the online network every 1,000 updates.
4. **Evaluation and Checkpointing**: Same protocol as PPO.

### 6.4 Resume Training (Incremental Learning)

Both agents support **incremental training** through checkpoint resumption. This enables:
- Training on subsets of data (e.g., first 200 files) and progressively scaling up
- Fine-tuning on additional data without restarting from scratch
- Efficient experimentation with different dataset sizes

---

## 7. Evaluation and Testing

### 7.1 Evaluation Metrics

The evaluation framework computes the following comprehensive set of metrics, specifically chosen for their relevance to clinical seizure detection:

| Metric              | Definition                               | Clinical Relevance                         |
|---------------------|------------------------------------------|--------------------------------------------|
| **Sensitivity** (Recall) | $\frac{TP}{TP + FN}$                | Ability to detect seizure events           |
| **Specificity**     | $\frac{TN}{TN + FP}$                    | Ability to correctly identify background   |
| **Precision**       | $\frac{TP}{TP + FP}$                    | Reliability of seizure detections          |
| **F1-Score**        | $\frac{2 \cdot Prec \cdot Sens}{Prec + Sens}$ | Harmonic mean of precision and sensitivity |
| **False Alarm Rate**| $\frac{FP}{FP + TN}$                    | Rate of false seizure detections           |
| **Accuracy**        | $\frac{TP + TN}{TP + TN + FP + FN}$     | Overall classification performance         |

where $TP$, $TN$, $FP$, and $FN$ denote true positives, true negatives, false positives, and false negatives, respectively.

### 7.2 Evaluation Protocol

1. **Global Evaluation**: The trained agent is evaluated in **inference mode** (greedy action selection for PPO, greedy Q-value selection for DQN) on the entire development or evaluation split. Window-level predictions are collected and compared against ground-truth labels.

2. **Confusion Matrix**: A $2 \times 2$ confusion matrix is computed and visualized as a heatmap to provide a comprehensive overview of model behavior across both classes.

3. **Per-File Analysis** (Detailed Mode): In detailed evaluation mode, each EDF recording is processed individually:
   - **Clinical EEG Overlay Plots**: Multi-channel EEG traces are plotted with seizure detections overlaid as colored regions, mimicking clinical EEG review software.
   - **Excel Event Reports**: Predicted and ground-truth seizure events are exported to structured Excel spreadsheets with onset/offset timestamps and durations.

### 7.3 Inference Pipeline

During inference, the evaluation pipeline follows these steps:

1. Load the preprocessed evaluation data (or process from raw EDF files)
2. Load the best or final trained checkpoint
3. Iterate through all windows sequentially, collecting agent predictions
4. Compute and report all evaluation metrics
5. Optionally generate detailed visualizations and export data

### 7.4 Model Selection

The model with the **highest F1-score** on the development set is selected as the best model. F1-score is chosen as the primary model selection criterion because it balances precision and sensitivity—both critical in the clinical seizure detection context where both missed seizures and false alarms carry significant consequences.

---

## 8. Implementation Framework

### 8.1 Software Dependencies

| Library        | Purpose                                      |
|----------------|----------------------------------------------|
| PyTorch        | Deep learning framework, GPU acceleration     |
| MNE-Python     | EEG data loading and signal processing        |
| Gymnasium      | RL environment interface (OpenAI Gym standard) |
| NumPy          | Numerical computation and array operations    |
| scikit-learn   | Evaluation metrics computation                |
| pandas         | Data export and event table management        |
| matplotlib     | Visualization and plotting                    |
| seaborn        | Enhanced statistical visualization            |
| tqdm           | Progress tracking for training loops          |

### 8.2 Project Structure

```
RL-EEG-TUH/
├── config.py                    # Central configuration and hyperparameters
├── train.py                     # Main training entry point
├── evaluate.py                  # Main evaluation entry point
├── data/edf/{train,dev,eval}/   # TUH EEG corpus data
├── cache/                       # Preprocessed data cache (.npz)
├── checkpoints/                 # Saved model checkpoints (.pt)
├── src/
│   ├── preprocessing/
│   │   ├── edf_loader.py        # EDF loading, filtering, normalization
│   │   ├── annotations.py       # Annotation CSV parsing
│   │   ├── windowing.py         # Sliding window segmentation
│   │   └── preprocess.py        # Orchestration and caching
│   ├── environment/
│   │   └── eeg_env.py           # Gymnasium RL environment
│   ├── models/
│   │   ├── feature_extractor.py # CNN and CNN-LSTM backbones
│   │   ├── ppo_agent.py         # PPO Actor-Critic agent
│   │   └── dqn_agent.py         # Double DQN agent
│   ├── training/
│   │   ├── train_ppo.py         # PPO training loop
│   │   └── train_dqn.py         # DQN training loop
│   └── evaluation/
│       └── evaluate.py          # Metrics, plotting, Excel export
└── evaluation_results/          # Generated reports and plots
```

### 8.3 Reproducibility

*All experiments use a fixed random seed of **42** for NumPy, PyTorch, and Python random generators to ensure full reproducibility. Hyperparameters are centralized in `config.py` to facilitate controlled experimentation and ablation studies.*

---
