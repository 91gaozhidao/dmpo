# DMPO — Project Architecture & Deep Dive

> **Dispersive MeanFlow Policy Optimization**: A unified framework for single-step (1-NFE) generative policy learning with online RL fine-tuning.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Repository Directory Map](#2-repository-directory-map)
3. [Core Data Flow](#3-core-data-flow)
4. [Model Layer Deep Dive](#4-model-layer-deep-dive)
5. [Agent Layer Deep Dive](#5-agent-layer-deep-dive)
6. [Configuration System](#6-configuration-system)
7. [Drifting Policy vs Mean Flow Policy — Mathematical & Code Comparison](#7-drifting-policy-vs-mean-flow-policy)
8. [Fine-Tuning Architectures: PPO & GRPO](#8-fine-tuning-architectures)
9. [Environment Integration](#9-environment-integration)

---

## 1. High-Level Overview

DMPO implements a two-stage training pipeline for continuous control:

```
Stage 1: Offline Pre-training          Stage 2: Online RL Fine-tuning
┌──────────────────────────┐          ┌──────────────────────────────┐
│  Expert Dataset (D4RL)   │          │  Environment Interaction     │
│         ↓                │          │         ↓                    │
│  Flow Matching / Drift   │   ──►    │  PPO or GRPO Objective       │
│  Loss Optimization       │          │  + Exploration Noise         │
│         ↓                │          │         ↓                    │
│  Pretrained Policy θ     │          │  Fine-tuned Policy θ*        │
└──────────────────────────┘          └──────────────────────────────┘
```

**Supported Policy Types:**
| Policy | NFE | Stage 1 | Stage 2 (PPO) | Stage 2 (GRPO) |
|--------|-----|---------|---------------|----------------|
| **Drifting** | 1 | ✅ | ✅ | ✅ |
| MeanFlow | 5 | ✅ | ✅ | — |
| ShortCut | 1–5 | ✅ | ✅ | — |
| ReFlow | 5+ | ✅ | ✅ | — |
| Diffusion | 10–100 | ✅ | ✅ | — |
| Consistency | 1 | ✅ | — | — |

---

## 2. Repository Directory Map

```
dmpo/
├── agent/                          # Training & evaluation agents
│   ├── pretrain/                   # Offline pre-training agents
│   │   ├── train_agent.py              # Base PreTrainAgent class
│   │   ├── train_drifting_agent.py     # Drifting Policy pretraining
│   │   ├── train_drifting_dispersive_agent.py  # Drifting + dispersive loss
│   │   ├── train_meanflow_agent.py     # MeanFlow pretraining
│   │   ├── train_improved_meanflow_agent.py
│   │   ├── train_shortcut_agent.py
│   │   ├── train_shortcut_dispersive_agent.py
│   │   ├── train_reflow_agent.py
│   │   ├── train_reflow_dispersive_agent.py
│   │   ├── train_consistency_agent.py
│   │   ├── train_diffusion_agent.py
│   │   └── train_gaussian_agent.py
│   │
│   ├── eval/                       # Evaluation agents
│   │   ├── eval_agent_base.py          # Base evaluation class
│   │   ├── eval_agent_img_base.py      # Image-based evaluation base
│   │   ├── eval_drifting_agent.py      # Drifting Policy evaluation
│   │   ├── eval_drifting_img_agent.py  # Drifting with image obs
│   │   ├── eval_meanflow_agent.py
│   │   ├── eval_meanflow_img_agent.py
│   │   ├── eval_shortcut_agent.py
│   │   ├── eval_shortcut_img_agent.py
│   │   ├── eval_reflow_agent.py
│   │   ├── eval_reflow_img_agent.py
│   │   ├── eval_diffusion_agent.py
│   │   ├── eval_diffusion_img_agent.py
│   │   └── eval_consistency_img_agent.py
│   │
│   ├── finetune/                   # Online RL fine-tuning agents
│   │   ├── reinflow/                   # PPO-based fine-tuning
│   │   │   ├── train_agent.py              # Base PPO agent
│   │   │   ├── train_ppo_shortcut_agent.py     # ShortCut PPO (state)
│   │   │   ├── train_ppo_shortcut_img_agent.py # ShortCut PPO (image)
│   │   │   ├── train_ppo_drifting_agent.py     # Drifting PPO (state)
│   │   │   ├── train_ppo_drifting_img_agent.py # Drifting PPO (image)
│   │   │   ├── train_ppo_meanflow_agent.py     # MeanFlow PPO (state)
│   │   │   ├── train_ppo_meanflow_img_agent.py # MeanFlow PPO (image)
│   │   │   ├── train_ppo_flow_agent.py         # Generic flow PPO
│   │   │   ├── train_ppo_flow_img_agent.py
│   │   │   ├── train_ppo_diffusion_agent.py
│   │   │   ├── train_ppo_diffusion_img_agent.py
│   │   │   ├── train_ppo_gaussian_agent.py
│   │   │   └── buffer.py                  # PPO replay buffers
│   │   │
│   │   ├── grpo/                       # GRPO fine-tuning (critic-free)
│   │   │   ├── train_grpo_drifting_agent.py    # GRPO for Drifting
│   │   │   └── buffer.py                  # GRPO group buffer
│   │   │
│   │   ├── dppo/                       # Diffusion PPO variants
│   │   ├── dpro/                       # Diffusion policy RL optimization
│   │   ├── diffusion_baselines/        # DIPO, QSM, DQL, AWR, RWR, IDQL
│   │   ├── flow_baselines/             # FQL, SAC
│   │   └── offlinerl_baselines/        # IBRL, CalQL, RLPD
│   │
│   └── dataset/                    # Data loading utilities
│       ├── sequence.py                 # StitchedSequenceDataset
│       └── d3il/                       # D3IL task datasets
│
├── model/                          # Neural network implementations
│   ├── common/                     # Shared components
│   │   ├── modules.py                  # MLP blocks, RandomShiftsAug, etc.
│   │   ├── critic.py                   # Critic networks (CriticObs)
│   │   └── normalizer.py              # Observation normalizers
│   │
│   ├── drifting/                   # Drifting Policy (1-NFE)
│   │   ├── drifting.py                 # DriftingPolicy core
│   │   ├── ft_ppo/
│   │   │   └── ppodrifting.py          # NoisyDriftingMLP + PPODrifting
│   │   └── ft_grpo/
│   │       └── grpodrifting.py         # NoisyDriftingPolicy + GRPODrifting
│   │
│   ├── flow/                       # Flow matching models
│   │   ├── meanflow.py                 # MeanFlow policy
│   │   ├── improved_meanflow.py        # Improved MeanFlow
│   │   ├── reflow.py                   # Rectified Flow
│   │   ├── shortcutflow.py             # ShortCut Flow
│   │   ├── consistency.py              # Consistency models
│   │   ├── mlp_meanflow.py             # MeanFlowMLP backbone
│   │   ├── mlp_shortcut.py             # ShortCut MLP backbone
│   │   ├── mlp_consistency.py          # Consistency MLP backbone
│   │   ├── ft_ppo/                     # PPO wrappers for flows
│   │   │   ├── ppoflow.py                 # Base PPOFlow class
│   │   │   ├── ppomeanflow.py              # PPOMeanFlow
│   │   │   └── pposhortcut.py              # PPOShortCut
│   │   └── ft_baselines/
│   │       └── fql.py                  # Flow Q-Learning
│   │
│   ├── diffusion/                  # DDPM/DDIM diffusion models
│   ├── gaussian/                   # Gaussian policies
│   └── rl/                         # RL utility modules
│
├── cfg/                            # Hydra configuration files
│   ├── gym/                        # OpenAI Gym & Franka Kitchen
│   ├── robomimic/                  # RoboMimic manipulation tasks
│   ├── furniture/                  # FurnitureBench assembly tasks
│   └── d3il/                       # D3IL imitation learning tasks
│
├── script/                         # Launch scripts
│   └── run.py                      # Unified experiment launcher
│
├── tests/                          # Test suite
│   ├── test_drifting_policy.py         # Core DriftingPolicy tests
│   ├── test_ppo_drifting.py            # PPO fine-tuning tests
│   ├── test_grpo_drifting.py           # GRPO fine-tuning tests
│   ├── test_grpo_buffer.py             # GRPO buffer tests
│   ├── test_end_to_end_smoke.py        # Integration smoke tests
│   └── test_static_and_config.py       # Import & config validation tests
│
├── util/                           # Utility functions
│   ├── dirs.py                         # Directory management
│   └── config.py                       # Config utilities
│
├── env/                            # Environment wrappers
└── data_process/                   # Data preprocessing scripts
```

---

## 3. Core Data Flow

### 3.1 Pre-training Data Flow

```
StitchedSequenceDataset
    ↓
batch_data = (actions, observations)
    │
    │  actions: Tensor[B, T_a, D_a]      # B=batch, T_a=horizon, D_a=action_dim
    │  observations: dict{"state": Tensor[B, T_c, D_obs]}
    │                                     # T_c=cond_steps, D_obs=obs_dim
    ↓
PreTrainAgent.get_loss(batch_data)
    ↓
model.loss(x1=actions, cond=observations)
    │
    │  [Inside DriftingPolicy.loss()]:
    │  1. x_gen = randn(B, T_a, D_a)          # Initial noise
    │  2. x_pred = network(x_gen, t=1, r=0, cond)   # 1-NFE forward
    │  3. V = compute_V(x_pred, actions)       # Drift field
    │  4. target = (x_pred + V).detach()       # Drifted target
    │  5. loss = MSE(x_pred, target)           # Training loss
    ↓
loss.backward()  →  optimizer.step()
```

### 3.2 Online RL Fine-tuning Data Flow (PPO)

```
Environment (vectorized, n_envs parallel)
    ↓
obs_venv = {"state": np.array[n_envs, D_obs]}
    ↓
cond = {"state": Tensor[n_envs, D_obs]}   # to device
    ↓
PPODrifting.get_actions(cond)
    │
    │  [Inside NoisyDriftingMLP]:
    │  1. x0 = randn(n_envs, T_a, D_a)         # Initial noise
    │  2. mean = network(x0, t=1, r=0, cond)    # 1-NFE deterministic output
    │  3. std = noise_network(mean)              # Learned exploration noise
    │  4. action = mean + std * randn(...)       # Stochastic action
    │  5. log_prob = Normal(mean, std).log_prob(action)
    ↓
action_venv = actions[:, :act_steps]     # First act_steps
    ↓
obs, reward, done = env.step(action_venv)
    ↓
PPOBuffer.add(obs, chains, reward, done)
    ↓  [After n_steps]
PPOBuffer.make_dataset()
    │  → GAE advantage estimation
    │  → Returns computation
    ↓
PPODrifting.loss(obs, chains, returns, values, advantages, old_logprobs)
    │  → Clipped surrogate loss
    │  → Value function loss
    │  → Entropy bonus
    │  → Optional BC regularization
    ↓
loss.backward()  →  actor_optimizer.step() + critic_optimizer.step()
```

### 3.3 Online RL Fine-tuning Data Flow (GRPO — Critic-Free)

```
Environment (vectorized)
    ↓
Homogeneous Reset: G trajectories from same initial state
    ↓
For each trajectory g in [1..G]:
    ├── Sample actions via NoisyDriftingPolicy
    ├── Collect (state, action, log_prob, reward) tuples
    └── Compute trajectory return R_g
    ↓
GRPOBuffer.normalize_advantages()
    │  advantages = (returns - mean(returns)) / (std(returns) + eps)
    │  Zero-variance protection: if std < 1e-6, set advantages = 0
    ↓
GRPODrifting.compute_loss(obs, actions, advantages, old_log_probs)
    │  1. curr_log_prob = NoisyDriftingPolicy.get_log_prob(obs, actions)
    │  2. ratio = exp(curr_log_prob - old_log_prob)
    │  3. surr1 = ratio * advantages
    │  4. surr2 = clamp(ratio, 1-eps, 1+eps) * advantages
    │  5. policy_loss = -min(surr1, surr2).mean()
    │  6. kl_div = analytical_kl(current_dist, ref_dist)   # No critic!
    │  7. loss = policy_loss + beta * kl_div
    ↓
loss.backward()  →  optimizer.step()
```

---

## 4. Model Layer Deep Dive

### 4.1 MeanFlowMLP — Shared Backbone

**File:** `model/flow/mlp_meanflow.py`

The `MeanFlowMLP` is the neural network backbone shared by both MeanFlow and Drifting policies. It accepts:

| Input | Shape | Description |
|-------|-------|-------------|
| `x` | `[B, T_a, D_a]` | Noisy action sequence |
| `t` | `[B]` | Time step (0→1) |
| `r` | `[B]` | Auxiliary variable (resolution) |
| `cond` | `dict{"state": [B, T_c, D_obs]}` | Conditioning observation |

**Output:** `[B, T_a, D_a]` — predicted velocity/action field.

### 4.2 DriftingPolicy

**File:** `model/drifting/drifting.py`

Core implementation of the 1-NFE Drifting Policy:

- **`compute_V(x, y_pos, y_neg)`**: Computes the drift field using RBF-weighted sample interactions
- **`loss(x1, cond)`**: Pre-training loss via MSE between network output and drifted targets
- **`sample(cond, ...)`**: Single forward pass with `t=1.0, r=0.0`

### 4.3 PPODrifting

**File:** `model/drifting/ft_ppo/ppodrifting.py`

Two key components:
- **`NoisyDriftingMLP`**: Wraps MeanFlowMLP with learned exploration noise (MLP predicts log-variance)
- **`PPODrifting(PPOFlow)`**: PPO objective with single-step log-probability computation

### 4.4 GRPODrifting

**File:** `model/drifting/ft_grpo/grpodrifting.py`

Three key components:
- **`_tanh_jacobian_correction(u)`**: Numerically stable Jacobian for Tanh-Normal distributions
- **`NoisyDriftingPolicy`**: Tanh-Normal policy wrapper with proper log-probability
- **`GRPODrifting`**: Critic-free GRPO objective with analytical KL divergence

---

## 5. Agent Layer Deep Dive

### 5.1 PreTrainAgent (Base)

**File:** `agent/pretrain/train_agent.py`

Provides the complete training loop:
1. Model instantiation via Hydra `_target_`
2. EMA model management
3. Dataset loading and splitting
4. Optimizer and LR scheduler
5. Checkpoint save/load with auto-resume
6. Optional in-training MuJoCo evaluation
7. WandB logging integration

**Key abstract methods that subclasses override:**
- `get_loss(batch_data)` — Compute training loss
- `inference(cond)` — Generate samples for evaluation

### 5.2 Training Agent Hierarchy

```
PreTrainAgent
├── TrainDriftingAgent             # 1-NFE, drift field loss
│   └── TrainDriftingDispersiveAgent   # + dispersive regularization
├── TrainMeanFlowAgent             # 5-step flow matching
├── TrainShortCutAgent             # Adaptive-step shortcut flow
│   ├── TrainShortCutDispersiveAgent
│   └── TrainMeanFlowDispersiveAgent
├── TrainReFlowAgent               # Rectified flow
├── TrainConsistencyAgent          # Consistency distillation
├── TrainDiffusionAgent            # DDPM/DDIM diffusion
└── TrainGaussianAgent             # Simple Gaussian BC

TrainPPOShortCutAgent (PPO base)
├── TrainPPODriftingAgent          # Drifting PPO (state)
├── TrainPPOMeanFlowAgent          # MeanFlow PPO (state)
├── TrainPPOFlowAgent              # Generic flow PPO
├── TrainPPODiffusionAgent         # Diffusion PPO
├── TrainPPOGaussianAgent          # Gaussian PPO
└── TrainPPOImgShortCutAgent       # ShortCut PPO (image)
    ├── TrainPPOImgDriftingAgent   # Drifting PPO (image)
    ├── TrainPPOImgMeanFlowAgent   # MeanFlow PPO (image)
    ├── TrainPPOImgFlowAgent       # Flow PPO (image)
    └── TrainPPOImgDiffusionAgent  # Diffusion PPO (image)

TrainGRPODriftingAgent             # GRPO (critic-free, Drifting only)
```

---

## 6. Configuration System

All experiments are configured via Hydra YAML files organized by:

```
cfg/{env_suite}/{stage}/{task}/{config_name}.yaml
```

**Example paths:**
```
cfg/gym/pretrain/hopper-medium-v2/pre_drifting_mlp.yaml
cfg/gym/finetune/hopper-v2/ft_ppo_drifting_mlp.yaml
cfg/gym/finetune/hopper-v2/ft_grpo_drifting_mlp.yaml
cfg/gym/eval/hopper-v2/eval_drifting_mlp.yaml
cfg/robomimic/pretrain/can/pre_drifting_mlp_img.yaml
```

**Key config sections:**

| Section | Purpose | Example Keys |
|---------|---------|--------------|
| `model` | Neural network architecture | `_target_`, `mlp_dims`, `act_min/max` |
| `train` | Training hyperparameters | `n_epochs`, `batch_size`, `learning_rate` |
| `env` | Environment settings | `n_envs`, `name`, `max_episode_steps` |
| `train_dataset` | Data loading config | `_target_`, `dataset_path`, `horizon_steps` |
| `ema` | EMA configuration | `decay` |

---

## 7. Drifting Policy vs Mean Flow Policy

### 7.1 Mathematical Foundations

**Mean Flow Policy** learns a velocity field `v(x, t)` that transports noise `x_0 ~ N(0,I)` to data `x_1` over time `t ∈ [0,1]`:

```
dx/dt = v(x, t)     →   x_1 = x_0 + ∫₀¹ v(x_t, t) dt
```

At inference, this integral is approximated with multiple Euler steps (typically 5 NFE).

**Drifting Policy** collapses this into a single step by training a network to directly predict the final action, then applying a drift correction field:

```
x_pred = f_θ(x_0, t=1, r=0, cond)           # Single forward pass
V = compute_V(x_pred, x_expert)              # Drift field toward expert data
target = (x_pred + V).detach()               # Drifted target
loss = MSE(x_pred, target)                   # Self-supervised refinement
```

### 7.2 Code-Level Comparison

| Aspect | MeanFlow | Drifting |
|--------|----------|---------|
| **Core file** | `model/flow/meanflow.py` | `model/drifting/drifting.py` |
| **Network** | `MeanFlowMLP` | `MeanFlowMLP` (same backbone) |
| **NFE at inference** | 5 (Euler steps) | 1 (single forward) |
| **Loss** | Velocity matching MSE | Drift-corrected MSE |
| **Training `t`** | Sampled uniformly `t ~ U(0,1)` | Fixed `t=1.0` |
| **Training `r`** | Sampled or fixed | Fixed `r=0.0` |
| **PPO wrapper** | `PPOMeanFlow` | `PPODrifting` |
| **GRPO wrapper** | — | `GRPODrifting` |
| **Pretrain agent** | `TrainMeanFlowAgent` | `TrainDriftingAgent` |
| **Eval agent** | `eval_meanflow_agent.py` | `eval_drifting_agent.py` |

### 7.3 Key Difference: Drift Field Computation

```python
# In DriftingPolicy.compute_V():
# 1. Compute pairwise distances between predictions and expert data
diff = x.unsqueeze(1) - y_pos.unsqueeze(0)    # [B, B, T_a, D_a]
dist_sq = (diff ** 2).sum(dim=(-1, -2))         # [B, B]

# 2. RBF kernel weighting
weights = torch.exp(-dist_sq / (2 * bandwidth**2))  # [B, B]

# 3. Optional self-masking
if mask_self:
    weights.fill_diagonal_(0)

# 4. Weighted drift toward expert data
weights_norm = weights / (weights.sum(dim=1, keepdim=True) + eps)
V_pos = (weights_norm.unsqueeze(-1).unsqueeze(-1) * diff).sum(dim=1)

# 5. Scale by drift coefficient
V_total = drift_coef * V_pos - neg_drift_coef * V_neg
```

---

## 8. Fine-Tuning Architectures

### 8.1 PPO Fine-Tuning

Both MeanFlow and Drifting use the same PPO framework (`PPOFlow` base class), differing only in:
- **Log-probability computation**: Drifting uses a single Gaussian transition; MeanFlow chains multiple steps
- **Inference steps**: Drifting forces `inference_steps=1`
- **Chain length**: Drifting chains are always length 2 (noise + action)

### 8.2 GRPO Fine-Tuning (Drifting Only)

GRPO is unique to Drifting Policy and offers:
- **No critic network** — reduces parameters and avoids value estimation bias
- **Group-relative advantages** — Z-score normalization within trajectory groups
- **Analytical KL divergence** — eliminates sampling variance in divergence estimation
- **Tanh-Normal distribution** — proper bounded action support with Jacobian correction

---

## 9. Environment Integration

### 9.1 Supported Environment Suites

| Suite | Tasks | Observation | Action Space |
|-------|-------|-------------|--------------|
| **Gym (D4RL)** | HalfCheetah, Hopper, Walker2d, Ant, Humanoid | State vector | Continuous |
| **Franka Kitchen** | kitchen-mixed, kitchen-complete, kitchen-partial | State vector | Continuous |
| **RoboMimic** | Can, Lift, Transport, Square | RGB images + state | Continuous |
| **FurnitureBench** | one_leg_low, one_leg_high, square_table | RGB images + state | Continuous |
| **D3IL** | Avoid, Aligning, Pushing, Sorting, Stacking | State vector | Continuous |

### 9.2 Observation Processing

State-based tasks use a simple dictionary:
```python
cond = {"state": Tensor[B, D_obs]}
```

Image-based tasks use a multi-modal dictionary:
```python
cond = {
    "rgb": Tensor[B, C, H, W],       # RGB image
    "state": Tensor[B, D_proprio],     # Proprioceptive state
}
```

The image PPO agents (`TrainPPOImgDriftingAgent`, etc.) handle image augmentation via `RandomShiftsAug` and support gradient accumulation for memory efficiency.
