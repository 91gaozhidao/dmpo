# Comprehensive Execution Guide

> Step-by-step instructions for pre-training, evaluation, and online RL fine-tuning with DMPO.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Pre-training with Drifting Policy](#2-pre-training-with-drifting-policy)
3. [Evaluation (Offline & Online)](#3-evaluation)
4. [PPO Fine-tuning](#4-ppo-fine-tuning)
5. [GRPO Fine-tuning (Critic-Free)](#5-grpo-fine-tuning)
6. [Configuration Reference](#6-configuration-reference)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Environment Setup

### 1.1 Install Dependencies

```bash
# Clone the repository
git clone https://github.com/xxx/DMPO.git
cd DMPO

# Install base package
pip install -e .

# For Gym/D4RL tasks
pip install -e ".[gym]"

# For Franka Kitchen tasks
pip install -e ".[kitchen]"

# For RoboMimic tasks
pip install -e ".[robomimic]"

# For D3IL tasks
pip install -e ".[d3il]"

# For FurnitureBench tasks
pip install -e ".[furniture]"
```

### 1.2 Set Environment Variables

```bash
# Required: set the root data and log directories
export REINFLOW_DIR=/path/to/your/workspace
export REINFLOW_DATA_DIR=$REINFLOW_DIR/data
export REINFLOW_LOG_DIR=$REINFLOW_DIR/logs
```

### 1.3 Download Datasets

The launcher script (`script/run.py`) automatically downloads datasets from Google Drive when needed. You can also download manually:

```bash
# Datasets are stored at $REINFLOW_DATA_DIR/{env_suite}/{task}/
# Example: $REINFLOW_DATA_DIR/gym/hopper-medium-v2/train.npz
```

---

## 2. Pre-training with Drifting Policy

### 2.1 Standard Drifting Pre-training

```bash
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp
```

**What happens:**
1. Loads expert dataset from `$REINFLOW_DATA_DIR/gym/hopper-medium-v2/`
2. Creates a `DriftingPolicy` model with `MeanFlowMLP` backbone
3. Trains for `n_epochs=40` using the drift field loss
4. Saves checkpoints at `$REINFLOW_LOG_DIR/` with WandB logging
5. Optionally evaluates in MuJoCo every `test_freq` epochs

**Key hyperparameters to tune:**
```yaml
train:
  n_epochs: 40            # Number of training epochs
  batch_size: 128         # Batch size
  learning_rate: 1e-3     # Learning rate
model:
  drift_coef: 0.1         # Positive drift strength
  neg_drift_coef: 0.05    # Negative drift strength  
  bandwidth: 1.0          # RBF kernel bandwidth
  mask_self: false         # Self-interaction masking
```

### 2.2 Drifting with Dispersive Loss

```bash
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_dispersive_mlp
```

This variant adds a dispersive regularization term to prevent mode collapse.

### 2.3 Available Pre-training Tasks

| Suite | Task | Config |
|-------|------|--------|
| Gym | Hopper | `cfg/gym/pretrain/hopper-medium-v2/pre_drifting_mlp.yaml` |
| Gym | Walker2d | `cfg/gym/pretrain/walker2d-medium-v2/pre_drifting_mlp.yaml` |
| Gym | Ant | `cfg/gym/pretrain/ant-medium-expert-v0/pre_drifting_mlp.yaml` |
| Gym | Humanoid | `cfg/gym/pretrain/humanoid-medium-v3/pre_drifting_mlp.yaml` |
| Kitchen | Mixed | `cfg/gym/pretrain/kitchen-mixed-v0/pre_drifting_mlp.yaml` |
| Kitchen | Complete | `cfg/gym/pretrain/kitchen-complete-v0/pre_drifting_mlp.yaml` |
| Kitchen | Partial | `cfg/gym/pretrain/kitchen-partial-v0/pre_drifting_mlp.yaml` |

---

## 3. Evaluation

### 3.1 Offline Evaluation (State-Based)

```bash
python script/run.py \
  --config-dir=cfg/gym/eval/hopper-v2 \
  --config-name=eval_drifting_mlp \
  checkpoint_path=/path/to/checkpoint.pt
```

### 3.2 Offline Evaluation (Image-Based)

```bash
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_drifting_mlp_img \
  checkpoint_path=/path/to/checkpoint.pt
```

### 3.3 What the Evaluation Agent Does

1. Loads the pretrained checkpoint (supports both EMA and standard weights)
2. Runs `N` episodes in the target environment
3. Records: episode reward, success rate, action statistics
4. Optionally saves video recordings
5. Reports metrics to WandB

**Key evaluation settings:**
```yaml
denoising_steps: [1]              # Always 1 for Drifting
load_ema: false                   # Load EMA weights or original
clip_intermediate_actions: true   # Clip actions to bounds
n_eval_episodes: 50               # Number of evaluation episodes
```

---

## 4. PPO Fine-tuning

### 4.1 State-Based PPO Fine-tuning

```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_drifting_mlp \
  base_policy_path=/path/to/pretrained_checkpoint.pt
```

### 4.2 Image-Based PPO Fine-tuning

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_drifting_mlp_img \
  base_policy_path=/path/to/pretrained_checkpoint.pt
```

### 4.3 PPO Training Pipeline

The PPO fine-tuning follows this cycle:

```
For each iteration:
  1. COLLECT: Run n_steps in n_envs parallel environments
     - Sample actions from NoisyDriftingMLP (mean + noise)
     - Store (obs, action_chain, reward, done) in PPO buffer
  
  2. COMPUTE: GAE advantages and returns
     - Generalized Advantage Estimation (γ=0.99, λ=0.95)
     - Value function bootstrapping
  
  3. UPDATE: PPO gradient updates
     - Multiple epochs over collected data
     - Clipped surrogate objective
     - Value function loss
     - Entropy bonus
     - Optional BC regularization loss
     - Gradient clipping
     - Adaptive KL-based early stopping
```

### 4.4 Key PPO Hyperparameters

```yaml
train:
  n_train_itr: 1000       # Total training iterations
  n_steps: 500             # Steps per iteration (per env)
  actor_lr: 3.0e-06        # Actor learning rate (small for fine-tuning)
  critic_lr: 4.5e-4        # Critic learning rate
  gamma: 0.99              # Discount factor
  gae_lambda: 0.95         # GAE lambda
  batch_size: 50000         # PPO minibatch size
  update_epochs: 10         # Gradient epochs per iteration
  vf_coef: 0.5             # Value loss coefficient
  ent_coef: 0.01            # Entropy coefficient
  target_kl: 0.01           # KL divergence threshold
  use_bc_loss: true         # Enable BC regularization
  bc_loss_coeff: 0.05       # BC loss weight

env:
  n_envs: 40               # Number of parallel environments
```

---

## 5. GRPO Fine-tuning (Critic-Free)

### 5.1 Running GRPO Fine-tuning

```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_grpo_drifting_mlp \
  base_policy_path=/path/to/pretrained_checkpoint.pt
```

### 5.2 GRPO Training Pipeline

GRPO (Group Relative Policy Optimization) removes the critic network entirely:

```
For each iteration:
  1. COLLECT: G trajectories from same initial state
     - Homogeneous reset: all G trajectories start identically
     - Sample actions from NoisyDriftingPolicy (Tanh-Normal)
     - Record (state, action, log_prob, reward) per step
  
  2. NORMALIZE: Compute group-relative advantages
     - Return R_g = Σ γ^t r_t for each trajectory g
     - Advantage A_g = (R_g - mean(R)) / (std(R) + ε)
     - Zero-variance protection: if std(R) < 1e-6, A_g = 0
  
  3. UPDATE: GRPO gradient updates
     - Clipped surrogate loss (same as PPO)
     - Analytical KL divergence penalty (no sampling variance)
     - Beta decay: β ← max(β * decay, β_min)
     - No value function loss (critic-free!)
```

### 5.3 Key GRPO Hyperparameters

```yaml
train:
  group_size: 16            # Trajectories per group (G)
  grpo_lr: 1e-5             # Actor learning rate
  update_epochs: 4           # Gradient epochs per iteration
  grpo_batch_size: 256       # Minibatch size
  kl_beta: 0.05              # Initial KL penalty coefficient
  kl_beta_min: 0.001         # Minimum KL penalty
  kl_beta_decay: 0.995       # Beta decay rate per iteration
  use_homogeneous_reset: true # Strict same-initial-state sampling

model:
  epsilon: 0.2               # PPO clipping range
  beta: 0.05                  # Initial KL penalty (model-level)
```

### 5.4 GRPO vs PPO Comparison

| Feature | PPO | GRPO |
|---------|-----|------|
| Critic network | ✅ Required | ❌ Not needed |
| Value estimation | GAE bootstrap | Group Z-score |
| KL computation | Approximate | Analytical |
| Action distribution | Gaussian | Tanh-Normal |
| Jacobian correction | — | ✅ Required |
| Memory footprint | Higher (actor + critic) | Lower (actor only) |
| Sample efficiency | Higher | Lower (needs G trajectories) |

---

## 6. Configuration Reference

### 6.1 Config File Structure

Every config YAML contains these standard sections:

```yaml
# Top-level settings
env_suite: gym                     # Environment suite
env: hopper-medium-v2              # Dataset/environment name
action_dim: 3                      # Action dimension
horizon_steps: 4                   # Action horizon (chunk size)
obs_dim: 11                        # Observation dimension
cond_steps: 1                      # Conditioning history steps

# Agent class selection (Hydra instantiation target)
_target_: agent.pretrain.train_drifting_agent.TrainDriftingAgent

# Model architecture
model:
  _target_: model.drifting.drifting.DriftingPolicy
  # ... model-specific params

# Training hyperparameters
train:
  # ... training-specific params

# Dataset (pretrain only)
train_dataset:
  _target_: agent.dataset.sequence.StitchedSequenceDataset
  # ... dataset params

# EMA (pretrain only)
ema:
  decay: 0.995
```

### 6.2 Overriding Parameters from Command Line

Hydra allows command-line overrides:

```bash
# Override learning rate
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp \
  train.learning_rate=5e-4

# Override model parameters
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp \
  model.drift_coef=0.2 \
  model.bandwidth=0.5
```

---

## 7. Troubleshooting

### 7.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'REINFLOW_DIR'` | Environment variable not set | `export REINFLOW_DIR=/path/to/workspace` |
| `NaN` in loss | Learning rate too high or drift_coef too large | Reduce `learning_rate` and `drift_coef` |
| OOM (GPU) | Batch size too large for image tasks | Increase `grad_accumulate`, decrease `batch_size` |
| Ratio ≠ 1.0 at epoch 0 | Bug in log-prob computation | Check chain format and noise std bounds |
| KL divergence explodes | Beta too small or learning rate too high | Increase `kl_beta`, decrease learning rate |

### 7.2 Debugging Tips

```bash
# Enable verbose logging
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_drifting_mlp \
  train.verbose=true

# Run with fewer iterations for quick testing
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp \
  train.n_epochs=2 \
  train.test_freq=1
```

### 7.3 Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_drifting_policy.py -v
python -m pytest tests/test_ppo_drifting.py -v
python -m pytest tests/test_grpo_drifting.py -v
python -m pytest tests/test_grpo_buffer.py -v
python -m pytest tests/test_end_to_end_smoke.py -v
python -m pytest tests/test_static_and_config.py -v
```
