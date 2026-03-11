# Drifting Policy — Developer's Cookbook

> A hands-on guide for modifying, extending, and debugging the Drifting Policy implementation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Modifying Pre-training](#2-modifying-pre-training)
3. [Modifying PPO Fine-tuning](#3-modifying-ppo-fine-tuning)
4. [Modifying GRPO Fine-tuning](#4-modifying-grpo-fine-tuning)
5. [Adding a New Environment](#5-adding-a-new-environment)
6. [Key Constants & Parameters](#6-key-constants--parameters)
7. [Debugging Guide](#7-debugging-guide)

---

## 1. Architecture Overview

The Drifting Policy consists of three layers that build on each other:

```
Layer 3: Agent (training loop orchestration)
  ├── agent/pretrain/train_drifting_agent.py
  ├── agent/finetune/reinflow/train_ppo_drifting_agent.py
  ├── agent/finetune/reinflow/train_ppo_drifting_img_agent.py
  └── agent/finetune/grpo/train_grpo_drifting_agent.py

Layer 2: Model (loss computation, action sampling, RL objectives)
  ├── model/drifting/drifting.py              # Core policy
  ├── model/drifting/ft_ppo/ppodrifting.py    # PPO wrapper
  └── model/drifting/ft_grpo/grpodrifting.py  # GRPO wrapper

Layer 1: Network (neural network backbone)
  └── model/flow/mlp_meanflow.py              # MeanFlowMLP
```

---

## 2. Modifying Pre-training

### 2.1 Changing the Network Architecture

**File to edit:** `model/flow/mlp_meanflow.py`

The `MeanFlowMLP` class defines the backbone network. To change the architecture:

```python
# model/flow/mlp_meanflow.py
class MeanFlowMLP(torch.nn.Module):
    def __init__(self, action_dim, horizon_steps, cond_dim, 
                 mlp_dims=[512, 512, 512],      # ← Change hidden dims
                 activation_type="Mish",         # ← Change activation
                 ...):
```

You can also change it via config without editing code:

```yaml
# In your config YAML:
model:
  network:
    mlp_dims: [1024, 1024, 1024]   # Wider network
    activation_type: ReLU           # Different activation
```

### 2.2 Changing the Drift Field Computation

**File to edit:** `model/drifting/drifting.py`, method `compute_V()`

The drift field determines how generated samples are attracted toward expert data:

```python
# model/drifting/drifting.py, class DriftingPolicy

def compute_V(self, x, y_pos, y_neg=None):
    """
    x:     [B, T_a, D_a]  - generated action predictions
    y_pos: [B, T_a, D_a]  - positive (expert) targets
    y_neg: [B, T_a, D_a]  - negative targets (optional)
    
    To modify the drift kernel:
    1. Change the distance metric (currently L2)
    2. Change the kernel function (currently RBF/Gaussian)
    3. Change the normalization scheme
    """
    # Current: RBF kernel
    diff = x.unsqueeze(1) - y_pos.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=(-1, -2))
    weights = torch.exp(-dist_sq / (2 * self.bandwidth**2))
    
    # Example modification: use Laplacian kernel instead
    # dist = diff.abs().sum(dim=(-1, -2))
    # weights = torch.exp(-dist / self.bandwidth)
```

### 2.3 Changing the Loss Function

**File to edit:** `model/drifting/drifting.py`, method `loss()`

```python
# model/drifting/drifting.py, class DriftingPolicy

def loss(self, x1, cond):
    """
    x1:   [B, T_a, D_a]  - expert actions (training targets)
    cond: dict            - conditioning observations
    
    Current loss: MSE between network output and drift-corrected target
    """
    # Step 1: Generate noise
    x_gen = torch.randn_like(x1)
    
    # Step 2: Forward pass (1-NFE)
    x_pred = self.network(x_gen, t=ones, r=zeros, cond=cond)
    
    # Step 3: Compute drift field
    V = self.compute_V(x_pred, y_pos=x1)
    
    # Step 4: Create target
    target = (x_pred + V).detach()
    
    # Step 5: Loss (modify here to change loss type)
    loss = F.mse_loss(x_pred, target)
    
    # Example: Add L1 regularization
    # loss = loss + 0.01 * x_pred.abs().mean()
    
    return loss
```

### 2.4 Changing the Dispersive Regularization

**File to edit:** `agent/pretrain/train_drifting_dispersive_agent.py`

The dispersive agent adds diversity-promoting regularization:

```python
# To customize the dispersive loss, override get_loss():
class TrainDriftingDispersiveAgent(TrainDriftingAgent):
    def get_loss(self, batch_data):
        actions, obs = batch_data
        # Standard drifting loss
        loss = self.model.loss(x1=actions, cond=obs)
        
        # Add your custom regularization here
        # Example: entropy bonus on action predictions
        # with torch.no_grad():
        #     x0 = torch.randn_like(actions)
        #     pred = self.model.network(x0, t=1, r=0, obs)
        #     entropy = -torch.mean(pred ** 2)
        # loss = loss - 0.01 * entropy
        
        return loss
```

---

## 3. Modifying PPO Fine-tuning

### 3.1 Changing the Exploration Noise

**File to edit:** `model/drifting/ft_ppo/ppodrifting.py`, class `NoisyDriftingMLP`

```python
# model/drifting/ft_ppo/ppodrifting.py

class NoisyDriftingMLP(torch.nn.Module):
    def forward(self, x, cond, ...):
        # Step 1: Deterministic mean from drifting policy
        mean = self.policy(x, t=1.0, r=0.0, cond=cond)
        
        # Step 2: Learned noise std (modify noise source here)
        log_var = self.MLP_logvar(mean)  # ← Change noise network
        std = torch.exp(0.5 * log_var).clamp(min_std, max_std)
        
        # Step 3: Stochastic action
        noise = torch.randn_like(mean)
        action = mean + std * noise
        
        # Example: Use OU noise instead of Gaussian
        # if hasattr(self, 'ou_state'):
        #     self.ou_state = 0.85 * self.ou_state + 0.15 * noise
        #     action = mean + std * self.ou_state
```

### 3.2 Changing the Log-Probability Computation

**File to edit:** `model/drifting/ft_ppo/ppodrifting.py`, class `PPODrifting`

The PPO objective needs accurate log-probabilities. For Drifting Policy, this is simplified to a single Gaussian:

```python
# model/drifting/ft_ppo/ppodrifting.py, class PPODrifting

def get_logprobs(self, cond, chains, ...):
    """
    chains: [B, 2, T_a, D_a]  - always length 2 for drifting (noise + action)
    
    Log-prob computation:
    1. Extract initial noise z and generated action a from chain
    2. Compute mean = network(z, t=1, r=0, cond)
    3. Compute std from noise network
    4. log_prob = Normal(mean, std).log_prob(a).sum()
    """
    z = chains[:, 0]  # Initial noise
    a = chains[:, 1]  # Generated action
    
    # Forward through noisy drifting MLP
    mean, std = self.actor_ft(z, cond, ...)
    
    # Gaussian log-probability
    dist = Normal(mean, std)
    log_prob = dist.log_prob(a).sum(dim=(-1, -2))
    
    # Example: Add Jacobian correction for bounded actions
    # if self.use_tanh_squashing:
    #     log_prob -= (2 * (math.log(2) - a - F.softplus(-2*a))).sum(dim=(-1,-2))
```

### 3.3 Changing the PPO Loss Components

**File to edit:** `model/flow/ft_ppo/ppoflow.py`, method `loss()`

The PPO loss in `PPOFlow` (parent of `PPODrifting`) combines:

```python
# Total loss = pg_loss + ent_coef * entropy_loss + vf_coef * v_loss + bc_coeff * bc_loss

# To modify BC loss (behavioral cloning regularization):
# File: model/flow/ft_ppo/ppoflow.py, around line 500
if bc_loss_type == 'W2':
    bc_loss = wasserstein_distance(...)
elif bc_loss_type == 'velocity':
    bc_loss = velocity_prediction_loss(...)
# Add your custom BC loss type here:
# elif bc_loss_type == 'my_custom_bc':
#     bc_loss = my_custom_function(...)
```

### 3.4 Modifying the PPO Agent Training Loop

**File to edit:** `agent/finetune/reinflow/train_ppo_drifting_agent.py`

For state-based tasks, or `agent/finetune/reinflow/train_ppo_drifting_img_agent.py` for image-based tasks.

Key methods to override:
- `agent_update()` — the PPO gradient update step
- `run()` — the main training loop
- `__init__()` — initialization and hyperparameter setup

---

## 4. Modifying GRPO Fine-tuning

### 4.1 Changing the Action Distribution

**File to edit:** `model/drifting/ft_grpo/grpodrifting.py`, class `NoisyDriftingPolicy`

```python
# model/drifting/ft_grpo/grpodrifting.py

class NoisyDriftingPolicy(torch.nn.Module):
    def get_log_prob(self, cond, action):
        """
        Computes log-probability under Tanh-Normal distribution.
        
        To change the distribution:
        1. Modify get_distribution() to return a different dist
        2. Update the Jacobian correction accordingly
        """
        mean, std = self.forward(cond)
        dist = Normal(mean, std)
        
        # Inverse tanh to get pre-squash value
        u = torch.atanh(action.clamp(-TANH_CLIP_THRESHOLD, TANH_CLIP_THRESHOLD))
        
        # Log-prob with Jacobian correction
        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= _tanh_jacobian_correction(u).sum(dim=-1)  # ← Key correction
        
        return log_prob
```

### 4.2 Changing the KL Divergence Computation

**File to edit:** `model/drifting/ft_grpo/grpodrifting.py`, class `GRPODrifting`

```python
# model/drifting/ft_grpo/grpodrifting.py, class GRPODrifting

def compute_loss(self, obs, actions, advantages, old_log_probs):
    # Current: Analytical KL for Gaussians (pre-tanh space)
    # KL(N(mu_curr, sigma_curr) || N(mu_ref, sigma_ref))
    kl = (
        torch.log(sigma_ref / sigma_curr)
        + (sigma_curr**2 + (mu_curr - mu_ref)**2) / (2 * sigma_ref**2)
        - 0.5
    ).sum(dim=-1).mean()
    
    # To use sampled KL instead of analytical:
    # kl = (curr_log_prob - ref_log_prob).mean()
    
    # To use reverse KL:
    # kl_reverse = (
    #     torch.log(sigma_curr / sigma_ref)
    #     + (sigma_ref**2 + (mu_ref - mu_curr)**2) / (2 * sigma_curr**2)
    #     - 0.5
    # ).sum(dim=-1).mean()
```

### 4.3 Changing the Advantage Computation

**File to edit:** `agent/finetune/grpo/buffer.py`, class `GRPOBuffer`

```python
# agent/finetune/grpo/buffer.py

ADVANTAGE_STD_THRESHOLD = 1e-6  # ← Zero-variance protection threshold

class GRPOBuffer:
    def normalize_advantages(self):
        """
        Current: Z-score normalization within group
        advantages = (returns - mean) / (std + eps)
        
        To modify advantage computation:
        """
        mean = self.returns.mean()
        std = self.returns.std(unbiased=False)  # Population std
        
        if std < ADVANTAGE_STD_THRESHOLD:
            # Zero-variance protection: all trajectories got same reward
            self.advantages = torch.zeros_like(self.returns)
        else:
            self.advantages = (self.returns - mean) / (std + ADVANTAGE_STD_THRESHOLD)
        
        # Example: Use percentile-based normalization instead
        # median = self.returns.median()
        # iqr = self.returns.quantile(0.75) - self.returns.quantile(0.25)
        # self.advantages = (self.returns - median) / (iqr + eps)
```

### 4.4 Changing the Beta Decay Schedule

**File to edit:** `agent/finetune/grpo/train_grpo_drifting_agent.py`

```python
# agent/finetune/grpo/train_grpo_drifting_agent.py

def update_beta(self):
    """
    Current: Exponential decay
    beta = max(beta * decay, beta_min)
    """
    self.beta = max(self.beta * self.beta_decay, self.beta_min)
    
    # Example: Cosine annealing
    # progress = self.itr / self.n_train_itr
    # self.beta = self.beta_min + 0.5 * (self.beta_init - self.beta_min) * (1 + math.cos(math.pi * progress))
```

---

## 5. Adding a New Environment

### 5.1 Create Config Files

Create three YAML files in the appropriate directory:

```bash
cfg/{env_suite}/pretrain/{task}/pre_drifting_mlp.yaml
cfg/{env_suite}/finetune/{task}/ft_ppo_drifting_mlp.yaml
cfg/{env_suite}/finetune/{task}/ft_grpo_drifting_mlp.yaml
cfg/{env_suite}/eval/{task}/eval_drifting_mlp.yaml
```

### 5.2 Config Template for a New Task

```yaml
# cfg/my_suite/pretrain/my_task/pre_drifting_mlp.yaml
env_suite: my_suite
env: my_task
action_dim: 6                  # ← Your action dimension
horizon_steps: 4               # ← Action chunk size
obs_dim: 20                    # ← Your observation dimension
cond_steps: 1

_target_: agent.pretrain.train_drifting_agent.TrainDriftingAgent

train:
  n_epochs: 40
  batch_size: 128
  learning_rate: 1e-3

model:
  _target_: model.drifting.drifting.DriftingPolicy
  network:
    _target_: model.flow.mlp_meanflow.MeanFlowMLP
    action_dim: ${action_dim}
    horizon_steps: ${horizon_steps}
    cond_dim: ${obs_dim}
    mlp_dims: [512, 512, 512]
    activation_type: Mish
  act_min: -1
  act_max: 1
  max_denoising_steps: 1
  drift_coef: 0.1
  neg_drift_coef: 0.05
  mask_self: false
  bandwidth: 1.0

train_dataset:
  _target_: agent.dataset.sequence.StitchedSequenceDataset
  dataset_path: ${oc.env:REINFLOW_DATA_DIR}/my_suite/my_task/train.npz
  horizon_steps: ${horizon_steps}
  cond_steps: ${cond_steps}

ema:
  decay: 0.995

test_in_mujoco: false
```

---

## 6. Key Constants & Parameters

### 6.1 Drifting Policy Constants

| Constant | Value | File | Purpose |
|----------|-------|------|---------|
| `drift_coef` | 0.1 | `model/drifting/drifting.py` | Positive drift strength |
| `neg_drift_coef` | 0.05 | `model/drifting/drifting.py` | Negative drift strength |
| `bandwidth` | 1.0 | `model/drifting/drifting.py` | RBF kernel bandwidth |

### 6.2 GRPO Constants

| Constant | Value | File | Purpose |
|----------|-------|------|---------|
| `LOG_2` | `ln(2)` | `model/drifting/ft_grpo/grpodrifting.py` | Tanh Jacobian |
| `TANH_CLIP_THRESHOLD` | 0.999999 | `model/drifting/ft_grpo/grpodrifting.py` | Numerical stability |
| `JACOBIAN_EPS` | 1e-6 | `model/drifting/ft_grpo/grpodrifting.py` | Division safety |
| `ADVANTAGE_STD_THRESHOLD` | 1e-6 | `agent/finetune/grpo/buffer.py` | Zero-variance guard |

### 6.3 PPO Constants

| Parameter | Typical Value | File |
|-----------|--------------|------|
| `min_sampling_denoising_std` | varies | `model/drifting/ft_ppo/ppodrifting.py` |
| `min_logprob_denoising_std` | varies | `model/drifting/ft_ppo/ppodrifting.py` |
| `inital_noise_scheduler_type` | varies | `model/drifting/ft_ppo/ppodrifting.py` |

> **Note:** The parameter name `inital_noise_scheduler_type` (missing an 'i') is an intentional legacy spelling kept for backward compatibility across all PPO flow models.

---

## 7. Debugging Guide

### 7.1 NaN in Loss

**Symptom:** Loss becomes `NaN` during training.

**Check these locations:**
1. `model/drifting/drifting.py:compute_V()` — RBF kernel weights can overflow with large distances
2. `model/drifting/ft_grpo/grpodrifting.py:_tanh_jacobian_correction()` — extreme action values
3. `agent/finetune/grpo/buffer.py:normalize_advantages()` — zero-variance division

**Fix:** Add gradient clipping, reduce learning rate, or increase `bandwidth`.

### 7.2 Ratio ≠ 1.0 at First Batch

**Symptom:** PPO ratio is not exactly 1.0 at `update_epoch=0, batch_id=0`.

**This indicates a bug** in the log-probability computation. The old and new log-probs should be identical before any gradient update.

**Check:** `model/drifting/ft_ppo/ppodrifting.py:get_logprobs()` — ensure the chain indexing is correct and noise std computation is deterministic.

### 7.3 KL Divergence Explosion in GRPO

**Symptom:** KL divergence grows rapidly, policy diverges from reference.

**Fix options:**
1. Increase `kl_beta` (stronger constraint)
2. Decrease `grpo_lr` (slower policy changes)
3. Enable KL early stopping:
   ```yaml
   train:
     target_kl: 0.01
     lr_schedule: adaptive_kl
   ```

### 7.4 Monitoring Training Health

Key metrics to watch in WandB:

| Metric | Healthy Range | Problem If |
|--------|---------------|------------|
| `noise_std` | 0.01–1.0 | Near 0 (no exploration) or >5 (unstable) |
| `approx_kl` | 0.001–0.05 | >0.1 (too fast divergence) |
| `clipfrac` | 0.05–0.3 | >0.5 (clipping too often) |
| `ratio` | 0.8–1.2 | <0.5 or >2.0 (unstable) |
| `entropy_loss` | varies | Monotonically decreasing → premature convergence |
