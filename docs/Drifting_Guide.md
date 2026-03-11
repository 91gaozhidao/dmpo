# Drifting Policy Guide

The **Drifting Policy** is a 1-NFE (one Neural Function Evaluation) generative model that uses a "drifting field" to map noise directly to expert actions in a single forward pass. It is designed for high-efficiency inference while maintaining the multi-modal modeling capabilities of flow-based models.

---

## 1. Mathematical Foundation

### 1.1 Background: Continuous-Time ODE for Flow Matching

Standard flow matching models (e.g., MeanFlow) define a continuous probability path via an Ordinary Differential Equation (ODE):

$$\frac{dx_t}{dt} = v_\theta(x_t, t, s), \quad t \in [0, 1]$$

where $v_\theta$ is a learned velocity field conditioned on state $s$, and $x_0 \sim \mathcal{N}(0, I)$ is Gaussian noise. Inference requires solving this ODE via numerical integration (e.g., Euler method), yielding $K$-NFE where $K \geq 1$ is the number of discretization steps.

**MeanFlow** improves upon this by learning an average velocity $u_\theta(x_t, t, r, s)$ that directly maps from time $t$ to time $r$:

$$x_r = x_t - (t - r) \cdot u_\theta(x_t, t, r, s)$$

This allows flexible step counts but the network must still learn the full velocity field.

### 1.2 Drifting Policy: Front-Loading Mean-Shift to Training

Drifting Policy takes a fundamentally different approach by **front-loading** the iterative generation process into the training phase. Instead of learning a velocity field that must be integrated at inference, it learns a direct mapping $f_\theta: \mathcal{N}(0, I) \times \mathcal{S} \to \mathcal{A}$ that produces actions in a single forward pass.

**Training Phase:** Given a batch of expert actions $\{a_i^*\}_{i=1}^B$ and noise samples $\{z_i\}_{i=1}^B$ where $z_i \sim \mathcal{N}(0, I)$:

1. Compute network prediction: $\hat{a}_i = f_\theta(z_i, s_i)$ with fixed $t=1, r=0$
2. Compute the drifting field $V_{\text{total}}$ that drives predictions toward expert data:

$$V_{\text{pos},i} = \alpha_+ \sum_{j} w_{ij}^+ (a_j^* - \hat{a}_i), \quad w_{ij}^+ = \frac{\exp(-\|{\hat{a}_i - a_j^*}\|^2 / 2h^2)}{\sum_k \exp(-\|{\hat{a}_i - a_k^*}\|^2 / 2h^2)}$$

$$V_{\text{neg},i} = \alpha_- \sum_{j} w_{ij}^- (\hat{a}_i - a_j^-), \quad w_{ij}^- = \frac{\exp(-\|{\hat{a}_i - a_j^-}\|^2 / 2h^2)}{\sum_k \exp(-\|{\hat{a}_i - a_k^-}\|^2 / 2h^2)}$$

3. Construct the drifted target: $\tilde{a}_i = \hat{a}_i + V_{\text{pos},i} + V_{\text{neg},i}$
4. Minimize: $\mathcal{L} = \frac{1}{B}\sum_i \| f_\theta(z_i, s_i) - \text{sg}(\tilde{a}_i) \|^2$ where $\text{sg}(\cdot)$ is stop-gradient

**Inference Phase (1-NFE):**
$$a = \text{clamp}\left(f_\theta(z, s)\big|_{t=1, r=0},\ [-1, 1]\right), \quad z \sim \mathcal{N}(0, I)$$

### 1.3 RBF Kernel and Drifting Field

The RBF kernel weights $w_{ij}$ ensure smooth, distance-sensitive interactions:
- Nearby expert samples exert stronger attraction (positive drift)
- Nearby negative samples exert stronger repulsion (negative drift)
- The bandwidth $h$ controls the interaction range

When `mask_self=True`, self-interactions ($i = j$) are excluded to prevent trivial solutions.

---

## 2. Architecture & Module Dependencies

```
model/drifting/
├── drifting.py                 # DriftingPolicy: core 1-NFE policy
│   ├── Uses: model.flow.mlp_meanflow.MeanFlowMLP (state-only)
│   └── Uses: model.flow.mlp_meanflow.MeanFlowViT (vision-based)
├── ft_ppo/
│   └── ppodrifting.py          # PPODrifting: PPO fine-tuning wrapper
│       ├── NoisyDriftingMLP: adds learnable exploration noise
│       └── Inherits: model.flow.ft_ppo.ppoflow.PPOFlow
└── ft_grpo/
    └── grpodrifting.py         # GRPODrifting: critic-free GRPO wrapper
        ├── NoisyDriftingPolicy: Tanh-Normal policy with Jacobian correction
        └── Uses: agent.finetune.grpo.buffer.GRPOBuffer
```

**Backbone Network Sharing:** Drifting Policy uses the same `MeanFlowMLP` and `MeanFlowViT` networks as MeanFlow, ensuring architectural alignment for controlled comparisons. The network receives inputs `(z, t=1.0, r=0.0, cond)` and outputs an action prediction of shape `(B, Ta, Da)`.

---

## 3. Key Hyperparameters

| Parameter             | Default | Description                                                                |
| --------------------- | ------- | -------------------------------------------------------------------------- |
| `drift_coef`          | 0.1     | Scale factor $\alpha_+$ for positive drift (towards expert actions).       |
| `neg_drift_coef`      | 0.05    | Scale factor $\alpha_-$ for negative drift (away from non-expert samples). |
| `mask_self`           | `False` | Whether to exclude self-interactions ($i=j$) in the drift field.           |
| `bandwidth`           | 1.0     | RBF kernel bandwidth $h$ controlling interaction range.                    |
| `max_denoising_steps` | 1       | **Must be 1**. Enforced at runtime; increasing this violates 1-NFE.        |

### PPO-Specific Parameters

| Parameter                       | Default  | Description                                      |
| ------------------------------- | -------- | ------------------------------------------------ |
| `min_logprob_denoising_std`     | 0.05     | Minimum exploration noise std for log-prob.       |
| `max_logprob_denoising_std`     | 0.12     | Maximum exploration noise std for log-prob.       |
| `use_time_independent_noise`    | `True`   | Use time-independent noise (appropriate for 1-NFE). |
| `denoising_steps`               | 1        | Must match the 1-NFE constraint.                 |

### GRPO-Specific Parameters

| Parameter           | Default | Description                                              |
| ------------------- | ------- | -------------------------------------------------------- |
| `group_size`        | 16      | Number of trajectories $G$ per group ($G \geq 16$).      |
| `kl_beta`           | 0.05    | Initial KL divergence penalty coefficient.               |
| `kl_beta_min`       | 0.001   | Minimum KL penalty after decay.                          |
| `kl_beta_decay`     | 0.995   | Exponential decay rate for KL penalty per iteration.     |
| `epsilon`           | 0.2     | PPO-style clipping range for the surrogate loss.         |
| `init_log_std`      | -0.5    | Initial log standard deviation for exploration noise.    |

---

## 4. Available Configurations

### Offline Pre-training

| Domain    | Config Name              | Network    | Command Directory           |
| --------- | ------------------------ | ---------- | --------------------------- |
| Gym       | `pre_drifting_mlp`       | MeanFlowMLP | `cfg/gym/pretrain/<task>/`  |
| Robomimic | `pre_drifting_mlp_img`   | MeanFlowViT | `cfg/robomimic/pretrain/<task>/` |
| D3IL      | `pre_drifting_mlp`       | MeanFlowMLP | `cfg/d3il/pretrain/<task>/` |
| Furniture | `pre_drifting_mlp`       | MeanFlowMLP | `cfg/furniture/pretrain/<task>/` |

### PPO Fine-tuning

| Domain    | Config Name                  | Command Directory              |
| --------- | ---------------------------- | ------------------------------ |
| Gym       | `ft_ppo_drifting_mlp`        | `cfg/gym/finetune/<task>/`     |
| Robomimic | `ft_ppo_drifting_mlp_img`    | `cfg/robomimic/finetune/<task>/` |
| D3IL      | `ft_ppo_drifting_mlp`        | `cfg/d3il/finetune/<task>/`    |
| Furniture | `ft_ppo_drifting_mlp`        | `cfg/furniture/finetune/<task>/` |

### GRPO Fine-tuning (Critic-Free)

| Domain    | Config Name                  | Command Directory              |
| --------- | ---------------------------- | ------------------------------ |
| Gym       | `ft_grpo_drifting_mlp`       | `cfg/gym/finetune/<task>/`     |
| Robomimic | `ft_grpo_drifting_mlp_img`   | `cfg/robomimic/finetune/<task>/` |
| D3IL      | `ft_grpo_drifting_mlp`       | `cfg/d3il/finetune/<task>/`    |
| Furniture | `ft_grpo_drifting_mlp`       | `cfg/furniture/finetune/<task>/` |

---

## 5. Training Stages

### Stage 1: Pre-training

Trains the base drifting field using behavior cloning on expert data.

**Gym (state-based):**
```bash
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp
```

**Robomimic (image-based):**
```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_drifting_mlp_img
```

### Stage 2: Fine-tuning

**PPO Fine-tuning:**
```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_drifting_mlp \
  base_policy_path=<PRETRAINED_CHECKPOINT_PATH>
```

**GRPO Fine-tuning (Critic-free, group-based):**
GRPO eliminates the Critic/Value network. Advantages are computed via intra-group Z-score normalization:
$$A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}, \quad \sigma_R = \text{std}(\{R_1, \ldots, R_G\}), \quad \epsilon = 10^{-6}$$

When $\sigma_R < \epsilon$ (e.g., all-zero returns in sparse reward tasks), advantages are set to zero to prevent division-by-zero errors.

```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_grpo_drifting_mlp \
  base_policy_path=<PRETRAINED_CHECKPOINT_PATH>
```

---

## 6. Runtime Assertions & Safety

The implementation includes runtime assertions to ensure correctness:

1. **Action Boundary Verification:** After `sample()`, actions are verified to lie in $[-1, 1]$.
2. **Log-Probability Finiteness:** In both PPO and GRPO, log-probabilities are checked for NaN/Inf after Jacobian correction.
3. **Zero-Variance Protection:** In GRPO, groups with identical returns ($\sigma_R < 10^{-6}$) produce zero advantages instead of NaN.

---

## 7. Tuning Recommendations

1.  **Stability**: If actions are too jittery, decrease `drift_coef` (e.g., from 0.1 to 0.05).
2.  **Exploration**: If the model collapses to a single mode, increase `neg_drift_coef` to push samples apart.
3.  **Visual Encoder**: For image-based tasks, ensure `img_cond_steps` matches your pre-training setup.
4.  **Inference steps**: Always verify `denoising_steps=1` and `ft_denoising_steps=1` in fine-tuning configs.
5.  **GRPO group size**: Use $G \geq 16$ for reliable advantage estimation. Smaller groups increase variance.
6.  **KL decay**: For sparse reward tasks (e.g., Adroit), use slower decay (`kl_beta_decay=0.999`).
