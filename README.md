# DMPO: Dispersive MeanFlow Policy Optimization

One-step generative policy with critic-guided online RL fine-tuning for continuous control.

## Overview

DMPO combines **Drifting Policy** (a single-step generative model inspired by *Generative Modeling via Drifting*) with **Q-Guided online RL fine-tuning** for embodied AI tasks. The framework supports:

- **Offline pre-training** via the drifting field algorithm (1-NFE / single forward pass inference).
- **Online fine-tuning** via a critic-guided drifting update that uses Q-values to select high-value positive samples for the drifting field, without converting drifting into a standard Gaussian policy.
- **Multiple backbones**: Transformer (`TransformerForDrifting`) and U-Net 1D (`ConditionalUnet1D`), with optional ViT vision encoding for image-based tasks.

### Drifting Policy in This Repository

The core Drifting Policy (`model/drifting/drifting.py`) implements the algorithm from the paper *Generative Modeling via Drifting* (Kaiming He):

- **Exp kernel**: `k(x, y) = exp(-||x - y|| / T)` with bi-directional normalization.
- **Second-order weighting** for balanced positive/negative contributions.
- **Temperature auto-scaling** by mean pairwise distance for robustness.
- **1-NFE inference**: a single forward pass maps noise to actions at test time.

The **Q-Guided Drifting** fine-tuning (`model/drifting/ft_qguided/qguided_drifting.py`) preserves drifting's native update mechanism:

1. Sample multiple candidate actions from the current actor.
2. Evaluate each candidate with a learned Q-function (double-Q critic with target network).
3. Select top-K candidates as the positive set `p(a|s)`.
4. Compute the conditioned drifting field `V_{p,q}` using these high-value positives.
5. Update the actor via `x ← x + V_{p,q}(x)` (drifting's native MSE regression toward the drifted target).

This design keeps drifting's generative mechanism intact rather than replacing it with policy-gradient methods.

## Core Method Overview

### Offline Pre-training (Drifting Field)

```
Training:   L = E_ε [ || f_θ(ε) - sg(f_θ(ε) + V_{p,q}(f_θ(ε))) ||² ]
Inference:  a = f_θ(ε),  ε ~ N(0, I)   [single forward pass]
```

The drifting field `V_{p,q}` tells each generated sample where to move. Positive samples come from expert data; negative samples come from the generator itself. As training converges, `V → 0` and the model directly maps noise to expert-like actions.

### Online Fine-tuning (Q-Guided Drifting)

```
1. Sample N candidates: {a_i}_{i=1}^N ~ actor(s)
2. Score each: Q(s, a_i) via double-Q critic
3. Select top-K as positives (high Q-value)
4. Compute V_{p,q} with positives vs. all candidates
5. Actor loss: MSE(x_query, sg(x_query + V))
6. Critic loss: standard Bellman TD error
```

The critic guides *which* samples become the target distribution, but the actor update itself remains drifting's native field-based regression.

## Supported Tasks and Configuration Matrix

### Gym / MuJoCo Locomotion

| Task | Env Name | Obs Dim | Act Dim | Horizon | Pretrain | Fine-tune | Eval |
|------|----------|---------|---------|---------|----------|-----------|------|
| Hopper | `hopper-medium-v2` | 11 | 3 | 4 | ✓ | ✓ | ✓ |
| Walker2d | `walker2d-medium-v2` | 17 | 6 | 4 | ✓ | ✓ | ✓ |
| Ant | `ant-medium-expert-v2` | 111 | 8 | 4 | ✓ | ✓ | ✓ |
| Humanoid | `Humanoid-medium-v3` | 376 | 17 | 4 | ✓ | ✓ | ✓ |

### Gym / Kitchen

| Task | Env Name | Obs Dim | Act Dim | Horizon | Pretrain | Fine-tune | Eval |
|------|----------|---------|---------|---------|----------|-----------|------|
| Kitchen Complete | `kitchen-complete-v0` | 60 | 9 | 4 | ✓ | ✓ | ✓ |
| Kitchen Partial | `kitchen-partial-v0` | 60 | 9 | 4 | ✓ | ✓ | ✓ |
| Kitchen Mixed | `kitchen-mixed-v0` | 60 | 9 | 4 | ✓ | ✓ | ✓ |

### RoboMimic (Image-based)

| Task | Env Name | Obs Dim | Act Dim | Horizon | Pretrain | Fine-tune | Eval |
|------|----------|---------|---------|---------|----------|-----------|------|
| Lift | `lift` | 9 | 7 | 4 | ✓ | ✓ | ✓ |
| Can | `can` | 9 | 7 | 4 | ✓ | ✓ | ✓ |
| Square | `square` | 14 | 7 | 4 | ✓ | ✓ | ✓ |
| Transport | `transport` | 41 | 14 | 4 | ✓ | ✓ | ✓ |

### Backbone Options

| Backbone | Config Suffix | Low-dim | Image |
|----------|--------------|---------|-------|
| TransformerForDrifting | `_transformer` | ✓ | ✓ (via DriftingViTWrapper) |
| ConditionalUnet1D | `_unet1d` | ✓ | ✓ (via DriftingViTWrapper) |

## Code Structure

```
dmpo/
├── agent/
│   ├── pretrain/
│   │   ├── train_agent.py              # Base pre-training agent
│   │   └── train_drifting_agent.py     # Drifting pre-training agent
│   ├── finetune/
│   │   ├── train_agent.py              # Base fine-tuning agent
│   │   └── drifting/
│   │       └── train_qguided_drifting_agent.py  # Q-Guided fine-tuning agent
│   ├── eval/
│   │   └── eval_drifting_agent.py      # Drifting evaluation agent
│   └── dataset/
│       └── sequence.py                 # StitchedSequenceDataset / QLearningDataset
│
├── model/
│   ├── drifting/
│   │   ├── drifting.py                 # Core: DriftingPolicy, compute_V, compute_drifting_loss
│   │   ├── backbone/
│   │   │   ├── transformer_for_drifting.py  # Transformer backbone (no timestep)
│   │   │   ├── conditional_unet1d.py        # U-Net 1D backbone
│   │   │   └── vit_wrapper.py               # ViT vision wrapper for image tasks
│   │   ├── ft_qguided/
│   │   │   └── qguided_drifting.py     # QGuidedDrifting: critic-guided fine-tuning
│   │   ├── ft_ppo/                     # Legacy PPO wrapper (compatibility only)
│   │   └── ft_grpo/                    # Legacy GRPO wrapper (compatibility only)
│   └── common/
│       └── critic.py                   # CriticObsAct, ViTCriticObsAct
│
├── cfg/
│   ├── templates/                      # Base YAML configs (inherited by task configs)
│   ├── gym/
│   │   ├── pretrain/                   # Per-task pre-training configs
│   │   ├── finetune/                   # Per-task Q-Guided fine-tuning configs
│   │   └── eval/                       # Per-task evaluation configs
│   └── robomimic/
│       ├── pretrain/                   # Per-task pre-training configs (image)
│       ├── finetune/                   # Per-task Q-Guided fine-tuning configs (image)
│       └── eval/                       # Per-task evaluation configs (image)
│
├── env/                                # Environment wrappers (MuJoCo, Kitchen, RoboMimic)
├── script/
│   ├── run.py                          # Main entry point (Hydra launcher)
│   ├── set_path.sh                     # Environment variable setup
│   └── download_dmpo_datasets.py       # Dataset download utility
├── util/                               # Schedulers, timers, HuggingFace download, etc.
├── requirements.txt
└── pyproject.toml
```

## Environment Setup

### 1. Clone and Install

```bash
git clone https://github.com/91gaozhidao/dmpo.git
cd dmpo
pip install -e .
```

### 2. Set Environment Variables

```bash
source script/set_path.sh
```

This sets three required environment variables (you can also set them manually):

```bash
export REINFLOW_DIR=/path/to/dmpo           # Repository root
export REINFLOW_DATA_DIR=/path/to/dmpo/data # Dataset directory
export REINFLOW_LOG_DIR=/path/to/dmpo/log   # Log/checkpoint directory
```

### 3. Install Task-Specific Dependencies

```bash
# For Gym / MuJoCo locomotion
pip install -e ".[gym]"

# For Kitchen tasks (requires MuJoCo + dm_control)
pip install -e ".[kitchen]"

# For RoboMimic (requires robomimic + robosuite)
pip install -e ".[robomimic]"

# For all tasks
pip install -e ".[all]"
```

### 4. WandB Setup (Optional)

```bash
export REINFLOW_WANDB_ENTITY="your_username"
```

To disable WandB logging, pass `wandb=null` when running scripts.

## Data Preparation

### Gym / MuJoCo / Kitchen

Datasets are automatically downloaded from D4RL on first use. The expected directory structure:

```
data/
├── gym/
│   ├── hopper-medium-v2/
│   │   ├── train.npz          # Pre-training data / offline dataset
│   │   └── normalization.npz  # Obs/act normalization statistics
│   ├── walker2d-medium-v2/
│   ├── ant-medium-expert-v2/
│   ├── humanoid-medium-v3/
│   ├── kitchen-complete-v0/
│   ├── kitchen-partial-v0/
│   └── kitchen-mixed-v0/
```

### RoboMimic

Datasets can be downloaded via HuggingFace (`hf://` prefix in configs) or manually placed:

```
data/
├── robomimic/
│   ├── lift/
│   │   ├── train.npz
│   │   └── normalization.npz
│   ├── can/
│   ├── square/
│   └── transport/
```

### Dataset Path Handling

The launcher (`script/run.py`) automatically resolves three types of dataset paths:

- **`train_dataset_path`** — used by pre-training configs.
- **`offline_dataset_path`** — used by Q-guided fine-tuning configs for the offline dataset in mixed (online + offline) batches.
- **`normalization_path`** — observation normalization statistics.

All three support `hf://` prefix for HuggingFace Hub downloads. If a local path does not exist, the launcher falls back to the legacy Google Drive download via `script/download_url.py`.

## Training

### Offline Pre-training

```bash
# Gym / Locomotion (Transformer backbone)
python script/run.py --config-name=pre_drifting_transformer \
    --config-path=cfg/gym/pretrain/hopper-medium-v2

# Gym / Locomotion (UNet1D backbone)
python script/run.py --config-name=pre_drifting_unet1d \
    --config-path=cfg/gym/pretrain/hopper-medium-v2

# RoboMimic / Image (Transformer backbone)
python script/run.py --config-name=pre_drifting_transformer_img \
    --config-path=cfg/robomimic/pretrain/lift
```

### Q-Guided Drifting Fine-tuning

```bash
# Gym / Locomotion (Transformer backbone)
python script/run.py --config-name=ft_qguided_drifting_transformer \
    --config-path=cfg/gym/finetune/hopper-v2

# Gym / Locomotion (UNet1D backbone)
python script/run.py --config-name=ft_qguided_drifting_unet1d \
    --config-path=cfg/gym/finetune/hopper-v2

# RoboMimic / Image (Transformer backbone)
python script/run.py --config-name=ft_qguided_drifting_transformer_img \
    --config-path=cfg/robomimic/finetune/lift
```

### Evaluation

```bash
# Gym
python script/run.py --config-name=eval_drifting_transformer \
    --config-path=cfg/gym/eval/hopper-medium-v2 \
    base_policy_path=/path/to/checkpoint.pt

# RoboMimic
python script/run.py --config-name=eval_drifting_transformer_img \
    --config-path=cfg/robomimic/eval/lift \
    base_policy_path=/path/to/checkpoint.pt
```

## Checkpoint System

### Pre-training Checkpoints

Saved to `{REINFLOW_LOG_DIR}/{domain}/pretrain/{env_name}/{config_name}/{timestamp}/checkpoint/`:

- `state_{step}.pt` — Model state dict at training step.
- Contains both `model` and `ema` keys (EMA model is used for evaluation by default).

### Fine-tuning Checkpoints

Saved to `{REINFLOW_LOG_DIR}/{domain}/finetune/{env_name}/{config_name}/{timestamp}/checkpoint/`:

- `state_{itr}.pt` — Full agent state (actor, critic, target critic, optimizers).

### Loading Checkpoints

Fine-tuning configs specify `base_policy_path` pointing to a pre-trained checkpoint. The Q-Guided system auto-detects checkpoint format and extracts the drifting backbone weights, supporting:

- Direct `DriftingPolicy` checkpoints.
- Legacy `PPODrifting` or `GRPODrifting` checkpoints (extracts the drifting backbone).
- HuggingFace-hosted checkpoints (`hf://` prefix).

## Log and Output Structure

```
log/
├── gym/
│   ├── pretrain/{env_name}/{config_name}/{timestamp}/
│   │   ├── checkpoint/          # Saved model weights
│   │   ├── .hydra/config.yaml   # Full resolved config
│   │   └── result.pkl           # Training metrics
│   ├── finetune/{env_name}/{config_name}/{timestamp}/
│   │   ├── checkpoint/
│   │   ├── .hydra/config.yaml
│   │   └── result.pkl
│   └── eval/{config_name}/{timestamp}/
└── robomimic/
    └── ... (same structure)
```

## Common Issues and Troubleshooting

### Environment Variable Errors

```
InterpolationResolutionError: Could not resolve 'oc.env:REINFLOW_LOG_DIR'
```
**Fix**: Run `source script/set_path.sh` or manually `export REINFLOW_LOG_DIR=...`.

### D4RL Import Errors

```
ImportError: d4rl gym_mujoco
```
**Fix**: `pip install -e ".[gym]"` and ensure MuJoCo is installed.

### CUDA / Rendering Issues

```
RuntimeError: MUJOCO_GL not set
```
**Fix**: The launcher auto-detects; for headless servers use `sim_device=cuda:0` in config or `export MUJOCO_GL=egl`.

### Non-Contiguous Tensor Errors in Critic

If you encounter errors like:
```
RuntimeError: view size is not compatible with input tensor's size and stride
```
This has been fixed in `model/common/critic.py` by using `.reshape()` instead of `.view()`.

### WandB Errors

**Fix**: Either set `export REINFLOW_WANDB_ENTITY=your_name` or pass `wandb=null` on the command line.

## License

MIT License — Copyright (c) 2025 ReinFlow Authors. See [LICENSE](LICENSE) for details.
