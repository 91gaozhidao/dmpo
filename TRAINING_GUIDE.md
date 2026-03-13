# DMPO Training Guide

A comprehensive guide for offline pre-training, Q-Guided online fine-tuning, evaluation, and troubleshooting with Drifting Policy.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Offline Pre-training](#offline-pre-training)
3. [Q-Guided Drifting Fine-tuning](#q-guided-drifting-fine-tuning)
4. [Evaluation](#evaluation)
5. [RoboMimic Training Steps](#robomimic-training-steps)
6. [Gym / Kitchen Training Steps](#gym--kitchen-training-steps)
7. [Configuration Reference](#configuration-reference)
8. [Resume and Continue Training](#resume-and-continue-training)
9. [Key Metrics and Interpretation](#key-metrics-and-interpretation)
10. [Common Failure Modes and Fixes](#common-failure-modes-and-fixes)
11. [Recommended Validation Order](#recommended-validation-order)

---

## Prerequisites

### Environment Variables

```bash
source script/set_path.sh
# Or manually:
export REINFLOW_DIR=/path/to/dmpo
export REINFLOW_DATA_DIR=/path/to/dmpo/data
export REINFLOW_LOG_DIR=/path/to/dmpo/log
```

### Dependencies

```bash
# Core
pip install -e .

# Task-specific (choose one or more)
pip install -e ".[gym]"        # MuJoCo locomotion (hopper, walker2d, ant, humanoid)
pip install -e ".[kitchen]"    # Kitchen tasks (requires dm_control + mujoco)
pip install -e ".[robomimic]"  # RoboMimic (requires robomimic + robosuite)
```

### WandB (Optional)

```bash
export REINFLOW_WANDB_ENTITY="your_username"
# Or disable: pass wandb=null on command line
```

### GPU

- Fine-tuning with environment rollouts requires a GPU with MuJoCo rendering support.
- Pre-training can run on any CUDA-capable GPU.
- Set `device: cuda:0` in the config or pass `device=cuda:0` on the command line.

---

## Offline Pre-training

### How It Works

Offline pre-training teaches the Drifting Policy to imitate expert demonstrations using the drifting field algorithm:

1. **Input**: Gaussian noise `ε ~ N(0, I)`.
2. **Forward**: Network maps `ε → f_θ(ε)` (predicted actions).
3. **Drifting field**: `V_{p,q}(x)` is computed between generated samples and expert data.
4. **Loss**: `L = MSE(f_θ(ε), sg(f_θ(ε) + V_{p,q}(f_θ(ε))))` — move predictions toward drifted targets.

After convergence, the model directly generates expert-like actions in a single forward pass (1-NFE).

### Training Commands

#### Gym / Locomotion

```bash
# Hopper (Transformer)
python script/run.py --config-name=pre_drifting_transformer \
    --config-path=cfg/gym/pretrain/hopper-medium-v2

# Walker2d (UNet1D)
python script/run.py --config-name=pre_drifting_unet1d \
    --config-path=cfg/gym/pretrain/walker2d-medium-v2

# Ant (Transformer)
python script/run.py --config-name=pre_drifting_transformer \
    --config-path=cfg/gym/pretrain/ant-medium-expert-v2

# Humanoid (Transformer)
python script/run.py --config-name=pre_drifting_transformer \
    --config-path=cfg/gym/pretrain/humanoid-medium-v3
```

#### Kitchen

```bash
# Kitchen Complete (Transformer)
python script/run.py --config-name=pre_drifting_transformer \
    --config-path=cfg/gym/pretrain/kitchen-complete-v0

# Kitchen Partial (UNet1D)
python script/run.py --config-name=pre_drifting_unet1d \
    --config-path=cfg/gym/pretrain/kitchen-partial-v0

# Kitchen Mixed (Transformer)
python script/run.py --config-name=pre_drifting_transformer \
    --config-path=cfg/gym/pretrain/kitchen-mixed-v0
```

#### RoboMimic (Image-based)

```bash
# Lift (Transformer with ViT)
python script/run.py --config-name=pre_drifting_transformer_img \
    --config-path=cfg/robomimic/pretrain/lift

# Can (UNet1D with ViT)
python script/run.py --config-name=pre_drifting_unet1d_img \
    --config-path=cfg/robomimic/pretrain/can

# Square (Transformer with ViT)
python script/run.py --config-name=pre_drifting_transformer_img \
    --config-path=cfg/robomimic/pretrain/square

# Transport (Transformer with ViT)
python script/run.py --config-name=pre_drifting_transformer_img \
    --config-path=cfg/robomimic/pretrain/transport
```

### What to Expect

- **Training loss** should steadily decrease as `V → 0`.
- **`train/V_norm_mean`**: Should decrease over training. When close to 0, the generator has converged.
- **`train/mean_cross_dist`**: Mean distance between generated and expert samples. Should decrease.
- **`train/drift_magnitude`**: Should decrease over training, indicating convergence.
- Checkpoints are saved periodically to `{REINFLOW_LOG_DIR}/{domain}/pretrain/{env}/{config}/{timestamp}/checkpoint/`.
- The EMA model (exponential moving average) is typically used for evaluation and downstream fine-tuning.

---

## Q-Guided Drifting Fine-tuning

This is the only recommended online RL fine-tuning path for drifting in the
current repository. PPO/GRPO-style adapters are not part of the drifting
training surface here.

### How It Works

Q-Guided fine-tuning improves the pre-trained policy through online environment interaction while preserving drifting's native update mechanism:

1. **Critic Training**: Standard double-Q Bellman TD update with target network EMA.
2. **Actor Update** (drifting-native):
   - Sample `N` candidate actions from the current actor.
   - Evaluate each with `Q(s, a)` via the learned critic.
   - Select top-K candidates as the positive set (high Q-value).
   - Compute the conditioned drifting field `V_{p,q}` using these positives.
   - Actor loss: `MSE(x_query, sg(x_query + V))`.
3. **Target critic**: EMA of the online critic for stable Q-targets.
4. **Offline regularization**: Mixed batches of offline data + online replay buffer data.

### Training Loop Summary

Each iteration:
1. Collect rollouts from the environment (skip during `offline_only_iters` warm-up).
2. Store transitions in the replay buffer.
3. For `updates_per_itr` gradient steps:
   - Sample a mixed batch (offline + online).
   - Update critic (Bellman TD loss).
   - Update actor (Q-guided drifting loss, skipped during critic warmup).
   - Soft-update target critic.
4. Periodically evaluate and checkpoint.

**Dataset paths**: Q-guided fine-tuning configs use `offline_dataset_path` (not `train_dataset_path`) to specify the offline dataset for mixed batches. The launcher (`script/run.py`) automatically resolves `hf://` paths via HuggingFace Hub, and falls back to Google Drive download for missing local paths.

For RoboMimic image tasks, the currently hosted `robomimic/*-img/train.npz`
files may still be legacy pre-training datasets without `rewards` and
`terminals`. The loader now synthesizes zero rewards and episode-end terminals
for smoke/debugging compatibility, but proper Q-guided image training should
use datasets regenerated with `script/dataset/process_robomimic_dataset.py`.

### Pure Online Q-Guided Drifting

Pure online mode keeps the same critic-guided drifting actor update, but drops
the offline reward dataset assumption entirely:

- Use a pre-trained drifting checkpoint via `base_policy_path`.
- Leave `offline_dataset_path: null` and `offline_dataset: null`.
- Set `offline_batch_ratio: 0.0` and `offline_only_iters: 0`.
- Use `min_replay_size` to delay critic/actor updates until the replay buffer
  has enough online transitions.
- If you start from an older mixed fine-tune config, passing
  `train.online_only=true` now also disables offline dataset resolution in the
  launcher.

The new `ft_qguided_drifting_online_*` configs are the recommended starting
point for current drifting online RL runs, especially when demonstrations are
reward-less and only environment rollouts provide rewards.

### Training Commands

#### Gym / Locomotion

```bash
# Hopper (Transformer)
python script/run.py --config-name=ft_qguided_drifting_transformer \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/hopper-v2

# Walker2d (UNet1D)
python script/run.py --config-name=ft_qguided_drifting_unet1d \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/walker2d-v2

# Ant (Transformer)
python script/run.py --config-name=ft_qguided_drifting_transformer \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/ant-v2

# Humanoid (Transformer)
python script/run.py --config-name=ft_qguided_drifting_transformer \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/Humanoid-v3
```

#### Pure Online Gym / Locomotion

```bash
# Hopper (UNet1D, pure online)
python script/run.py --config-name=ft_qguided_drifting_online_unet1d \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/hopper-v2

# Walker2d (UNet1D, pure online)
python script/run.py --config-name=ft_qguided_drifting_online_unet1d \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/walker2d-v2

# Kitchen Partial (UNet1D, pure online)
python script/run.py --config-name=ft_qguided_drifting_online_unet1d \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/kitchen-partial-v0
```

#### Kitchen

```bash
# Kitchen Complete (Transformer)
python script/run.py --config-name=ft_qguided_drifting_transformer \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/kitchen-complete-v0

# Kitchen Partial (UNet1D)
python script/run.py --config-name=ft_qguided_drifting_unet1d \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/kitchen-partial-v0

# Kitchen Mixed (Transformer)
python script/run.py --config-name=ft_qguided_drifting_transformer \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/kitchen-mixed-v0
```

#### RoboMimic (Image-based)

```bash
# Lift (Transformer with ViT)
python script/run.py --config-name=ft_qguided_drifting_transformer_img \
    --config-path=$REINFLOW_DIR/cfg/robomimic/finetune/lift

# Can (UNet1D with ViT)
python script/run.py --config-name=ft_qguided_drifting_unet1d_img \
    --config-path=$REINFLOW_DIR/cfg/robomimic/finetune/can

# Square (Transformer with ViT)
python script/run.py --config-name=ft_qguided_drifting_transformer_img \
    --config-path=$REINFLOW_DIR/cfg/robomimic/finetune/square

# Transport (UNet1D with ViT)
python script/run.py --config-name=ft_qguided_drifting_unet1d_img \
    --config-path=$REINFLOW_DIR/cfg/robomimic/finetune/transport
```

### Setting `base_policy_path`

Fine-tuning requires a pre-trained checkpoint. Set `base_policy_path` in the config or on the command line:

```bash
python script/run.py --config-name=ft_qguided_drifting_transformer \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/hopper-v2 \
    base_policy_path=/path/to/pretrained/state_200000.pt
```

The system auto-detects the checkpoint format (DriftingPolicy, PPODrifting, GRPODrifting, or EMA) and extracts the backbone weights.

HuggingFace-hosted checkpoints are also supported:

```bash
base_policy_path=hf://your_org/your_model/checkpoint.pt
```

---

## Evaluation

### Gym / Locomotion

```bash
python script/run.py --config-name=eval_drifting_transformer \
    --config-path=cfg/gym/eval/hopper-medium-v2 \
    base_policy_path=/path/to/checkpoint.pt

python script/run.py --config-name=eval_drifting_unet1d \
    --config-path=cfg/gym/eval/walker2d-medium-v2 \
    base_policy_path=/path/to/checkpoint.pt
```

### Kitchen

```bash
python script/run.py --config-name=eval_drifting_transformer \
    --config-path=cfg/gym/eval/kitchen-complete-v0 \
    base_policy_path=/path/to/checkpoint.pt
```

### RoboMimic

```bash
python script/run.py --config-name=eval_drifting_transformer_img \
    --config-path=cfg/robomimic/eval/lift \
    base_policy_path=/path/to/checkpoint.pt
```

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_steps` | 500 | Max env steps for evaluation |
| `env.n_envs` | 40 | Parallel eval environments |
| `render_num` | 0 | Number of episodes to render (set > 0 for video) |
| `load_ema` | true | Load EMA weights (recommended) |
| `denoising_steps` | 1 | Always 1 for Drifting Policy |

---

## RoboMimic Training Steps

### Step-by-step for the `lift` task

1. **Install dependencies**:
   ```bash
   pip install -e ".[robomimic]"
   ```

2. **Download dataset**: Datasets are auto-downloaded on first run via HuggingFace or Google Drive.

3. **Pre-train**:
   ```bash
   python script/run.py --config-name=pre_drifting_transformer_img \
       --config-path=cfg/robomimic/pretrain/lift
   ```
   Wait for training to converge (check `train/V_norm_mean` decreasing to near 0).

4. **Fine-tune**:
   ```bash
   python script/run.py --config-name=ft_qguided_drifting_transformer_img \
       --config-path=$REINFLOW_DIR/cfg/robomimic/finetune/lift \
       base_policy_path=/path/to/pretrain_checkpoint.pt
   ```

5. **Evaluate**:
   ```bash
   python script/run.py --config-name=eval_drifting_transformer_img \
       --config-path=cfg/robomimic/eval/lift \
       base_policy_path=/path/to/finetune_checkpoint.pt
   ```

### Task-specific notes

- **Transport**: Has 2 robot arms, `action_dim=14`, `obs_dim=41`. Largest RoboMimic task.
- **Square**: Harder manipulation; may need longer fine-tuning.
- All tasks use `max_episode_steps=400` and image observations from `agentview_image` (and `robot0_eye_in_hand_image` for some).

---

## Gym / Kitchen Training Steps

### Step-by-step for Hopper

1. **Install dependencies**:
   ```bash
   pip install -e ".[gym]"
   ```

2. **Download D4RL data**: Automatically handled on first run.

3. **Pre-train**:
   ```bash
   python script/run.py --config-name=pre_drifting_transformer \
       --config-path=cfg/gym/pretrain/hopper-medium-v2
   ```

4. **Fine-tune**:
   ```bash
   python script/run.py --config-name=ft_qguided_drifting_transformer \
       --config-path=$REINFLOW_DIR/cfg/gym/finetune/hopper-v2 \
       base_policy_path=/path/to/pretrain_checkpoint.pt
   ```

5. **Evaluate**:
   ```bash
   python script/run.py --config-name=eval_drifting_transformer \
       --config-path=cfg/gym/eval/hopper-medium-v2 \
       base_policy_path=/path/to/checkpoint.pt
   ```

### Kitchen tasks

Kitchen environments use `max_episode_steps=280` and success is measured by achieving a threshold reward of 4.0 (number of subtasks completed).

```bash
# Pre-train
python script/run.py --config-name=pre_drifting_transformer \
    --config-path=cfg/gym/pretrain/kitchen-partial-v0

# Fine-tune
python script/run.py --config-name=ft_qguided_drifting_transformer \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/kitchen-partial-v0 \
    base_policy_path=/path/to/checkpoint.pt
```

---

## Configuration Reference

### Pre-training Config (Key Fields)

| Field | Description | Example |
|-------|-------------|---------|
| `model._target_` | Model class | `model.drifting.drifting.DriftingPolicy` |
| `model.network._target_` | Backbone class | `model.drifting.backbone.transformer_for_drifting.TransformerForDrifting` |
| `model.temperatures` | Multi-temperature base values for drift field | `[0.1]` |
| `model.mask_self` | Mask self-interaction in V (True when y_neg = x) | `true` |
| `model.horizon_steps` | Action trajectory length | `4` |
| `model.act_min` / `model.act_max` | Action clipping range | `-1` / `1` |
| `train.n_train_steps` | Total training steps | `200000` |
| `train.batch_size` | Training batch size | `256` |
| `train.learning_rate` | Learning rate | `1e-4` |
| `train_dataset_path` | Path to training data | `data/gym/hopper-medium-v2/train.npz` |

### Fine-tuning Config (Key Fields)

| Field | Description | Example |
|-------|-------------|---------|
| `model._target_` | Model class | `model.drifting.ft_qguided.qguided_drifting.QGuidedDrifting` |
| `model.policy._target_` | Actor policy class | `model.drifting.drifting.DriftingPolicy` |
| `model.critic._target_` | Critic class | `model.common.critic.CriticObsAct` |
| `base_policy_path` | Pre-trained checkpoint path | `/path/to/state_200000.pt` |
| `offline_dataset_path` | Offline dataset for Q-guided fine-tuning (supports `hf://` prefix and local paths) | `data/gym/hopper-medium-v2/train.npz` |
| `normalization_path` | Observation normalization file | `data/gym/hopper-medium-v2/normalization.npz` |
| `train.n_train_itr` | Total training iterations | `1000` |
| `train.batch_size` | Batch size per update | `256` |
| `train.actor_lr` | Actor learning rate | `1e-5` |
| `train.critic_lr` | Critic learning rate | `3e-4` |
| `train.gamma` | Discount factor | `0.99` |
| `train.target_ema_rate` | Target critic EMA rate | `0.005` |
| `train.n_critic_warmup_itr` | Critic-only warmup iterations | `25` |
| `train.updates_per_itr` | Gradient steps per iteration | `8` |
| `train.offline_batch_ratio` | Fraction of offline data in each batch | `0.5` |
| `train.offline_only_iters` | Iterations before starting env rollouts | `25` |
| `train.buffer_size` | Online replay buffer capacity | `200000` |
| `train.scale_reward_factor` | Reward scaling factor | `1.0` |
| `model.num_action_samples` | Candidates per Q-guided selection | `8` |
| `model.num_positive_samples` | Top-K samples as positives | `2` |
| `model.num_query_samples` | Query samples for actor loss | `4` |
| `model.sample_latent_scale` | Noise scale during training | `1.0` |
| `model.eval_latent_scale` | Noise scale during evaluation | `0.0` |
| `model.reference_anchor_coeff` | Reference policy anchor weight (0 = off) | `0.0` |

### Environment Config

| Field | Description | Example |
|-------|-------------|---------|
| `env.n_envs` | Number of parallel environments | `40` |
| `env.max_episode_steps` | Max steps per episode | `1000` |
| `env.best_reward_threshold_for_success` | Success threshold | `3.0` |
| `env.wrappers.mujoco_locomotion_lowdim.normalization_path` | Obs normalization file | `data/gym/.../normalization.npz` |

---

## Resume and Continue Training

### Pre-training

The base `PreTrainAgent` supports checkpoint resumption. Set the `logdir` to the existing run directory:

```bash
python script/run.py --config-name=pre_drifting_transformer \
    --config-path=cfg/gym/pretrain/hopper-medium-v2 \
    logdir=/path/to/existing/run/dir
```

The agent will find and load the latest checkpoint in `{logdir}/checkpoint/`.

### Fine-tuning

For fine-tuning, to resume from an existing Q-Guided checkpoint (not the pre-trained base), set `base_policy_path=null` so the system loads from the fine-tuning checkpoint directory:

```bash
python script/run.py --config-name=ft_qguided_drifting_transformer \
    --config-path=$REINFLOW_DIR/cfg/gym/finetune/hopper-v2 \
    base_policy_path=null \
    logdir=/path/to/existing/finetune/dir
```

---

## Key Metrics and Interpretation

### Pre-training Metrics

| Metric | Good Trend | Meaning |
|--------|------------|---------|
| `train/loss` | ↓ Decreasing | MSE between predictions and drifted targets |
| `train/V_norm_mean` | ↓ Decreasing | Drift field magnitude; near 0 = converged |
| `train/mean_cross_dist` | ↓ Decreasing | Distance between generated and real samples |
| `train/drift_magnitude` | ↓ Decreasing | RMS of the drift field |
| `train/pos_neg_dist_ratio` | → ~1.0 | Balance between positive and negative distances |
| `train/drifting_lambda_T{T}` | ↓ Decreasing | Per-temperature drift RMS |

### Fine-tuning Metrics

| Metric | Good Trend | Meaning |
|--------|------------|---------|
| `loss - actor` | ↓ Decreasing | Q-guided drifting actor loss |
| `loss - critic` | ↓ Decreasing | Bellman TD critic loss |
| `critic/q1_mean` | ↑ Increasing | Mean Q-value estimate (should grow with rewards) |
| `critic/target_mean` | ↑ Steady growth | Target Q-value (stable growth = healthy training) |
| `actor/query_q_mean` | ↑ Increasing | Q-value of actor's generated actions |
| `actor/positive_q_mean` | ↑ Increasing | Q-value of selected high-quality positives |
| `actor/V_norm_mean` | Small, stable | Drift field magnitude during actor update |
| `success rate - eval` | ↑ Increasing | Task success rate |
| `avg episode reward - eval` | ↑ Increasing | Average evaluation reward |

### Warning Signs

- **`critic/q1_mean` exploding**: Q-values growing unboundedly → reduce `critic_lr` or increase `target_ema_rate`.
- **`actor/V_norm_mean` stuck at 0**: Actor not being updated → check `n_critic_warmup_itr`.
- **`loss - critic` not decreasing**: Reward signal may be too sparse → try `scale_reward_factor > 1`.
- **`success rate` not improving after many iterations**: Check `offline_batch_ratio` (too high = not enough online data; too low = catastrophic forgetting).

---

## Common Failure Modes and Fixes

### 1. Missing Environment Variables

```
omegaconf.errors.InterpolationResolutionError: Could not resolve 'oc.env:REINFLOW_LOG_DIR'
```
**Fix**: `source script/set_path.sh` or manually export `REINFLOW_DIR`, `REINFLOW_DATA_DIR`, `REINFLOW_LOG_DIR`.

### 2. Missing Checkpoint / Wrong Path

```
FileNotFoundError: Could not find a sibling Hydra config for checkpoint: ...
```
**Fix**: `base_policy_path` must point to a valid `.pt` file that has a `.hydra/config.yaml` in a parent directory.

### 3. Dimension Mismatch

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```
**Fix**: Check that `obs_dim`, `action_dim`, `horizon_steps`, `act_steps`, and `cond_steps` in the config match the task and the pre-trained model.

### 4. CUDA Out of Memory

**Fix**: Reduce `train.batch_size`, `model.num_action_samples`, or `env.n_envs`.

### 5. Q-Values Diverging

If Q-values grow to ±1e6 or NaN:
- Reduce `train.critic_lr` (e.g., from 3e-4 to 1e-4).
- Increase `train.target_ema_rate` (e.g., from 0.005 to 0.01).
- Add or increase `model.reference_anchor_coeff` (e.g., 0.01–0.1) to anchor actor near pre-trained behavior.

### 6. Actor Loss Not Decreasing

- Ensure `train.n_critic_warmup_itr > 0` so the critic stabilizes first.
- Check `train.actor_lr` is not too small.
- Verify that `model.num_positive_samples < model.num_action_samples`.

### 7. D4RL / Kitchen Import Error

```
ImportError: No module named 'd4rl.gym_mujoco'
```
**Fix**: `pip install -e ".[gym]"` or `pip install -e ".[kitchen]"`.

### 8. Rendering Failures on Headless Servers

```
RuntimeError: Failed to initialize OpenGL
```
**Fix**: `export MUJOCO_GL=egl` or set `sim_device=cuda:0` in config.

---

## Recommended Validation Order

### Experiment Checklist

Use this checklist when running a new experiment:

1. **[ ] Environment setup**: Verify `REINFLOW_DIR`, `REINFLOW_DATA_DIR`, `REINFLOW_LOG_DIR` are set.
2. **[ ] Data availability**: Confirm dataset files exist at the expected paths.
3. **[ ] Config check**: Verify `obs_dim`, `action_dim`, `horizon_steps`, `act_steps` match the task.
4. **[ ] Pre-train smoke test**: Run 100 steps, check loss is decreasing and `V_norm_mean` is finite.
5. **[ ] Pre-train full run**: Run to convergence, save checkpoint.
6. **[ ] Checkpoint load test**: Verify the fine-tuning config can load the pre-trained checkpoint.
7. **[ ] Fine-tune smoke test**: Run 10 iterations, check critic loss decreasing and actor loss finite.
8. **[ ] Fine-tune full run**: Run to target iterations, monitor success rate.
9. **[ ] Eval**: Run evaluation on the fine-tuned checkpoint, report success rate and reward.
10. **[ ] Repeat for other backbones**: If using Transformer, also test UNet1D (or vice versa).

### Task Validation Order (Suggested)

Start with simpler tasks to validate the pipeline, then scale up:

1. **Hopper** (simplest: 3D actions, 11D obs)
2. **Walker2d** (6D actions, 17D obs)
3. **Kitchen-Partial** (9D actions, 60D obs, multi-task)
4. **Kitchen-Complete** (harder multi-task)
5. **Ant** (8D actions, 111D obs)
6. **Humanoid** (17D actions, 376D obs, most challenging)
7. **Lift** (simplest RoboMimic, image-based)
8. **Can** (harder manipulation)
9. **Square** (precise manipulation)
10. **Transport** (dual-arm, 14D actions)
