# Drifting Policy Guide

The **Drifting Policy** is a 1-NFE (one-step) generative model that uses a "drifting field" to map noise directly to expert actions. It is designed for high-efficiency inference while maintaining the multi-modal modeling capabilities of flow-based models.

## 1. Core Concept

Unlike standard Flow Matching which predicts velocities along a trajectory, Drifting Policy treats the generation as a single jump:
1.  Sample noise $z \sim \mathcal{N}(0, I)$.
2.  Compute drift $V = \text{Drift}(z, \text{condition})$.
3.  Action $a = z + V$.

## 2. Key Hyperparameters

| Parameter             | Default | Description                                                                |
| --------------------- | ------- | -------------------------------------------------------------------------- |
| `drift_coef`          | 0.1     | Scale factor for positive drift (towards expert actions).                  |
| `neg_drift_coef`      | 0.05    | Scale factor for negative drift (away from non-expert/noisy samples).      |
| `mask_self`           | `False` | Whether to ignore the current sample when computing pairwise drift fields. |
| `max_denoising_steps` | 1       | **Must be 1**. Increasing this breaks the 1-NFE design.                    |

## 3. Training Stages

### Stage 1: Pre-training
Trains the base drift field using behavior cloning on expert data.
```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/square \
  --config-name=pre_drifting_mlp_img
```

### Stage 2: Fine-tuning
Optimizes the drift field using online RL.

**PPO Fine-tuning:**
```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/square \
  --config-name=ft_ppo_drifting_mlp_img \
  base_policy_path=[PRETRAINED_CHECKPOINT]
```

**GRPO Fine-tuning (Group-based):**
GRPO is particularly effective for Drifting Policy as it eliminates the need for a critic.
```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/square \
  --config-name=ft_grpo_drifting_mlp_img \
  base_policy_path=[PRETRAINED_CHECKPOINT]
```

## 4. Tuning Recommendations

1.  **Stability**: If the actions are too jittery, decrease `drift_coef`.
2.  **Exploration**: If the model collapses to a single mode, increase `neg_drift_coef` to push samples apart.
3.  **Visual Encoder**: For image-based tasks, ensure `img_cond_steps` matches your pre-training setup.
4.  **Inference steps**: Always verify `inference_steps=1` in the fine-tuning config.
