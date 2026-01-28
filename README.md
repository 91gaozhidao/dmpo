# DMPO: Dispersive MeanFlow Policy Optimization

<div align="center">

[![Project Page](https://img.shields.io/badge/Project_Page-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](<DMPO_PROJECT_PAGE_URL>)
[![Paper](https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](<DMPO_PAPER_URL>)
[![Code](https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github&logoColor=white)](<DMPO_RELEASE_REPO_URL>)
[![Demo](https://img.shields.io/badge/Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](<DMPO_DEMO_VIDEO_URL>)

> **One Step Is Enough: Dispersive MeanFlow Policy Optimization**
> A unified framework enabling true one-step generation for real-time robotic control via MeanFlow, dispersive regularization, and RL fine-tuning.

**Authors:** Anonymous
**Affiliation:** Anonymous

</div>

---

## Overview

<div align="center">
<img src="sample_figs/abstract_image_page.png" alt="DMPO Overview" width="800"/>

*From efficiency-performance trade-off to practical real-time control. Existing methods lie on the trade-off curve: multi-step approaches achieve strong performance but slow inference, while one-step methods are fast but unstable. DMPO breaks this trade-off by occupying the upper-right region.*
</div>

---

## Architecture at a Glance

<div align="center">
<img src="sample_figs/DMPO-Framework.png" alt="DMPO Architecture" width="800"/>

*DMPO workflow: Stage 1 (Top & Middle) – Pre-training with dispersive MeanFlow. Stage 2 (Bottom) – PPO fine-tuning formulated as a two-layer policy factorization.*
</div>

---

## Highlights

- **Single-Step Inference** – MeanFlow enables mathematically-derived one-step generation without knowledge distillation.
- **Dispersive Regularization** – Prevents representation collapse in one-step policies via information-theoretic foundations.
- **RL Fine-Tuning** – PPO-based optimization to surpass expert demonstrations with BC regularization.
- **Lightweight Architecture** – 1.78M parameters enabling >120Hz real-time control.
- **5-20× Speedup** – Significant inference acceleration over multi-step baselines.

---

## Quick Start

### 1. Clone & Environment Setup

```bash
git clone <DMPO_RELEASE_REPO_URL>
cd dmpo-release
conda create -n dmpo python=3.10 -y
conda activate dmpo
pip install -e .
```

Optional extras:
```bash
# Vision manipulation stack (Robomimic)
pip install -e .[robomimic]

# Full environment suite
pip install -e .[all]
```

### 2. External Dependencies

| Environment Suite | Requirement | Notes |
| ----------------- | ----------- | ----- |
| Robomimic | MuJoCo 2.1.0 | see `installation/install_mujoco.md` |
| OpenAI Gym | D4RL datasets | see `installation/install_d4rl.md` |
| Franka Kitchen | MuJoCo 2.1.0 | see `installation/install_kitchen.md` |

Set shared paths and logging endpoints:
```bash
source script/set_path.sh  # defines DATA_ROOT, LOG_ROOT, WANDB_ENTITY
```

---

## Datasets & Checkpoints

- **Demonstration datasets:** Downloaded automatically to `DATA_ROOT` when launching pre-training.
- **Pretrained checkpoints:** [Google Drive](<DMPO_CHECKPOINT_URL>)
  Contains DMPO pretraining weights; sync into `LOG_ROOT` for evaluation scripts.
- **Evaluation statistics:** [Google Drive](<DMPO_EVAL_STATS_URL>)
  Contains aggregated `.npz` metrics corresponding to the checkpoints above.

To use custom data, place trajectories under your data directory and update the corresponding YAML in `cfg/<ENV_GROUP>/pretrain/<TASK>.yaml`.

---

## Running DMPO

### Stage 1: Dispersive Pre-Training (Image-Based)

```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/<TASK_NAME> \
  --config-name=pre_meanflow_mlp_img_dispersive \
  denoising_steps=1 \
  dispersive.loss_type=infonce_l2 \
  dispersive.weight=0.5
```
Available `<TASK_NAME>`: `lift`, `can`, `square`, `transport`.

### Stage 1: State-Based Variants

```bash
python script/run.py \
  --config-dir=cfg/<ENV_GROUP>/pretrain/<TASK_NAME> \
  --config-name=pre_meanflow_mlp_state_dispersive
```
`<ENV_GROUP>` can be `gym`, `robomimic`, or `kitchen`.

### Stage 2: PPO Fine-Tuning

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/<TASK_NAME> \
  --config-name=ft_ppo_meanflow_mlp \
  base_policy_path=<PRETRAINED_CHECKPOINT_PATH>
```

### Evaluation & Rollouts

```bash
python script/run.py \
  --config-dir=cfg/robomimic/eval/<TASK_NAME> \
  --config-name=eval_meanflow_mlp \
  checkpoint_path=<CHECKPOINT_PATH>
```
Metrics and plots are stored in `dmpo_eval_results/`.

---

## Dispersive Loss Configuration

```yaml
model:
  use_dispersive_loss: true
  dispersive:
    weight: 0.5                    # regularization strength
    temperature: 0.3               # contrastive temperature
    loss_type: "infonce_l2"        # infonce_l2 | infonce_cosine | hinge | covariance
    target_layer: "mid"            # early | mid | late | all
```

> **Tip**: Start with `loss_type: infonce_l2`, `weight: 0.5`, `target_layer: mid` for Robomimic image tasks. Increase `weight` if training diverges or features collapse.

---

## Supported Tasks

| Domain | Tasks | Notes |
| ------ | ----- | ----- |
| Robomimic (RGB) | lift, can, square, transport | default configs under `cfg/robomimic` |
| OpenAI Gym | hopper, walker2d, ant, humanoid | state-based locomotion |
| Franka Kitchen | kitchen-partial, kitchen-complete, kitchen-mixed | state-based high-DOF control |

Real robot deployment scripts (Franka-Emika-Panda) are provided under `script/real_robot/`.

---

## Reference Metrics

### Comparison with One-Step Baselines (Robomimic)

| Method | NFE | Distill. | Lift | Can | Square | Transport |
| ------ | --- | -------- | ---- | --- | ------ | --------- |
| DP-C (Teacher) | 100 | - | 97% | 96% | 82% | 46% |
| CP | 1 | Yes | - | - | 65% | 38% |
| OneDP-S | 1 | Yes | - | - | 77% | 72% |
| MP1 | 1 | No | 95% | 80% | 35% | 38% |
| **DMPO (Ours)** | **1** | **No** | **100%** | **100%** | **83%** | **88%** |

### Model Efficiency Comparison

| Model | Vision | Params | Steps | Time (4090) | Freq | Speedup |
| ----- | ------ | ------ | ----- | ----------- | ---- | ------- |
| DP (DDPM) | ResNet-18x2 | 281M | 100 | 391.1ms | 2.6Hz | 1x |
| CP | ResNet-18x2 | 285M | 1 | 5.4ms | 187Hz | 73x |
| MP1 | PointNet | 256M | 1 | 4.1ms | 244Hz | 96x |
| **DMPO (Ours)** | **light ViT** | **1.78M** | **1** | **0.6ms** | **1770Hz** | **694x** |

### Holistic Radar Comparison

<div align="center">
<img src="sample_figs/radar_comparison_dual.png" alt="Radar Comparison" width="800"/>

*Holistic radar comparison across eight dimensions. (a) RL fine-tuning methods: DMPO forms the outer envelope, achieving top scores across all dimensions. (b) Generation methods: DMPO outperforms all baselines by combining one-step inference with lightweight architecture, high data efficiency, and the ability to go beyond demonstrations.*
</div>

---

## Repository Map

```
dmpo-release/
├── agent/                    # training & evaluation agents
│   ├── pretrain/            # pre-training scripts
│   └── finetune/            # PPO fine-tuning scripts
├── cfg/                      # experiment YAMLs (Hydra configs)
│   ├── robomimic/           # Robomimic tasks
│   ├── gym/                 # OpenAI Gym tasks
│   └── kitchen/             # Franka Kitchen tasks
├── model/                    # model architectures
│   ├── flow/                # MeanFlow implementation
│   ├── diffusion/           # diffusion baselines
│   └── common/              # shared components (ViT, MLP)
├── env/                      # environment wrappers
├── util/                     # utilities
├── script/                   # launch scripts
│   ├── run.py               # unified launcher
│   └── real_robot/          # real robot deployment
├── installation/             # environment setup guides
├── docs/                     # extended documentation
└── sample_figs/              # sample figures
```

---

## Our Contributions

1. **Framework:** We introduce DMPO, a unified framework enabling stable one-step generation via principled co-design of architecture and algorithms, with 5-20× speedup over multi-step baselines.

2. **Theory:** We establish the first information-theoretic foundation proving dispersive regularization is necessary for stable one-step generation, and derive the first mathematical formulation for RL fine-tuning of one-step policies.

3. **Validation:** We achieve state-of-the-art on RoboMimic and OpenAI Gym benchmarks, and validate real-time control (>120Hz) on a Franka robot.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{dmpo2025,
  title={One Step Is Enough: Dispersive MeanFlow Policy Optimization},
  author={Anonymous},
  year={2025}
}
```

---

## Acknowledgments

DMPO builds upon several excellent open-source projects:
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [ReinFlow](https://github.com/ReinFlow/ReinFlow)
- [DPPO](https://github.com/irom-princeton/dppo)
- [Robomimic](https://github.com/ARISE-Initiative/robomimic)
- [MeanFlow / Shortcut Models](https://github.com/kvfrans/shortcut-models)

See `THIRD_PARTY_LICENSES.md` for complete dependency attributions.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

- Submit issues: [GitHub Issues](<DMPO_ISSUES_URL>)
- Email: anonymous
