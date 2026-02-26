# DMPO Training and Fine-tuning Guide / DMPO 各阶段训练与微调详细操作指南

This document provides a complete step-by-step guide for the DMPO (Dispersive MeanFlow Policy Optimization) project, covering environment setup, data preparation, pre-training, fine-tuning, and evaluation across all supported environments and algorithms.

本文档提供 DMPO 项目从环境搭建、数据准备、预训练、微调到评估的完整操作指南，覆盖所有支持的环境和算法。

---

## 目录

- [0. 环境搭建](#0-环境搭建)
- [1. 数据准备](#1-数据准备)
- [2. 阶段一：预训练（Pre-training）](#2-阶段一预训练pre-training)
- [3. 阶段二：RL 微调（Fine-tuning）](#3-阶段二rl-微调fine-tuning)
- [4. 阶段三：评估（Evaluation）](#4-阶段三评估evaluation)
- [5. 从零训练（Scratch Training）](#5-从零训练scratch-training)
- [6. 高级用法与常见问题](#6-高级用法与常见问题)

---

## 0. 环境搭建

### 0.1 基础环境

```bash
# 克隆仓库
git clone https://github.com/Guowei-Zou/dmpo-release.git
cd dmpo-release

# 创建 Conda 环境
conda create -n dmpo python=3.10 -y
conda activate dmpo

# 安装核心依赖
pip install -e .
```

### 0.2 按需安装额外依赖

```bash
# Robomimic 视觉操作（含 robosuite）
pip install -e .[robomimic]

# OpenAI Gym 运动任务（含 D4RL）
pip install -e .[gym]

# Franka Kitchen 多任务操作
pip install -e .[kitchen]

# D3IL 操作任务
pip install -e .[d3il]

# Furniture-Bench 组装任务
pip install -e .[furniture]

# 安装所有环境
pip install -e .[all]
```

> **注意：** MuJoCo 2.1.0 需要单独安装，详见 `installation/install_mujoco.md`。D3IL 和 Furniture-Bench 的额外安装步骤分别见 `installation/install_d3il.md` 和 `installation/install_furniture.md`。

### 0.3 设置路径和日志

```bash
# 交互式设置 DATA_ROOT、LOG_ROOT、WANDB_ENTITY
source script/set_path.sh
```

该脚本将提示输入以下路径（按回车使用默认值）：
- **项目路径**：默认为当前目录
- **数据路径**：默认为 `./data`
- **日志路径**：默认为 `./log`
- **WandB 实体名**：可选，留空则需在运行时设置 `wandb=null`

### 0.4 验证安装

```bash
# 验证 MuJoCo 渲染（Robomimic）
python script/test_robomimic_render.py

# 验证 D3IL 渲染
python script/test_d3il_render.py
```

---

## 1. 数据准备

### 1.1 自动下载（推荐）

大部分数据集在首次运行预训练命令时会**自动下载**（从 Google Drive 或 Hugging Face）。无需手动准备。

### 1.2 OpenAI Gym（D4RL 数据集）

#### 方法一：自动下载（推荐）

运行预训练命令时设置 `use_d4rl_dataset=True`（默认），数据会自动下载处理：

```bash
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_reflow_mlp
```

#### 方法二：手动下载与处理

```bash
# 1. 从 Hugging Face 下载原始 HDF5 文件
wget https://huggingface.co/datasets/imone/D4RL/resolve/main/hopper_medium-v2.hdf5
wget https://huggingface.co/datasets/imone/D4RL/resolve/main/walker2d_medium-v2.hdf5
wget https://huggingface.co/datasets/imone/D4RL/resolve/main/ant_medium_expert-v2.hdf5

# 2. 查看 HDF5 文件结构
python data_process/read_hdf5.py --file_path=hopper_medium-v2.hdf5

# 3. 转换为 NPZ 格式并归一化
python data_process/hdf5_to_npz.py --data_path=hopper_medium-v2.hdf5

# 4. 检查生成的 NPZ 文件
python data_process/read_npz.py --data_path=normalization.npz

# 5. 将文件移动到数据目录
mkdir -p data/gym/hopper-medium-v2
mv train.npz normalization.npz data/gym/hopper-medium-v2/
```

### 1.3 Robomimic（像素数据集）

预训练命令自动下载 DPPO 简化版 Robomimic 像素数据集。直接运行预训练命令即可。

### 1.4 Franka Kitchen

使用 D4RL 数据集，预训练命令自动处理下载和归一化。

### 1.5 D3IL 数据集

```bash
# 处理 D3IL 数据集
python script/dataset/process_d3il_dataset.py

# 过滤避障数据（可选）
python script/dataset/filter_d3il_avoid_data.py
```

### 1.6 自定义数据集

预训练数据加载器期望 `.npz` 文件包含以下 NumPy 数组：
- `states`：形状 `(num_total_steps, obs_dim)`
- `actions`：形状 `(num_total_steps, act_dim)`
- `images`：形状 `(num_total_steps, C, H, W)`（仅图像任务，H=W 且为 8 的倍数）
- `traj_lengths`：1-D 数组，用于索引轨迹边界

详见 `docs/Custom.md`。

---

## 2. 阶段一：预训练（Pre-training）

预训练使用离线演示数据集训练生成式策略模型。所有预训练命令的基本格式为：

```bash
python script/run.py \
  --config-dir=cfg/<环境组>/pretrain/<任务名> \
  --config-name=<预训练配置名> \
  [额外参数覆盖]
```

### 2.1 Robomimic 环境（RGB 图像输入）

**支持任务：** `lift`、`can`、`square`、`transport`

#### DMPO（Dispersive MeanFlow，推荐）

```bash
# Lift 任务 — Dispersive MeanFlow（DMPO 核心方法）
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/lift \
  --config-name=pre_meanflow_dispersive_mlp_img

# Can 任务
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_meanflow_dispersive_mlp_img

# Square 任务
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/square \
  --config-name=pre_meanflow_dispersive_mlp_img

# Transport 任务
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/transport \
  --config-name=pre_meanflow_dispersive_mlp_img
```

#### 自定义 Dispersive Loss 参数

```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_meanflow_dispersive_mlp_img \
  dispersive.weight=0.5 \
  dispersive.temperature=0.3 \
  dispersive.loss_type=infonce_l2 \
  dispersive.target_layer=mid
```

**Dispersive Loss 类型选项：**
| 参数值 | 说明 |
|--------|------|
| `infonce_l2` | InfoNCE + L2 距离（推荐，默认） |
| `infonce_cosine` | InfoNCE + 余弦相似度 |
| `hinge` | Hinge Loss |
| `covariance` | 协方差正则化 |

#### 其他 Dispersive Loss 变体（Robomimic）

```bash
# Shortcut Flow + Dispersive Loss
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_shortcut_dispersive_mlp_img

# Shortcut Flow + Dispersive (Cosine)
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_shortcut_dispersive_cosine_mlp_img

# Shortcut Flow + Dispersive (Hinge)
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_shortcut_dispersive_hinge_mlp_img

# Shortcut Flow + Dispersive (Covariance)
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_shortcut_dispersive_covariance_mlp_img
```

#### MeanFlow 基线（无 Dispersive Loss）

```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_meanflow_mlp_img
```

#### ReFlow

```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_reflow_mlp_img
```

#### Shortcut Flow

```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_shortcut_mlp_img
```

#### Consistency Model

```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_consistency_mlp_img
```

#### DDPM 扩散策略

```bash
# MLP 版扩散策略（图像）
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_diffusion_mlp_img

# U-Net 版扩散策略（图像）
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_diffusion_unet_img

# MLP 版扩散策略（状态）
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_diffusion_mlp

# U-Net 版扩散策略（状态）
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_diffusion_unet
```

#### 高斯 / GMM 策略

```bash
# 高斯策略（图像）
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_gaussian_mlp_img

# 高斯策略（状态）
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_gaussian_mlp

# GMM 策略
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_gmm_mlp

# Transformer 版高斯策略
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_gaussian_transformer

# Transformer 版 GMM 策略
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_gmm_transformer
```

### 2.2 OpenAI Gym 环境（状态输入）

**支持任务：** `hopper-medium-v2`、`walker2d-medium-v2`、`ant-medium-expert-v0`、`humanoid-medium-v3`

#### DMPO（推荐）

```bash
# Improved MeanFlow + Dispersive Loss（推荐）
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_improved_meanflow_dispersive_mlp \
  device=cuda:0 sim_device=cuda:0

# Walker2d
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_improved_meanflow_dispersive_mlp \
  device=cuda:0 sim_device=cuda:0

# Ant
python script/run.py \
  --config-dir=cfg/gym/pretrain/ant-medium-expert-v0 \
  --config-name=pre_improved_meanflow_dispersive_mlp \
  device=cuda:0 sim_device=cuda:0

# Humanoid
python script/run.py \
  --config-dir=cfg/gym/pretrain/humanoid-medium-v3 \
  --config-name=pre_improved_meanflow_dispersive_mlp \
  device=cuda:0 sim_device=cuda:0
```

#### MeanFlow / Improved MeanFlow（无 Dispersive）

```bash
# Improved MeanFlow（无 Dispersive）
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_improved_meanflow_mlp

# 标准 MeanFlow
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_meanflow_mlp

# MeanFlow + Dispersive
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_meanflow_dispersive_mlp
```

#### 其他模型

```bash
# ReFlow
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_reflow_mlp

# Shortcut Flow
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_shortcut_mlp

# DDPM 扩散策略
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_diffusion_mlp
```

#### 在 MuJoCo 环境中实时测试

```bash
# 在预训练期间周期性测试策略表现
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_improved_meanflow_dispersive_mlp \
  device=cuda:0 sim_device=cuda:0 \
  test_in_mujoco=True
```

### 2.3 Franka Kitchen 环境（状态输入）

**支持任务：** `kitchen-partial-v0`、`kitchen-complete-v0`、`kitchen-mixed-v0`

```bash
# DMPO（Improved MeanFlow + Dispersive）
python script/run.py \
  --config-dir=cfg/gym/pretrain/kitchen-mixed-v0 \
  --config-name=pre_improved_meanflow_dispersive_mlp

# MeanFlow + Dispersive
python script/run.py \
  --config-dir=cfg/gym/pretrain/kitchen-mixed-v0 \
  --config-name=pre_meanflow_dispersive_mlp

# 标准 MeanFlow
python script/run.py \
  --config-dir=cfg/gym/pretrain/kitchen-mixed-v0 \
  --config-name=pre_meanflow_mlp

# Shortcut Flow
python script/run.py \
  --config-dir=cfg/gym/pretrain/kitchen-mixed-v0 \
  --config-name=pre_shortcut_mlp

# DDPM 扩散策略
python script/run.py \
  --config-dir=cfg/gym/pretrain/kitchen-mixed-v0 \
  --config-name=pre_diffusion_mlp

# 高斯策略
python script/run.py \
  --config-dir=cfg/gym/pretrain/kitchen-mixed-v0 \
  --config-name=pre_gaussian_mlp
```

### 2.4 D3IL 环境

**支持任务：** `avoid_m1`、`avoid_m2`、`avoid_m3`

```bash
# 高斯策略
python script/run.py \
  --config-dir=cfg/d3il/pretrain/avoid_m1 \
  --config-name=pre_gaussian_mlp

# DDPM 扩散策略
python script/run.py \
  --config-dir=cfg/d3il/pretrain/avoid_m1 \
  --config-name=pre_diffusion_mlp

# GMM 策略
python script/run.py \
  --config-dir=cfg/d3il/pretrain/avoid_m1 \
  --config-name=pre_gmm_mlp
```

### 2.5 Furniture-Bench 环境

**支持任务：** `lamp_low`、`lamp_med`、`one_leg_low`、`one_leg_med`、`round_table_low`、`round_table_med`

```bash
# 高斯策略
python script/run.py \
  --config-dir=cfg/furniture/pretrain/one_leg_low \
  --config-name=pre_gaussian_mlp

# DDPM MLP 扩散策略
python script/run.py \
  --config-dir=cfg/furniture/pretrain/one_leg_low \
  --config-name=pre_diffusion_mlp

# DDPM U-Net 扩散策略
python script/run.py \
  --config-dir=cfg/furniture/pretrain/one_leg_low \
  --config-name=pre_diffusion_unet
```

### 2.6 预训练通用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `device` | 计算 GPU | `device=cuda:0` |
| `sim_device` | MuJoCo 渲染 GPU | `sim_device=cuda:0`（无 EGL 时设为 `null`） |
| `seed` | 随机种子 | `seed=3407` |
| `wandb.offline_mode` | 离线 WandB 日志 | `wandb.offline_mode=True` |
| `wandb=null` | 关闭 WandB 日志 | `wandb=null` |
| `test_in_mujoco` | 预训练中周期性测试 | `test_in_mujoco=True` |
| `denoising_steps` | 最大去噪步数 | `denoising_steps=20` |
| `use_d4rl_dataset` | 使用 D4RL 数据集 | `use_d4rl_dataset=True` |

---

## 3. 阶段二：RL 微调（Fine-tuning）

微调使用在线 RL 算法优化预训练策略。预训练的 checkpoint 会在首次运行微调命令时**自动从 Hugging Face 下载**。

基本命令格式：

```bash
python script/run.py \
  --config-dir=cfg/<环境组>/finetune/<任务名> \
  --config-name=<微调配置名> \
  [额外参数覆盖]
```

### 3.1 Robomimic 环境微调

**支持任务：** `lift`、`can`、`square`、`transport`

#### DMPO（PPO + MeanFlow，推荐）

```bash
# Lift 任务 — PPO 微调 MeanFlow（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/lift \
  --config-name=ft_ppo_meanflow_mlp_img

# Can 任务
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_meanflow_mlp_img

# Square 任务
python script/run.py \
  --config-dir=cfg/robomimic/finetune/square \
  --config-name=ft_ppo_meanflow_mlp_img

# Transport 任务
python script/run.py \
  --config-dir=cfg/robomimic/finetune/transport \
  --config-name=ft_ppo_meanflow_mlp_img
```

#### 使用自定义预训练 checkpoint

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_meanflow_mlp_img \
  base_policy_path=/path/to/your/pretrained_checkpoint.pt
```

#### 使用 Hugging Face 上的预训练 checkpoint

```bash
# 使用 hf:// 前缀自动从 Hugging Face 下载
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_meanflow_mlp_img \
  base_policy_path=hf://pretrained_checkpoints/DMPO_pretraining_robomimic_checkpoints/w_0p5/can/can_w0p5_08_meanflow_dispersive.pt
```

#### PPO + ReFlow（图像输入）

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_reflow_mlp_img
```

#### PPO + Shortcut Flow（图像输入）

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_shortcut_mlp_img
```

#### PPO + DDPM 扩散策略

```bash
# MLP（状态输入）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_diffusion_mlp

# MLP（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_diffusion_mlp_img

# U-Net（状态输入）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_diffusion_unet

# U-Net（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_diffusion_unet_img
```

#### PPO + 高斯 / GMM 策略

```bash
# 高斯（状态输入）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_gaussian_mlp

# 高斯（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_gaussian_mlp_img

# GMM
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_gmm_mlp

# Transformer 版高斯策略
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_gaussian_transformer

# Transformer 版 GMM 策略
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_gmm_transformer
```

#### 其他 RL 微调算法（Robomimic）

```bash
# AWR（Advantage Weighted Regression）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_awr_diffusion_mlp

# DQL（Diffusion Q-Learning）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_dql_diffusion_mlp

# IDQL（Implicit DQL）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_idql_diffusion_mlp

# QSM（Q-Score Matching）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_qsm_diffusion_mlp

# RWR（Reward Weighted Regression）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_rwr_diffusion_mlp

# DIPO（Diffusion Policy Optimization）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_dipo_diffusion_mlp

# PPO + 精确似然扩散策略
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_exact_diffusion_mlp
```

#### 离线 RL 基线（Robomimic）

```bash
# Cal-QL（离线预训练 + 在线微调）
# 步骤 1：离线预训练
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=calql_mlp_offline

# 步骤 2：在线微调
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=calql_mlp_online

# IBRL（Imitation Bootstrapped RL）
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ibrl_mlp
```

### 3.2 OpenAI Gym 环境微调

**支持任务：** `hopper-v2`、`walker2d-v2`、`ant-v2`、`Humanoid-v3`、`halfcheetah-v2`

#### DMPO（PPO + MeanFlow）

```bash
# Hopper
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_meanflow_mlp

# Walker2d
python script/run.py \
  --config-dir=cfg/gym/finetune/walker2d-v2 \
  --config-name=ft_ppo_meanflow_mlp

# Ant
python script/run.py \
  --config-dir=cfg/gym/finetune/ant-v2 \
  --config-name=ft_ppo_meanflow_mlp

# Humanoid
python script/run.py \
  --config-dir=cfg/gym/finetune/Humanoid-v3 \
  --config-name=ft_ppo_meanflow_mlp
```

#### 使用 Hugging Face 上的预训练 checkpoint

```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_meanflow_mlp \
  base_policy_path=hf://pretrained_checkpoints/DMPO_pretrained_gym_checkpoints/gym_improved_meanflow_dispersive/hopper-medium-v2_best.pt
```

#### 其他模型

```bash
# PPO + ReFlow
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_reflow_mlp

# PPO + Shortcut Flow
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_shortcut_mlp

# PPO + DDPM 扩散策略
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_diffusion_mlp

# PPO + DDIM 扩散策略
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_ddim_mlp

# FQL（Flow Q-Learning）
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_fql_mlp
```

#### ReinFlow 特定参数

```bash
# ReFlow + 自定义噪声标准差和熵系数
python script/run.py \
  --config-dir=cfg/gym/finetune/ant-v2 \
  --config-name=ft_ppo_reflow_mlp \
  min_std=0.08 max_std=0.16 train.ent_coef=0.03

# 指定去噪步数
python script/run.py \
  --config-dir=cfg/gym/finetune/ant-v2 \
  --config-name=ft_ppo_reflow_mlp \
  denoising_steps=1 ft_denoising_steps=1
```

### 3.3 Franka Kitchen 环境微调

**支持任务：** `kitchen-partial-v0`、`kitchen-complete-v0`、`kitchen-mixed-v0`

```bash
# PPO + MeanFlow
python script/run.py \
  --config-dir=cfg/gym/finetune/kitchen-mixed-v0 \
  --config-name=ft_ppo_meanflow_mlp

# PPO + Shortcut Flow
python script/run.py \
  --config-dir=cfg/gym/finetune/kitchen-mixed-v0 \
  --config-name=ft_ppo_shortcut_mlp

# PPO + DDPM 扩散策略
python script/run.py \
  --config-dir=cfg/gym/finetune/kitchen-mixed-v0 \
  --config-name=ft_ppo_diffusion_mlp

# FQL
python script/run.py \
  --config-dir=cfg/gym/finetune/kitchen-mixed-v0 \
  --config-name=ft_fql_mlp

# Cal-QL（在线微调）
python script/run.py \
  --config-dir=cfg/gym/finetune/kitchen-mixed-v0 \
  --config-name=calql_mlp_online

# IBRL
python script/run.py \
  --config-dir=cfg/gym/finetune/kitchen-mixed-v0 \
  --config-name=ibrl_mlp
```

### 3.4 D3IL 环境微调

**支持任务：** `avoid_m1`、`avoid_m2`、`avoid_m3`

```bash
# PPO + 高斯策略
python script/run.py \
  --config-dir=cfg/d3il/finetune/avoid_m1 \
  --config-name=ft_ppo_gaussian_mlp

# PPO + DDPM 扩散策略
python script/run.py \
  --config-dir=cfg/d3il/finetune/avoid_m1 \
  --config-name=ft_ppo_diffusion_mlp

# PPO + GMM 策略
python script/run.py \
  --config-dir=cfg/d3il/finetune/avoid_m1 \
  --config-name=ft_ppo_gmm_mlp
```

### 3.5 Furniture-Bench 环境微调

**支持任务：** `lamp_low`、`lamp_med`、`one_leg_low`、`one_leg_med`、`round_table_low`、`round_table_med`

```bash
# PPO + 高斯策略
python script/run.py \
  --config-dir=cfg/furniture/finetune/one_leg_low \
  --config-name=ft_ppo_gaussian_mlp

# PPO + DDPM MLP 扩散策略
python script/run.py \
  --config-dir=cfg/furniture/finetune/one_leg_low \
  --config-name=ft_ppo_diffusion_mlp

# PPO + DDPM U-Net 扩散策略
python script/run.py \
  --config-dir=cfg/furniture/finetune/one_leg_low \
  --config-name=ft_ppo_diffusion_unet
```

> **注意：** Furniture-Bench 使用 IsaacGym，需在同一 GPU 上运行。

### 3.6 微调通用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `base_policy_path` | 预训练 checkpoint 路径 | `base_policy_path=/path/to/checkpoint.pt` |
| `normalization_path` | 归一化统计文件路径 | `normalization_path=/path/to/normalization.npz` |
| `seed` | 随机种子 | `seed=3407` |
| `device` | 计算 GPU | `device=cuda:0` |
| `sim_device` | 渲染 GPU | `sim_device=cuda:1` |
| `name` | WandB 运行名称 | `name=my_experiment_seed42` |
| `resume_path` | 恢复训练的 checkpoint | `resume_path=/path/to/failed_checkpoint.pt` |
| `denoising_steps` | 去噪步数 | `denoising_steps=1` |
| `ft_denoising_steps` | 微调去噪步数 | `ft_denoising_steps=1` |
| `min_std` / `max_std` | 噪声标准差范围 | `min_std=0.08 max_std=0.16` |
| `train.ent_coef` | 熵系数 | `train.ent_coef=0.03` |

---

## 4. 阶段三：评估（Evaluation）

评估预训练或微调后的策略，生成性能指标和可视化图表。

基本命令格式：

```bash
python script/run.py \
  --config-dir=cfg/<环境组>/eval/<任务名> \
  --config-name=<评估配置名> \
  base_policy_path=<checkpoint路径> \
  [额外参数覆盖]
```

### 4.1 Robomimic 评估

```bash
# 评估 MeanFlow 策略（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_meanflow_mlp_img \
  base_policy_path=/path/to/checkpoint.pt

# 评估 ReFlow 策略（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_reflow_mlp_img \
  base_policy_path=/path/to/checkpoint.pt

# 评估 Shortcut Flow 策略（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_shortcut_mlp_img \
  base_policy_path=/path/to/checkpoint.pt

# 评估 Dispersive Shortcut Flow（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_shortcut_dispersive_mlp_img \
  base_policy_path=/path/to/checkpoint.pt

# 评估 DDPM 扩散策略（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_diffusion_mlp_img \
  base_policy_path=/path/to/checkpoint.pt

# 评估高斯策略（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_gaussian_mlp_img \
  base_policy_path=/path/to/checkpoint.pt

# 评估 Consistency 模型（图像输入）
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_consistency_mlp_img \
  base_policy_path=/path/to/checkpoint.pt
```

#### 评估微调后的策略（Stage 2 checkpoint）

```bash
# 微调后 MeanFlow 评估
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_meanflow_mlp_img_stage2 \
  base_policy_path=/path/to/finetuned_checkpoint.pt

# 微调后 ReFlow 评估
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_reflow_mlp_img_stage2 \
  base_policy_path=/path/to/finetuned_checkpoint.pt

# 微调后 Shortcut Flow 评估
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_shortcut_mlp_img_stage2 \
  base_policy_path=/path/to/finetuned_checkpoint.pt
```

### 4.2 OpenAI Gym 评估

```bash
# 评估 MeanFlow 策略
python script/run.py \
  --config-dir=cfg/gym/eval/hopper-medium-v2 \
  --config-name=eval_meanflow_mlp \
  base_policy_path=/path/to/checkpoint.pt

# 评估 ReFlow 策略
python script/run.py \
  --config-dir=cfg/gym/eval/walker2d-medium-v2 \
  --config-name=eval_reflow_mlp \
  base_policy_path=/path/to/checkpoint.pt

# 评估 Shortcut Flow 策略
python script/run.py \
  --config-dir=cfg/gym/eval/ant-medium-expert-v0 \
  --config-name=eval_shortcut_mlp \
  base_policy_path=/path/to/checkpoint.pt

# 评估 DDPM 扩散策略
python script/run.py \
  --config-dir=cfg/gym/eval/Humanoid-expert-v3 \
  --config-name=eval_diffusion_mlp \
  base_policy_path=/path/to/checkpoint.pt
```

### 4.3 多步去噪评估

在不同去噪步数下评估策略表现：

```bash
python script/run.py \
  --config-dir=cfg/robomimic/eval/transport \
  --config-name=eval_reflow_mlp_img \
  base_policy_path=/path/to/checkpoint.pt \
  denoising_step_list=[1,2,4,5,8,16,32,64,128] \
  load_ema=False
```

### 4.4 评估参数说明

| 参数 | 说明 | 建议 |
|------|------|------|
| `base_policy_path` | 要评估的 checkpoint 路径 | 必填 |
| `load_ema` | 是否加载 EMA 权重 | 预训练策略用 `True`，微调策略用 `False` |
| `denoising_step_list` | 测试的去噪步数列表 | `[1,2,4,5,8,16,32,64,128]` |
| `clip_intermediate_actions` | 是否裁剪中间动作 | ReinFlow 微调后建议 `True` |

### 4.5 评估输出

评估结果保存在 `dmpo_eval_results/` 目录下，包含：
- `.png` 图表：episode reward、success rate、episode length、inference frequency、duration、best reward
- 数据文件：可用于后续分析

---

## 5. 从零训练（Scratch Training）

无需预训练，直接在环境中从头训练 RL 策略。

### 5.1 OpenAI Gym 从零训练

```bash
# PPO + 高斯策略
python script/run.py \
  --config-dir=cfg/gym/scratch/hopper-v2 \
  --config-name=ppo_gaussian_mlp

# PPO + DDPM 扩散策略
python script/run.py \
  --config-dir=cfg/gym/scratch/hopper-v2 \
  --config-name=ppo_diffusion_mlp

# DQL
python script/run.py \
  --config-dir=cfg/gym/scratch/hopper-v2 \
  --config-name=dql_diffusion_mlp

# IDQL
python script/run.py \
  --config-dir=cfg/gym/scratch/hopper-v2 \
  --config-name=idql_diffusion_mlp

# AWR
python script/run.py \
  --config-dir=cfg/gym/scratch/hopper-v2 \
  --config-name=awr_diffusion_mlp

# RWR
python script/run.py \
  --config-dir=cfg/gym/scratch/hopper-v2 \
  --config-name=rwr_diffusion_mlp

# DIPO
python script/run.py \
  --config-dir=cfg/gym/scratch/hopper-v2 \
  --config-name=dipo_diffusion_mlp

# QSM
python script/run.py \
  --config-dir=cfg/gym/scratch/hopper-v2 \
  --config-name=qsm_diffusion_mlp

# GRPO + 高斯策略
python script/run.py \
  --config-dir=cfg/gym/scratch/hopper-v2 \
  --config-name=grpo_gaussian_mlp
```

### 5.2 Robomimic 从零训练

```bash
# RLPD
python script/run.py \
  --config-dir=cfg/robomimic/scratch/can \
  --config-name=rlpd_mlp
```

---

## 6. 高级用法与常见问题

### 6.1 恢复中断的训练

```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/walker2d-v2 \
  --config-name=ft_ppo_meanflow_mlp \
  resume_path=/path/to/crashed_checkpoint.pt
```

> 如果配置文件中没有 `resume_path` 字段，在对应的 YAML 配置文件末尾添加 `resume_path: null`。

### 6.2 更换预训练策略

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/transport \
  --config-name=ft_ppo_meanflow_mlp_img \
  base_policy_path=/path/to/new_pretrained_policy.pt
```

### 6.3 使用不同数据集训练的策略进行微调

需要同时更新归一化路径，否则训练曲线会异常：

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/transport \
  --config-name=ft_ppo_meanflow_mlp_img \
  base_policy_path=/path/to/new_policy.pt \
  normalization_path=/path/to/new_data_dir/normalization.npz
```

### 6.4 后台运行与日志输出

```bash
# 后台运行并保存日志
nohup python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_meanflow_mlp_img \
  > ./finetune_can_meanflow.log 2>&1 &
```

### 6.5 参数扫描（Parameter Sweep）

```bash
# 为每个实验指定名称
nohup python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_meanflow_mlp \
  seed=42 name=hopper_meanflow_seed42 \
  > ./hopper_meanflow_seed42.log 2>&1 &

nohup python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_meanflow_mlp \
  seed=3407 name=hopper_meanflow_seed3407 \
  > ./hopper_meanflow_seed3407.log 2>&1 &
```

### 6.6 多 GPU 训练

```bash
# 指定不同的计算和渲染 GPU
python script/run.py \
  --config-dir=cfg/robomimic/finetune/transport \
  --config-name=ft_ppo_meanflow_mlp_img \
  device=cuda:0 sim_device=cuda:1
```

### 6.7 GPU 内存不足

减少并行环境数，同时增加每个环境的步数以保持总步数一致：

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_meanflow_mlp_img \
  env.n_envs=10 train.n_steps=50
```

> **公式：** `total_steps = n_envs × n_steps × act_steps`，调整时保持 `total_steps` 不变。
> `train.n_steps` 应为 `env.max_episode_steps / act_steps` 的倍数。

### 6.8 WandB 日志管理

```bash
# 离线模式（无网络环境）
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_improved_meanflow_dispersive_mlp \
  wandb.offline_mode=True

# 完全关闭 WandB
python script/run.py \
  --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
  --config-name=pre_improved_meanflow_dispersive_mlp \
  wandb=null

# 批量同步离线 WandB 数据
for dir in ./wandb_offline/wandb/offline-run-20250510*; do
  wandb sync "${dir}"
done

# 从 PKL 恢复 WandB 日志
python util/pkl2wandb.py
```

### 6.9 采样分布敏感性分析

```bash
# 使用 Beta 分布
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/square \
  --config-name=pre_reflow_mlp_img \
  model.sample_t_type=beta

# 使用 Logit-Normal 分布
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/square \
  --config-name=pre_reflow_mlp_img \
  model.sample_t_type=logitnormal
```

### 6.10 环境渲染与视频录制

```bash
# Robomimic 视频录制
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_meanflow_mlp_img \
  env.save_video=True train.render.freq=10 train.render.num=3

# Furniture-Bench 可视化
python script/run.py \
  --config-dir=cfg/furniture/finetune/one_leg_low \
  --config-name=ft_ppo_gaussian_mlp \
  env.specific.headless=False env.n_envs=1

# D3IL 渲染
python script/run.py \
  --config-dir=cfg/d3il/finetune/avoid_m1 \
  --config-name=ft_ppo_gaussian_mlp \
  +env.render=True env.n_envs=1 train.render.num=1
```

### 6.11 观测历史

```bash
# 使用多步状态观测历史
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_meanflow_dispersive_mlp_img \
  cond_steps=3 img_cond_steps=2
```

> 微调时需使用与预训练一致的 `cond_steps` 和 `img_cond_steps`。

---

## 附录：推荐训练流程总结

### DMPO 完整流程（Robomimic 示例）

```bash
# 步骤 1：设置环境
source script/set_path.sh

# 步骤 2：预训练 Dispersive MeanFlow
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_meanflow_dispersive_mlp_img \
  device=cuda:0 sim_device=cuda:0

# 步骤 3：PPO 微调
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_meanflow_mlp_img \
  base_policy_path=/path/to/pretrained_checkpoint.pt

# 步骤 4：评估
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_meanflow_mlp_img_stage2 \
  base_policy_path=/path/to/finetuned_checkpoint.pt \
  load_ema=False
```

### DMPO 完整流程（Gym 示例）

```bash
# 步骤 1：设置环境
source script/set_path.sh

# 步骤 2：预训练 Improved MeanFlow + Dispersive
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_improved_meanflow_dispersive_mlp \
  device=cuda:0 sim_device=cuda:0 test_in_mujoco=True

# 步骤 3：PPO 微调
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_meanflow_mlp

# 步骤 4：评估
python script/run.py \
  --config-dir=cfg/gym/eval/hopper-medium-v2 \
  --config-name=eval_meanflow_mlp \
  base_policy_path=/path/to/finetuned_checkpoint.pt \
  load_ema=False
```

### 快速使用预训练 checkpoint 微调（跳过预训练）

```bash
# 直接使用 Hugging Face 上的预训练 checkpoint 进行微调
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_meanflow_mlp_img \
  base_policy_path=hf://pretrained_checkpoints/DMPO_pretraining_robomimic_checkpoints/w_0p5/can/can_w0p5_08_meanflow_dispersive.pt
```

---

## 附录：支持的配置名速查表

### 预训练配置名（`--config-name`）

| 配置名 | 模型类型 | 输入类型 | 说明 |
|--------|---------|---------|------|
| `pre_meanflow_dispersive_mlp_img` | MeanFlow | 图像 | **DMPO 核心**（Robomimic） |
| `pre_improved_meanflow_dispersive_mlp` | Improved MeanFlow | 状态 | **DMPO 核心**（Gym） |
| `pre_meanflow_mlp_img` | MeanFlow | 图像 | MeanFlow 基线 |
| `pre_meanflow_mlp` | MeanFlow | 状态 | MeanFlow 基线 |
| `pre_meanflow_dispersive_mlp` | MeanFlow | 状态 | MeanFlow + Dispersive |
| `pre_improved_meanflow_mlp` | Improved MeanFlow | 状态 | Improved MeanFlow 基线 |
| `pre_reflow_mlp_img` | ReFlow | 图像 | 1-ReFlow |
| `pre_reflow_mlp` | ReFlow | 状态 | 1-ReFlow |
| `pre_shortcut_mlp_img` | Shortcut Flow | 图像 | Shortcut Flow |
| `pre_shortcut_mlp` | Shortcut Flow | 状态 | Shortcut Flow |
| `pre_shortcut_dispersive_mlp_img` | Shortcut Flow | 图像 | + Dispersive (L2) |
| `pre_shortcut_dispersive_cosine_mlp_img` | Shortcut Flow | 图像 | + Dispersive (Cosine) |
| `pre_shortcut_dispersive_hinge_mlp_img` | Shortcut Flow | 图像 | + Dispersive (Hinge) |
| `pre_shortcut_dispersive_covariance_mlp_img` | Shortcut Flow | 图像 | + Dispersive (Covariance) |
| `pre_consistency_mlp_img` | Consistency | 图像 | Consistency Model |
| `pre_diffusion_mlp_img` | DDPM | 图像 | 扩散策略 (MLP) |
| `pre_diffusion_mlp` | DDPM | 状态 | 扩散策略 (MLP) |
| `pre_diffusion_unet_img` | DDPM | 图像 | 扩散策略 (U-Net) |
| `pre_diffusion_unet` | DDPM | 状态 | 扩散策略 (U-Net) |
| `pre_gaussian_mlp_img` | Gaussian | 图像 | 高斯策略 |
| `pre_gaussian_mlp` | Gaussian | 状态 | 高斯策略 |
| `pre_gmm_mlp` | GMM | 状态 | 高斯混合模型 |
| `pre_gaussian_transformer` | Gaussian | 状态 | Transformer 版 |
| `pre_gmm_transformer` | GMM | 状态 | Transformer 版 |

### 微调配置名（`--config-name`）

| 配置名 | 模型类型 | 输入类型 | 算法 |
|--------|---------|---------|------|
| `ft_ppo_meanflow_mlp_img` | MeanFlow | 图像 | **PPO (DMPO)** |
| `ft_ppo_meanflow_mlp` | MeanFlow | 状态 | PPO |
| `ft_ppo_reflow_mlp_img` | ReFlow | 图像 | PPO |
| `ft_ppo_reflow_mlp` | ReFlow | 状态 | PPO |
| `ft_ppo_shortcut_mlp_img` | Shortcut | 图像 | PPO |
| `ft_ppo_shortcut_mlp` | Shortcut | 状态 | PPO |
| `ft_ppo_diffusion_mlp_img` | DDPM | 图像 | PPO |
| `ft_ppo_diffusion_mlp` | DDPM | 状态 | PPO |
| `ft_ppo_diffusion_unet_img` | DDPM U-Net | 图像 | PPO |
| `ft_ppo_diffusion_unet` | DDPM U-Net | 状态 | PPO |
| `ft_ppo_ddim_mlp` | DDIM | 状态 | PPO |
| `ft_ppo_gaussian_mlp_img` | Gaussian | 图像 | PPO |
| `ft_ppo_gaussian_mlp` | Gaussian | 状态 | PPO |
| `ft_ppo_gmm_mlp` | GMM | 状态 | PPO |
| `ft_ppo_gaussian_transformer` | Gaussian | 状态 | PPO (Transformer) |
| `ft_ppo_gmm_transformer` | GMM | 状态 | PPO (Transformer) |
| `ft_ppo_exact_diffusion_mlp` | DDPM | 状态 | PPO (精确似然) |
| `ft_awr_diffusion_mlp` | DDPM | 状态 | AWR |
| `ft_dql_diffusion_mlp` | DDPM | 状态 | DQL |
| `ft_idql_diffusion_mlp` | DDPM | 状态 | IDQL |
| `ft_qsm_diffusion_mlp` | DDPM | 状态 | QSM |
| `ft_rwr_diffusion_mlp` | DDPM | 状态 | RWR |
| `ft_dipo_diffusion_mlp` | DDPM | 状态 | DIPO |
| `ft_fql_mlp` | Flow | 状态 | FQL |
| `calql_mlp_online` | — | 状态 | Cal-QL (在线) |
| `ibrl_mlp` | — | 状态 | IBRL |

### 评估配置名（`--config-name`）

| 配置名 | 模型类型 | 输入类型 | 说明 |
|--------|---------|---------|------|
| `eval_meanflow_mlp_img` | MeanFlow | 图像 | 预训练评估 |
| `eval_meanflow_mlp_img_stage2` | MeanFlow | 图像 | 微调后评估 |
| `eval_meanflow_mlp` | MeanFlow | 状态 | 预训练评估 |
| `eval_reflow_mlp_img` | ReFlow | 图像 | 预训练评估 |
| `eval_reflow_mlp_img_stage2` | ReFlow | 图像 | 微调后评估 |
| `eval_reflow_mlp` | ReFlow | 状态 | 预训练评估 |
| `eval_shortcut_mlp_img` | Shortcut | 图像 | 预训练评估 |
| `eval_shortcut_mlp_img_stage2` | Shortcut | 图像 | 微调后评估 |
| `eval_shortcut_mlp` | Shortcut | 状态 | 预训练评估 |
| `eval_shortcut_dispersive_mlp_img` | Shortcut + Disp. | 图像 | 预训练评估 |
| `eval_consistency_mlp_img` | Consistency | 图像 | 预训练评估 |
| `eval_diffusion_mlp_img` | DDPM | 图像 | 预训练评估 |
| `eval_diffusion_mlp` | DDPM | 状态 | 预训练评估 |
| `eval_diffusion_unet_img` | DDPM U-Net | 图像 | 预训练评估 |
| `eval_diffusion_unet` | DDPM U-Net | 状态 | 预训练评估 |
| `eval_gaussian_mlp_img` | Gaussian | 图像 | 预训练评估 |
| `eval_gaussian_mlp` | Gaussian | 状态 | 预训练评估 |
