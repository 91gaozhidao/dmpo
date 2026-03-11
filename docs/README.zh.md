# DMPO: Dispersive MeanFlow Policy Optimization (分散式 MeanFlow 策略优化)

<div align="center">

[![Project Page](https://img.shields.io/badge/Project_Page-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://guowei-zou.github.io/dmpo-page/)
[![arXiv](https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2601.20701)
[![Datasets](https://img.shields.io/badge/Datasets-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Guowei-Zou/DMPO-datasets)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Guowei-Zou/DMPO-checkpoints)
[![Youtube](https://img.shields.io/badge/Youtube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=_vB_mchoux8)
[![Bilibili](https://img.shields.io/badge/Bilibili-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV133zXBPEdb/?share_source=copy_web&vd_source=af323cc810d69452bd73799b93e838d6)

> **One Step Is Enough: Dispersive MeanFlow Policy Optimization**
> 一个统一的框架，通过 MeanFlow、分散正则化（dispersive regularization）和 RL 微调，实现真正的单步生成，用于实时机器人控制。

[Guowei Zou](https://guowei-zou.github.io/Guowei-Zou/), Haitao Wang, [Hejun Wu](https://cse.sysu.edu.cn/teacher/WuHejun), Yukun Qian, [Yuhang Wang](https://hanlanqian.github.io/about/?lang=en), [Weibing Li](https://cse.sysu.edu.cn/teacher/LiWeibing)
<br>
中山大学

</div>

---

## 概述

<div align="center">
<img src="sample_figs/abstract_image_page.png" alt="DMPO Overview" width="800"/>

*从效率-性能权衡到实际的实时控制。现有方法处于权衡曲线上：多步方法性能强但推理慢，而单步方法快但极不稳定。DMPO 通过占据右上角区域突破了这一权衡。*
</div>

---

## 架构一览

<div align="center">
<img src="sample_figs/DMPO-Framework.png" alt="DMPO Architecture" width="800"/>

*DMPO 工作流：阶段 1（上部和中部）—— 使用分散式 MeanFlow 进行预训练。阶段 2（下部）—— 形式化为两层策略分解的 PPO 微调。*
</div>

---

## 亮点

- **单步推理** – MeanFlow 实现了数学推导的单步生成，无需知识蒸馏。
- **Drifting Policy** – 采用双场（正向/负向）漂移逻辑的高效 1-NFE（单步）生成。
- **GRPO 微调** – 用于 Drifting Policy 的无 Critic 组相对策略优化（Group Relative Policy Optimization），使用组内 Z-score 优势归一化。
- **分散正则化** – 基于信息论基础，防止单步策略中的表示崩溃。
- **RL 微调** – 基于 PPO 和 GRPO 的优化，通过 BC 正则化超越专家演示。
- **轻量化架构** – 1.78M 参数量，支持 >120Hz 的实时控制。
- **5-20 倍加速** – 相比多步基准模型，推理速度显著提升。

---

## 快速开始

### 1. 克隆与环境配置

```bash
git clone https://github.com/Guowei-Zou/dmpo-release.git
cd dmpo-release
conda create -n dmpo python=3.10 -y
conda activate dmpo
pip install -e .
```

可选组件：
```bash
# 视觉操作栈 (Robomimic)
pip install -e .[robomimic]

# 全环境套件
pip install -e .[all]
```

### 2. 外部依赖

| 环境套件       | 需求         | 备注                                 |
| -------------- | ------------ | ------------------------------------ |
| Robomimic      | MuJoCo 2.1.0 | 见 `installation/install_mujoco.md`  |
| OpenAI Gym     | D4RL 数据集  | 见 `installation/install_d4rl.md`    |
| Franka Kitchen | MuJoCo 2.1.0 | 见 `installation/install_kitchen.md` |

设置共享路径和日志：
```bash
source script/set_path.sh  # 定义 DATA_ROOT, LOG_ROOT, WANDB_ENTITY
```

---

## 数据集与检查点

- **演示数据集：** 在启动预训练时自动从 Google Drive 下载。也可在 [Hugging Face](https://huggingface.co/datasets/Guowei-Zou/DMPO-datasets) 上获取。
- **预训练检查点：** [Hugging Face](https://huggingface.co/Guowei-Zou/DMPO-checkpoints)

### 预训练检查点结构

```
pretrained_checkpoints/
├── DMPO_pretrained_gym_checkpoints/
│   ├── gym_improved_meanflow/           # 不带分散损失的 MeanFlow
│   │   └── {task}_best.pt               # hopper, walker2d, ant, Humanoid, kitchen-*
│   └── gym_improved_meanflow_dispersive/  # 带分散损失的 MeanFlow (推荐)
│       └── {task}_best.pt
└── DMPO_pretraining_robomimic_checkpoints/
    ├── w_0p1/                           # 分散权重 = 0.1
    ├── w_0p5/                           # 分散权重 = 0.5 (推荐)
    └── w_0p9/                           # 分散权重 = 0.9
        └── {task}/                      # lift, can, square, transport
            ├── {task}_w*_08_meanflow_dispersive.pt  # DMPO (推荐)
            ├── {task}_w*_02_meanflow_baseline.pt    # MeanFlow 基准
            ├── {task}_w*_03_reflow_baseline.pt      # Reflow 基准
            └── {task}_w*_01_shortcut_flow_baseline.pt
```

### 从 Hugging Face 下载

在配置文件中使用 `hf://` 前缀可从 Hugging Face 自动下载：

```yaml
# Gym 任务 (微调)
base_policy_path: hf://pretrained_checkpoints/DMPO_pretrained_gym_checkpoints/gym_improved_meanflow_dispersive/hopper-medium-v2_best.pt

# Robomimic 任务 (微调)
base_policy_path: hf://pretrained_checkpoints/DMPO_pretraining_robomimic_checkpoints/w_0p5/can/can_w0p5_08_meanflow_dispersive.pt
```

要使用自定义数据，请将轨迹文件放在您的数据目录下，并更新 `cfg/<ENV_GROUP>/pretrain/<TASK>.yaml` 中相应的 YAML。

---

## 运行 DMPO

### 阶段 1：分散式预训练 (基于图像)

```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/<TASK_NAME> \
  --config-name=pre_meanflow_mlp_img_dispersive \
  denoising_steps=1 \
  dispersive.loss_type=infonce_l2 \
  dispersive.weight=0.5
```
可用 `<TASK_NAME>`: `lift`, `can`, `square`, `transport`。

### 阶段 1：基于状态的变体

```bash
python script/run.py \
  --config-dir=cfg/<ENV_GROUP>/pretrain/<TASK_NAME> \
  --config-name=pre_meanflow_mlp_state_dispersive
```
`<ENV_GROUP>` 可以是 `gym`, `robomimic`, 或 `kitchen`。

### 阶段 2：PPO 微调

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/<TASK_NAME> \
  --config-name=ft_ppo_meanflow_mlp \
  base_policy_path=<PRETRAINED_CHECKPOINT_PATH>
```

### 阶段 2b：Drifting Policy 预训练与微调

Drifting Policy 通过将均值漂移（mean-shift）计算前置到训练阶段来实现 1-NFE 推理。

**预训练 (基于状态):**
```bash
python script/run.py \
  --config-dir=cfg/gym/pretrain/<TASK_NAME> \
  --config-name=pre_drifting_mlp
```

**预训练 (基于图像):**
```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/<TASK_NAME> \
  --config-name=pre_drifting_mlp_img
```

**PPO 微调:**
```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/<TASK_NAME> \
  --config-name=ft_ppo_drifting_mlp \
  base_policy_path=<PRETRAINED_DRIFTING_CHECKPOINT>
```

**GRPO 微调 (无 Critic):**
```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/<TASK_NAME> \
  --config-name=ft_grpo_drifting_mlp \
  base_policy_path=<PRETRAINED_DRIFTING_CHECKPOINT>
```

> 📖 **有关 Drifting Policy 的数学基础和完整参数参考**，请参阅 [Drifting 指南](docs/Drifting_Guide.md)。

### 评估与 Rollouts

```bash
python script/run.py \
  --config-dir=cfg/robomimic/eval/<TASK_NAME> \
  --config-name=eval_meanflow_mlp \
  checkpoint_path=<CHECKPOINT_PATH>
```
指标和图表存储在 `dmpo_eval_results/` 中。

> 📖 **有关涵盖所有环境、模型和算法的综合指南**，请参阅 [训练指南](docs/Training_Guide.md)。

---

## 分散损失配置

```yaml
model:
  use_dispersive_loss: true
  dispersive:
    weight: 0.5                    # 正则化强度
    temperature: 0.3               # 对比温度
    loss_type: "infonce_l2"        # infonce_l2 | infonce_cosine | hinge | covariance
    target_layer: "mid"            # early | mid | late | all
```

> **提示**: 对于 Robomimic 图像任务，建议从 `loss_type: infonce_l2`, `weight: 0.5`, `target_layer: mid` 开始。如果训练发散或特征崩溃，请增加 `weight`。

---

## 支持的任务

| 领域            | 任务                                             | 策略类型                      | 备注                             |
| --------------- | ------------------------------------------------ | ----------------------------- | -------------------------------- |
| Robomimic (RGB) | lift, can, square, transport                     | MeanFlow, Drifting, Diffusion | `cfg/robomimic` 下的基于图像配置 |
| OpenAI Gym      | hopper, walker2d, ant, humanoid                  | MeanFlow, Drifting            | 基于状态的运动任务               |
| Franka Kitchen  | kitchen-partial, kitchen-complete, kitchen-mixed | MeanFlow, Drifting            | 基于状态的高自由度控制           |
| D3IL            | avoid_m1, avoid_m2, avoid_m3                     | Drifting, Diffusion, Gaussian | 多峰基准测试                     |
| Furniture-Bench | lamp, one_leg, round_table (low/med)             | Drifting, Diffusion, Gaussian | 组装任务                         |

所有环境都支持带 PPO 和 GRPO 微调的 Drifting Policy。

---

## 参考指标

### 与单步基准模型的比较 (Robomimic)

| 方法            | NFE   | 蒸馏   | Lift     | Can      | Square  | Transport |
| --------------- | ----- | ------ | -------- | -------- | ------- | --------- |
| DP-C (Teacher)  | 100   | -      | 97%      | 96%      | 82%     | 46%       |
| CP              | 1     | 是     | -        | -        | 65%     | 38%       |
| OneDP-S         | 1     | 是     | -        | -        | 77%     | 72%       |
| MP1             | 1     | 否     | 95%      | 80%      | 35%     | 38%       |
| **DMPO (Ours)** | **1** | **否** | **100%** | **100%** | **83%** | **88%**   |

### 模型效率比较

| 模型            | 视觉           | 参数量    | 步数  | 时间 (4090) | 频率       | 加速比   |
| --------------- | -------------- | --------- | ----- | ----------- | ---------- | -------- |
| DP (DDPM)       | ResNet-18x2    | 281M      | 100   | 391.1ms     | 2.6Hz      | 1x       |
| CP              | ResNet-18x2    | 285M      | 1     | 5.4ms       | 187Hz      | 73x      |
| MP1             | PointNet       | 256M      | 1     | 4.1ms       | 244Hz      | 96x      |
| **DMPO (Ours)** | **轻量级 ViT** | **1.78M** | **1** | **0.6ms**   | **1770Hz** | **694x** |

### 综合雷达图对比

<div align="center">
<img src="sample_figs/radar_comparison_dual.png" alt="Radar Comparison" width="800"/>

*八个维度的综合雷达图对比。(a) RL 微调方法：DMPO 构成了外包络线，在所有维度上都获得了最高分。(b) 生成方法：DMPO 结合了单步推理和轻量级架构、高数据效率以及超越演示的能力，优于所有基准模型。*
</div>

---

## 仓库结构图

```
dmpo-release/
├── agent/                    # 训练与评估智能体
│   ├── pretrain/            # 预训练脚本 (diffusion, meanflow, drifting)
│   ├── finetune/            # 微调脚本
│   │   ├── reinflow/        # PPO 微调 (meanflow, drifting, shortcut)
│   │   ├── grpo/            # GRPO 微调 (无 critic, drifting)
│   │   ├── dppo/            # DPPO 微调 (diffusion)
│   │   └── dpro/            # DPRO 微调
│   └── eval/                # 评估智能体
├── cfg/                      # 实验 YAML (Hydra 配置)
│   ├── robomimic/           # Robomimic 任务 (基于图像)
│   ├── gym/                 # OpenAI Gym & Franka Kitchen 任务
│   ├── d3il/                # D3IL 多峰任务
│   └── furniture/           # Furniture-Bench 任务
├── model/                    # 模型架构
│   ├── flow/                # MeanFlow, ReFlow, Shortcut 实现
│   ├── drifting/            # Drifting Policy (1-NFE)
│   │   ├── drifting.py      # 核心漂移场计算
│   │   ├── ft_ppo/          # PPO 包装器 (NoisyDriftingMLP)
│   │   └── ft_grpo/         # GRPO 包装器 (NoisyDriftingPolicy)
│   ├── diffusion/           # Diffusion 基准
│   ├── common/              # 共享组件 (ViT, MLP, Critic)
│   └── loss/                # 分散损失函数
├── env/                      # 环境包装器
├── util/                     # 工具类
├── script/                   # 启动脚本
│   ├── run.py               # 统一启动器
│   └── real_robot/          # 真实机器人部署
├── installation/             # 环境安装指南
├── docs/                     # 扩展文档
│   ├── Training_Guide.md    # 综合训练 SOP
│   ├── Drifting_Guide.md    # Drifting Policy 数学与配置参考
│   ├── Custom.md            # 添加自定义数据集/环境
│   └── REPO_STRUCTURE.md    # 详细文件描述
└── sample_figs/              # 示例图表
```

---

## 我们的贡献

1. **框架：** 我们引入了 DMPO，一个统一的框架，通过架构和算法的原则性协同设计实现了稳定的单步生成，推理速度比多步基准模型快 5-20 倍。

2. **理论：** 我们建立了第一个信息论基础，证明了分散正则化对于稳定单步生成是必要的，并推导了单步策略 RL 微调的第一个数学公式。

3. **验证：** 我们在 RoboMimic 和 OpenAI Gym 基准测试中达到了 SOTA，并在 Franka 机器人上验证了实时控制 (>120Hz)。

---

## 引用

如果您觉得这项工作有用，请引用：

```bibtex
@misc{zou2026stepenoughdispersivemeanflow,
      title={One Step Is Enough: Dispersive MeanFlow Policy Optimization}, 
      author={Guowei Zou and Haitao Wang and Hejun Wu and Yukun Qian and Yuhang Wang and Weibing Li},
      year={2026},
      eprint={2601.20701},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.20701}, 
}
```

---

## 致谢

DMPO 基于以下优秀的开源项目：
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [ReinFlow](https://github.com/ReinFlow/ReinFlow)
- [DPPO](https://github.com/irom-princeton/dppo)
- [Robomimic](https://github.com/ARISE-Initiative/robomimic)
- [MeanFlow / Shortcut Models](https://github.com/kvfrans/shortcut-models)

请参阅 `THIRD_PARTY_LICENSES.md` 了解完整的依赖项归属。

---

## 许可证

基于 MIT 许可证发布。详情见 [LICENSE](LICENSE)。

---

## 联系方式

- 提交问题: [GitHub Issues](https://github.com/Guowei-Zou/dmpo-release/issues)
- 邮箱: zougw3@mail2.sysu.edu.cn (Guowei Zou)

---

## Star 历史

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=guowei-zou/dmpo-release&type=Date)](https://star-history.com/#guowei-zou/dmpo-release&Date)

</div>
