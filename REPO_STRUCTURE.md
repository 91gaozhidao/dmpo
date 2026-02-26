# DMPO 仓库详细文件说明

本文档详细介绍了 DMPO（Dispersive MeanFlow Policy Optimization）仓库中的所有文件和目录结构。

---

## 顶层文件

| 文件 | 说明 |
| --- | --- |
| `README.md` | 项目主页，包含概览、快速开始、架构说明、性能对比、引用信息等 |
| `LICENSE` | MIT 开源许可证 |
| `pyproject.toml` | Python 项目构建配置，定义依赖项和可选依赖（gym、robomimic、kitchen 等） |
| `requirements.txt` | Python 依赖清单（PyTorch、Hydra、WandB、NumPy 等） |
| `__init__.py` | 包初始化文件，定义版本号 `1.0.0` |
| `.gitignore` | Git 忽略规则（缓存、日志、checkpoint、临时文件等） |

---

## 目录结构总览

```
dmpo/
├── agent/                    # 训练与评估代理
│   ├── dataset/              # 数据集加载器
│   ├── eval/                 # 评估代理（多种模型类型）
│   ├── finetune/             # 微调代理（多种 RL 算法）
│   └── pretrain/             # 预训练代理（多种生成模型）
├── cfg/                      # Hydra 实验配置（YAML）
│   ├── d3il/                 # D3IL 环境配置
│   ├── furniture/            # Furniture-Bench 环境配置
│   ├── gym/                  # OpenAI Gym / Franka Kitchen 配置
│   └── robomimic/            # Robomimic 环境配置
├── data_process/             # 数据预处理工具
├── docs/                     # 扩展文档
├── env/                      # 环境封装器
│   ├── gym_utils/            # Gym 环境工具与向量化封装
│   └── plot_traj.py          # 轨迹可视化脚本
├── installation/             # 环境安装指南
├── model/                    # 模型架构
│   ├── common/               # 通用网络组件（MLP、ViT、Critic 等）
│   ├── diffusion/            # 扩散模型实现
│   ├── flow/                 # Flow Matching / MeanFlow / Shortcut 模型
│   ├── gaussian/             # 高斯策略 + RL 算法
│   ├── loss/                 # 损失函数（Dispersive Loss）
│   └── rl/                   # 离线 RL 策略
├── sample_figs/              # 示例图片（论文中的图表）
├── script/                   # 启动脚本与工具
│   ├── dataset/              # 数据集下载与处理脚本
│   └── run.py                # 统一启动入口
└── util/                     # 实用工具集
```

---

## 各目录详细说明

### `agent/` — 训练与评估代理

#### `agent/dataset/` — 数据集加载

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化 |
| `sequence.py` | 序列数据集加载器，支持从 `.npz` 文件加载 states、actions、images |
| `d3il_dataset/` | D3IL 环境专用数据集（避障、推箱、排序、堆叠等任务） |
| `d3il_dataset/__init__.py` | 包初始化 |
| `d3il_dataset/aligning_dataset.py` | 对齐任务数据集 |
| `d3il_dataset/avoiding_dataset.py` | 避障任务数据集 |
| `d3il_dataset/base_dataset.py` | D3IL 数据集基类 |
| `d3il_dataset/geo_transform.py` | 几何变换工具 |
| `d3il_dataset/pushing_dataset.py` | 推箱任务数据集 |
| `d3il_dataset/sorting_dataset.py` | 排序任务数据集 |
| `d3il_dataset/stacking_dataset.py` | 堆叠任务数据集 |

#### `agent/eval/` — 评估代理

| 文件 | 说明 |
| --- | --- |
| `eval_agent_base.py` | 评估代理基类（状态输入） |
| `eval_agent_img_base.py` | 评估代理基类（图像输入） |
| `eval_diffusion_agent.py` | 扩散模型评估（状态） |
| `eval_diffusion_img_agent.py` | 扩散模型评估（图像） |
| `eval_meanflow_agent.py` | MeanFlow 模型评估（状态） |
| `eval_meanflow_img_agent.py` | MeanFlow 模型评估（图像） |
| `eval_reflow_agent.py` | ReFlow 模型评估（状态） |
| `eval_reflow_img_agent.py` | ReFlow 模型评估（图像） |
| `eval_shortcut_agent.py` | Shortcut 模型评估（状态） |
| `eval_shortcut_img_agent.py` | Shortcut 模型评估（图像） |
| `eval_consistency_img_agent.py` | Consistency 模型评估（图像） |

#### `agent/pretrain/` — 预训练代理

| 文件 | 说明 |
| --- | --- |
| `train_agent.py` | 预训练代理基类 |
| `train_diffusion_agent.py` | DDPM 扩散策略预训练 |
| `train_consistency_agent.py` | Consistency 模型预训练 |
| `train_gaussian_agent.py` | 高斯策略预训练 |
| `train_meanflow_agent.py` | MeanFlow 策略预训练 |
| `train_improved_meanflow_agent.py` | 改进版 MeanFlow 预训练（含 Dispersive Loss） |
| `train_reflow_agent.py` | 1-ReFlow 策略预训练 |
| `train_reflow_dispersive_agent.py` | 带 Dispersive Loss 的 ReFlow 预训练 |
| `train_shortcut_agent.py` | Shortcut Flow 策略预训练 |
| `train_shortcut_dispersive_agent.py` | 带 Dispersive Loss 的 Shortcut Flow 预训练 |
| `utils.py` | 预训练辅助工具 |

#### `agent/finetune/` — 微调代理

| 文件 | 说明 |
| --- | --- |
| `train_agent.py` | 微调代理基类 |

**`agent/finetune/dppo/`** — DPPO 微调

| 文件 | 说明 |
| --- | --- |
| `train_ppo_agent.py` | PPO 微调基类 |
| `train_ppo_diffusion_agent.py` | DDPM 策略 PPO 微调（状态） |
| `train_ppo_diffusion_img_agent.py` | DDPM 策略 PPO 微调（图像） |
| `train_ppo_gaussian_agent.py` | 高斯策略 PPO 微调（状态） |
| `train_ppo_gaussian_img_agent.py` | 高斯策略 PPO 微调（图像） |

**`agent/finetune/reinflow/`** — ReinFlow 微调

| 文件 | 说明 |
| --- | --- |
| `buffer.py` | 经验回放缓冲区 |
| `train_agent.py` | ReinFlow 微调基类 |
| `train_ppo_agent.py` | ReinFlow PPO 基类 |
| `train_ppo_diffusion_agent.py` | 扩散策略 ReinFlow PPO（状态） |
| `train_ppo_diffusion_img_agent.py` | 扩散策略 ReinFlow PPO（图像） |
| `train_ppo_flow_agent.py` | Flow 策略 ReinFlow PPO（状态） |
| `train_ppo_flow_img_agent.py` | Flow 策略 ReinFlow PPO（图像） |
| `train_ppo_gaussian_agent.py` | 高斯策略 ReinFlow PPO |
| `train_ppo_meanflow_agent.py` | MeanFlow 策略 ReinFlow PPO（状态） |
| `train_ppo_meanflow_img_agent.py` | MeanFlow 策略 ReinFlow PPO（图像） |
| `train_ppo_shortcut_agent.py` | Shortcut 策略 ReinFlow PPO（状态） |
| `train_ppo_shortcut_img_agent.py` | Shortcut 策略 ReinFlow PPO（图像） |

**`agent/finetune/diffusion_baselines/`** — 扩散基线微调

| 文件 | 说明 |
| --- | --- |
| `train_awr_diffusion_agent.py` | AWR 算法微调扩散策略 |
| `train_dipo_diffusion_agent.py` | DIPO 算法微调扩散策略 |
| `train_dql_diffusion_agent.py` | DQL 算法微调扩散策略 |
| `train_idql_diffusion_agent.py` | IDQL 算法微调扩散策略 |
| `train_qsm_diffusion_agent.py` | QSM 算法微调扩散策略 |
| `train_rwr_diffusion_agent.py` | RWR 算法微调扩散策略 |

**`agent/finetune/flow_baselines/`** — Flow 基线微调

| 文件 | 说明 |
| --- | --- |
| `train_fql_agent.py` | FQL 算法微调 Flow 策略 |
| `train_sac_agent.py` | SAC 算法微调 Flow 策略 |

**`agent/finetune/offlinerl_baselines/`** — 离线 RL 基线

| 文件 | 说明 |
| --- | --- |
| `train_calql_agent.py` | Cal-QL 算法训练 |
| `train_ibrl_agent.py` | IBRL 算法训练 |
| `train_rlpd_agent.py` | RLPD 算法训练 |

---

### `model/` — 模型架构

#### `model/common/` — 通用网络组件

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化 |
| `mlp.py` | 多层感知机（MLP）网络 |
| `mlp_gaussian.py` | 高斯输出 MLP |
| `mlp_gmm.py` | 高斯混合模型 MLP |
| `gaussian.py` | 高斯分布工具 |
| `gmm.py` | 高斯混合模型工具 |
| `critic.py` | Critic（价值函数）网络 |
| `modules.py` | 通用网络模块（残差块、归一化等） |
| `transformer.py` | Transformer 网络 |
| `vit.py` | 轻量级 Vision Transformer（ViT）视觉编码器 |

#### `model/diffusion/` — 扩散模型

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化 |
| `diffusion.py` | DDPM 扩散模型核心实现 |
| `mlp_diffusion.py` | MLP 版扩散网络 |
| `unet.py` | U-Net 版扩散网络 |
| `sampling.py` | 扩散采样方法（DDPM、DDIM 等） |
| `sde_lib.py` | 随机微分方程（SDE）库 |
| `exact_likelihood.py` | 精确似然计算 |
| `eta.py` | 学习噪声调度参数 |
| `modules.py` | 扩散模型专用模块 |
| `diffusion_ppo.py` | 扩散策略 + PPO |
| `diffusion_ppo_exact.py` | 扩散策略 + 精确似然 PPO |
| `diffusion_awr.py` | 扩散策略 + AWR |
| `diffusion_dipo.py` | 扩散策略 + DIPO |
| `diffusion_dql.py` | 扩散策略 + DQL |
| `diffusion_idql.py` | 扩散策略 + IDQL |
| `diffusion_qsm.py` | 扩散策略 + QSM |
| `diffusion_rwr.py` | 扩散策略 + RWR |
| `diffusion_vpg.py` | 扩散策略 + VPG |
| `diffusion_eval.py` | 扩散策略评估 |

#### `model/flow/` — Flow Matching 模型（核心）

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化 |
| `mlp_flow.py` | MLP 版 Flow Matching 网络 |
| `mlp_meanflow.py` | MLP 版 MeanFlow 网络（平均速度预测） |
| `mlp_shortcut.py` | MLP 版 Shortcut Flow 网络 |
| `mlp_consistency.py` | MLP 版 Consistency 模型网络 |
| `reflow.py` | 1-ReFlow 算法实现 |
| `reflow_dispersive.py` | 带 Dispersive Loss 的 ReFlow |
| `meanflow.py` | MeanFlow 算法核心实现 |
| `improved_meanflow.py` | 改进版 MeanFlow（含 Dispersive 正则化） |
| `shortcutflow.py` | Shortcut Flow 算法实现 |
| `shortcut_dispersive.py` | 带 Dispersive Loss 的 Shortcut Flow |
| `consistency.py` | Consistency 模型实现 |

**`model/flow/ft_ppo/`** — Flow 模型 PPO 微调

| 文件 | 说明 |
| --- | --- |
| `ppoflow.py` | ReFlow PPO 微调 |
| `ppomeanflow.py` | MeanFlow PPO 微调 |
| `pposhortcut.py` | Shortcut Flow PPO 微调 |

**`model/flow/ft_baselines/`** — Flow 模型基线微调

| 文件 | 说明 |
| --- | --- |
| `fql.py` | Flow Q-Learning (FQL) 实现 |
| `utils.py` | 基线微调辅助工具 |

#### `model/loss/` — 损失函数

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化 |
| `dispersive_loss.py` | **Dispersive Loss** 核心实现（InfoNCE-L2、InfoNCE-Cosine、Hinge、Covariance 等变体） |

#### `model/gaussian/` — 高斯策略 + RL

| 文件 | 说明 |
| --- | --- |
| `gaussian_ppo.py` | 高斯策略 PPO |
| `gaussian_vpg.py` | 高斯策略 VPG |
| `gaussian_vpg_grpo.py` | 高斯策略 VPG+GRPO |
| `gaussian_grpo.py` | 高斯策略 GRPO |
| `gaussian_awr.py` | 高斯策略 AWR |
| `gaussian_sac.py` | 高斯策略 SAC |
| `gaussian_calql.py` | 高斯策略 Cal-QL |
| `gaussian_ibrl.py` | 高斯策略 IBRL |
| `gaussian_rlpd.py` | 高斯策略 RLPD |
| `gaussian_rwr.py` | 高斯策略 RWR |
| `gmm_ppo.py` | 高斯混合模型 PPO |
| `gmm_vpg.py` | 高斯混合模型 VPG |

#### `model/rl/` — 离线 RL 策略

| 文件 | 说明 |
| --- | --- |
| `gaussian_ppo.py` | 离线 RL 高斯 PPO |
| `gaussian_vpg.py` | 离线 RL 高斯 VPG |
| `gaussian_awr.py` | 离线 RL 高斯 AWR |
| `gaussian_sac.py` | 离线 RL 高斯 SAC |
| `gaussian_calql.py` | 离线 RL Cal-QL |
| `gaussian_ibrl.py` | 离线 RL IBRL |
| `gaussian_rlpd.py` | 离线 RL RLPD |
| `gaussian_rwr.py` | 离线 RL RWR |
| `gmm_ppo.py` | 离线 RL GMM PPO |
| `gmm_vpg.py` | 离线 RL GMM VPG |

---

### `cfg/` — 实验配置文件（Hydra YAML）

共包含 **469 个 YAML 配置文件**，按环境和阶段组织：

```
cfg/
├── d3il/                         # D3IL 环境
│   ├── eval/avoid_m1/            # 评估配置
│   ├── finetune/avoid_m{1,2,3}/  # 微调配置
│   └── pretrain/avoid_m{1,2,3}/  # 预训练配置
├── furniture/                    # Furniture-Bench 环境
│   ├── eval/one_leg_low/
│   ├── finetune/{lamp,one_leg,round_table}_{low,med}/
│   └── pretrain/{lamp,one_leg,round_table}_{low,med}/
├── gym/                          # OpenAI Gym / Franka Kitchen
│   ├── eval/{hopper,walker2d,ant,Humanoid,kitchen-*}/
│   ├── finetune/{hopper,walker2d,ant,Humanoid,halfcheetah,kitchen-*}-v{0,2,3}/
│   ├── pretrain/{hopper,walker2d,ant,humanoid,kitchen-*}-medium-v{0,2,3}/
│   └── scratch/{hopper,walker2d,halfcheetah,Humanoid,kitchen-*}-v{2,3}/
└── robomimic/                    # Robomimic 环境
    ├── env_meta/                 # 环境元信息
    ├── eval/{lift,can,square,transport}/
    ├── finetune/{lift,can,square,transport}/
    ├── pretrain/{lift,can,square,transport}/
    └── scratch/{can,square}/
```

配置文件命名规则：
- `pre_*` — 预训练配置（如 `pre_meanflow_mlp_img_dispersive.yaml`）
- `ft_*` — 微调配置（如 `ft_ppo_meanflow_mlp.yaml`）
- `eval_*` — 评估配置（如 `eval_meanflow_mlp_img.yaml`）

---

### `env/` — 环境封装器

| 文件 | 说明 |
| --- | --- |
| `plot_traj.py` | 轨迹可视化脚本 |

**`env/gym_utils/`** — Gym 环境工具

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，提供 `make_async` 环境创建入口 |
| `async_vector_env.py` | 异步向量化环境（多进程并行） |
| `sync_vector_env.py` | 同步向量化环境 |
| `vector_env.py` | 向量化环境基类 |
| `furniture_normalizer.py` | Furniture-Bench 环境归一化工具 |

**`env/gym_utils/wrapper/`** — 环境封装器

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化 |
| `multi_step.py` | 多步动作执行封装器（观测历史 + 多步环境步） |
| `robomimic_image.py` | Robomimic 图像观测封装器 |
| `robomimic_lowdim.py` | Robomimic 低维观测封装器（归一化等） |
| `mujoco_locomotion_lowdim.py` | MuJoCo 运动任务封装器 |
| `d3il_lowdim.py` | D3IL 低维观测封装器 |
| `furniture.py` | Furniture-Bench 环境封装器 |

---

### `data_process/` — 数据预处理工具

| 文件 | 说明 |
| --- | --- |
| `downloader.py` | 数据集下载工具 |
| `hdf5_to_npz.py` | HDF5 → NPZ 格式转换（D4RL 数据集） |
| `normalize.py` | 在线数据归一化 |
| `normalize_offline.py` | 离线数据归一化 |
| `read_hdf5.py` | HDF5 文件内容查看工具 |
| `read_npz.py` | NPZ 文件内容查看工具 |
| `robomimic_convertor.py` | Robomimic 数据格式转换 |

---

### `script/` — 启动脚本与工具

| 文件 | 说明 |
| --- | --- |
| `run.py` | **统一启动入口**，通过 Hydra 配置运行预训练/微调/评估 |
| `set_path.sh` | 设置共享路径（`DATA_ROOT`、`LOG_ROOT`、`WANDB_ENTITY`） |
| `activate.sh` | Conda 环境激活脚本 |
| `download_url.py` | URL 下载工具 |
| `test_d3il_render.py` | D3IL 环境渲染测试 |
| `test_robomimic_render.py` | Robomimic 环境渲染测试 |

**`script/dataset/`** — 数据集处理脚本

| 文件 | 说明 |
| --- | --- |
| `README.md` | 数据集处理说明 |
| `get_d4rl_dataset.py` | D4RL 数据集下载脚本 |
| `process_robomimic_dataset.py` | Robomimic 数据集处理 |
| `process_d3il_dataset.py` | D3IL 数据集处理 |
| `filter_d3il_avoid_data.py` | D3IL 避障数据过滤 |

---

### `util/` — 实用工具集

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化 |
| `dirs.py` | 目录管理工具 |
| `logging_custom.py` | 自定义日志记录 |
| `timer.py` | 计时器工具 |
| `process.py` | 进程管理工具 |
| `reproducibility.py` | 可复现性工具（随机种子设置等） |
| `scheduler.py` | 学习率调度器 |
| `scheduler_simple.py` | 简化版学习率调度器 |
| `setup.py` | 训练环境初始化工具 |
| `drawer.py` | 绘图工具（评估结果可视化） |
| `reward_scaling.py` | 奖励缩放工具 |
| `reward_scaling_ts.py` | 时间步奖励缩放工具 |
| `hf_download.py` | Hugging Face 模型/数据下载工具（支持 `hf://` 前缀） |
| `merge_wandb.py` | WandB 日志合并工具 |
| `merge_wandb_online.py` | 在线 WandB 日志合并 |
| `pkl2wandb.py` | PKL → WandB 日志转换 |
| `compare_ckpts.py` | Checkpoint 对比工具 |
| `compare_npz.py` | NPZ 文件对比工具 |
| `insert_key_to_cfgs.py` | 批量向配置文件插入键值 |
| `license_marker.py` | 许可证标记工具 |
| `clear_pycache.py` | 清除 `__pycache__` 缓存目录 |
| `sac_humanoid_sampler.py` | SAC Humanoid 任务采样器 |
| `test_robomimic.py` | Robomimic 环境测试 |

---

### `installation/` — 环境安装指南

| 文件 | 说明 |
| --- | --- |
| `install_mujoco.md` | MuJoCo 2.1.0 安装说明（Robomimic 和 Kitchen 需要） |
| `install_d3il.md` | D3IL 环境安装说明 |
| `install_furniture.md` | Furniture-Bench 安装说明 |
| `reinflow-setup.md` | ReinFlow 环境快速配置 |
| `reinflow-setup-verbose.md` | ReinFlow 环境详细配置 |

---

### `docs/` — 扩展文档

| 文件 | 说明 |
| --- | --- |
| `Custom.md` | 自定义数据集和环境指南（如何添加新任务） |
| `ReproduceExps.md` | 实验复现完整指南（数据集获取、预训练、微调、评估等） |

---

### `sample_figs/` — 示例图片

| 文件 | 说明 |
| --- | --- |
| `abstract_image_page.png` | 论文摘要图（性能-效率权衡示意） |
| `DMPO-Framework.png` | DMPO 架构框图（两阶段工作流程） |
| `radar_comparison_dual.png` | 八维雷达图对比（与基线的全面比较） |

---

## 统计信息

| 类别 | 数量 |
| --- | --- |
| Python 源文件（`.py`） | 190 |
| YAML 配置文件 | 469 |
| Markdown 文档 | 10 |
| 图片文件 | 3 |

---

## 核心工作流程

```
1. 预训练 (Stage 1)
   script/run.py → agent/pretrain/ → model/flow/ + model/loss/dispersive_loss.py
                                        ↓
                                   保存 checkpoint (.pt)

2. PPO 微调 (Stage 2)
   script/run.py → agent/finetune/reinflow/ → model/flow/ft_ppo/
                                                 ↓
                                            保存微调后 checkpoint

3. 评估
   script/run.py → agent/eval/ → 输出指标和图表到 dmpo_eval_results/
```

---

## 支持的算法

### 生成模型（预训练）
- **DDPM** — 去噪扩散概率模型
- **1-ReFlow** — 一步重排 Flow Matching
- **Shortcut Flow** — 快捷 Flow Matching
- **MeanFlow** — 平均速度 Flow Matching（DMPO 核心）
- **Consistency Model** — 一致性模型
- **Dispersive 变体** — 上述模型 + Dispersive Loss 正则化

### RL 微调算法
- **PPO** — 近端策略优化
- **AWR** — Advantage Weighted Regression
- **DQL** — Diffusion Q-Learning
- **IDQL** — Implicit DQL
- **DIPO** — Diffusion Policy Optimization
- **QSM** — Q-Score Matching
- **RWR** — Reward Weighted Regression
- **SAC** — Soft Actor-Critic
- **Cal-QL** — Calibrated Q-Learning
- **FQL** — Flow Q-Learning
- **GRPO** — Group Relative Policy Optimization
- **IBRL** — Imitation Bootstrapped RL
- **RLPD** — RL with Prior Data

### 支持环境
- **Robomimic** — lift、can、square、transport（RGB 图像输入）
- **OpenAI Gym** — hopper、walker2d、ant、humanoid（状态输入）
- **Franka Kitchen** — partial、complete、mixed（状态输入）
- **D3IL** — avoiding、pushing、sorting、stacking 等
- **Furniture-Bench** — lamp、one_leg、round_table
