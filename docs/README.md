# DMPO — 代码库全景与深度剖析 (Project Architecture & Deep Dive)

> **Dispersive MeanFlow Policy Optimization (弥散均值流策略优化)**：一个统一的框架，用于单步（1-NFE）生成式策略学习及在线强化学习微调。

---

## 目录

1. [高层级概览](#1-高层级概览)
2. [代码库目录结构图](#2-代码库目录结构图)
3. [核心数据流](#3-核心数据流)
4. [模型层深度剖析](#4-模型层深度剖析)
5. [智能体层深度剖析](#5-智能体层深度剖析)
6. [配置系统](#6-配置系统)
7. [Drifting Policy 与 Mean Flow Policy — 数学与代码的对比](#7-drifting-policy-vs-mean-flow-policy)
8. [微调架构：PPO 与 GRPO](#8-微调架构)
9. [环境集成](#9-环境集成)

---

## 1. 高层级概览

DMPO 为连续控制任务实现了一个两阶段训练的流水线：

```
阶段 1: 离线预训练 (Offline Pre-training)    阶段 2: 在线强化学习微调 (Online RL Fine-tuning)
┌──────────────────────────┐          ┌──────────────────────────────┐
│  专家数据集 (Expert Dataset)│          │  环境交互 (Env Interaction)    │
│         ↓                │          │         ↓                    │
│ 流匹配 / 漂移场优化        │   ──►    │  PPO 或 GRPO 目标函数          │
│ (Flow Matching/Drift Loss│          │  + 探索噪声 (Exploration Noise)│
│         ↓                │          │         ↓                    │
│  预训练策略 θ               │          │  微调后策略 θ*                 │
└──────────────────────────┘          └──────────────────────────────┘
```

**支持的策略类型：**
| 策略 (Policy) | 网络函数评估次数 (NFE) | 阶段 1 | 阶段 2 (PPO) | 阶段 2 (GRPO) |
|--------|-----|---------|---------------|----------------|
| **Drifting** | 1 | ✅ | ✅ | ✅ |
| MeanFlow | 5 | ✅ | ✅ | — |
| ShortCut | 1–5 | ✅ | ✅ | — |
| ReFlow | 5+ | ✅ | ✅ | — |
| Diffusion | 10–100 | ✅ | ✅ | — |
| Consistency | 1 | ✅ | — | — |

---

## 2. 代码库目录结构图

```
dmpo/
├── agent/                          # 训练与评估智能体
│   ├── pretrain/                   # 离线预训练智能体
│   │   ├── train_agent.py              # 基础 PreTrainAgent 类
│   │   ├── train_drifting_agent.py     # Drifting Policy 预训练
│   │   ├── train_drifting_dispersive_agent.py  # Drifting + 弥散损失 (dispersive loss)
│   │   ├── train_meanflow_agent.py     # MeanFlow 预训练
│   │   ├── train_improved_meanflow_agent.py
│   │   ├── train_shortcut_agent.py
│   │   ├── train_shortcut_dispersive_agent.py
│   │   ├── train_reflow_agent.py
│   │   ├── train_reflow_dispersive_agent.py
│   │   ├── train_consistency_agent.py
│   │   ├── train_diffusion_agent.py
│   │   └── train_gaussian_agent.py
│   │
│   ├── eval/                       # 评估智能体
│   │   ├── eval_agent_base.py          # 基础评估类
│   │   ├── eval_agent_img_base.py      # 基于图像的评估基类
│   │   ├── eval_drifting_agent.py      # Drifting Policy 评估
│   │   ├── eval_drifting_img_agent.py  # 基于图像观测的 Drifting 评估
│   │   ├── eval_meanflow_agent.py
│   │   ├── eval_meanflow_img_agent.py
│   │   ├── eval_shortcut_agent.py
│   │   ├── eval_shortcut_img_agent.py
│   │   ├── eval_reflow_agent.py
│   │   ├── eval_reflow_img_agent.py
│   │   ├── eval_diffusion_agent.py
│   │   ├── eval_diffusion_img_agent.py
│   │   └── eval_consistency_img_agent.py
│   │
│   ├── finetune/                   # 在线强化学习微调智能体
│   │   ├── reinflow/                   # 基于 PPO 的微调
│   │   │   ├── train_agent.py              # 基础 PPO 智能体
│   │   │   ├── train_ppo_shortcut_agent.py     # ShortCut PPO (状态)
│   │   │   ├── train_ppo_shortcut_img_agent.py # ShortCut PPO (图像)
│   │   │   ├── train_ppo_drifting_agent.py     # Drifting PPO (状态)
│   │   │   ├── train_ppo_drifting_img_agent.py # Drifting PPO (图像)
│   │   │   ├── train_ppo_meanflow_agent.py     # MeanFlow PPO (状态)
│   │   │   ├── train_ppo_meanflow_img_agent.py # MeanFlow PPO (图像)
│   │   │   ├── train_ppo_flow_agent.py         # 通用流 PPO
│   │   │   ├── train_ppo_flow_img_agent.py
│   │   │   ├── train_ppo_diffusion_agent.py
│   │   │   ├── train_ppo_diffusion_img_agent.py
│   │   │   ├── train_ppo_gaussian_agent.py
│   │   │   └── buffer.py                  # PPO 经验回放缓冲区
│   │   │
│   │   ├── grpo/                       # GRPO 微调 (无 Critic)
│   │   │   ├── train_grpo_drifting_agent.py    # 适用于 Drifting 的 GRPO
│   │   │   └── buffer.py                  # GRPO 分组缓冲区
│   │   │
│   │   ├── dppo/                       # Diffusion PPO 变体
│   │   ├── dpro/                       # Diffusion policy RL 优化
│   │   ├── diffusion_baselines/        # DIPO, QSM, DQL, AWR, RWR, IDQL 算法基线
│   │   ├── flow_baselines/             # FQL, SAC 算法基线
│   │   └── offlinerl_baselines/        # IBRL, CalQL, RLPD 算法基线
│   │
│   └── dataset/                    # 数据加载工具
│       ├── sequence.py                 # StitchedSequenceDataset
│       └── d3il/                       # D3IL 任务数据集
│
├── model/                          # 神经网络实现
│   ├── common/                     # 共享组件
│   │   ├── modules.py                  # MLP 块, RandomShiftsAug 等
│   │   ├── critic.py                   # Critic 网络 (CriticObs)
│   │   └── normalizer.py              # 观测数据归一化
│   │
│   ├── drifting/                   # Drifting Policy (1-NFE)
│   │   ├── drifting.py                 # DriftingPolicy 核心
│   │   ├── ft_ppo/
│   │   │   └── ppodrifting.py          # NoisyDriftingMLP + PPODrifting
│   │   └── ft_grpo/
│   │       └── grpodrifting.py         # NoisyDriftingPolicy + GRPODrifting
│   │
│   ├── flow/                       # 流匹配模型 (Flow matching)
│   │   ├── meanflow.py                 # MeanFlow 策略
│   │   ├── improved_meanflow.py        # 改进的 MeanFlow
│   │   ├── reflow.py                   # Rectified Flow
│   │   ├── shortcutflow.py             # ShortCut Flow
│   │   ├── consistency.py              # 一致性模型 (Consistency models)
│   │   ├── mlp_meanflow.py             # MeanFlowMLP 骨干网络
│   │   ├── mlp_shortcut.py             # ShortCut MLP 骨干网络
│   │   ├── mlp_consistency.py          # Consistency MLP 骨干网络
│   │   ├── ft_ppo/                     # 针对流模型的 PPO 包装器
│   │   │   ├── ppoflow.py                 # 基础 PPOFlow 类
│   │   │   ├── ppomeanflow.py              # PPOMeanFlow
│   │   │   └── pposhortcut.py              # PPOShortCut
│   │   └── ft_baselines/
│   │       └── fql.py                  # Flow Q-Learning
│   │
│   ├── diffusion/                  # DDPM/DDIM 扩散模型
│   ├── gaussian/                   # 高斯策略
│   └── rl/                         # 强化学习通用模块
│
├── cfg/                            # Hydra 配置文件
│   ├── gym/                        # OpenAI Gym & Franka Kitchen 任务配置
│   ├── robomimic/                  # RoboMimic 操作任务配置
│   ├── furniture/                  # FurnitureBench 组装任务配置
│   └── d3il/                       # D3IL 模仿学习任务配置
│
├── script/                         # 启动脚本
│   └── run.py                      # 统一实验启动器
│
├── tests/                          # 测试套件
│   ├── test_drifting_policy.py         # DriftingPolicy 核心功能测试
│   ├── test_ppo_drifting.py            # PPO 微调测试
│   ├── test_grpo_drifting.py           # GRPO 微调测试
│   ├── test_grpo_buffer.py             # GRPO 缓冲区测试
│   ├── test_end_to_end_smoke.py        # 端到端冒烟测试
│   └── test_static_and_config.py       # 导入与配置有效性测试
│
├── util/                           # 工具函数
│   ├── dirs.py                         # 目录管理
│   └── config.py                       # 配置管理工具
│
├── env/                            # 环境包装器
└── data_process/                   # 数据预处理脚本
```

---

## 3. 核心数据流

### 3.1 预训练数据流 (Pre-training Data Flow)

```
StitchedSequenceDataset (数据集)
    ↓
batch_data = (actions, observations)  # 动作和观测
    │
    │  actions: Tensor[B, T_a, D_a]      # B=批次大小, T_a=动作规划长度, D_a=动作维度
    │  observations: dict{"state": Tensor[B, T_c, D_obs]}
    │                                     # T_c=条件输入步数, D_obs=观测维度
    ↓
PreTrainAgent.get_loss(batch_data)
    ↓
model.loss(x1=actions, cond=observations)
    │
    │  [在 DriftingPolicy.loss() 内部]:
    │  1. x_gen = randn(B, T_a, D_a)          # 初始噪声
    │  2. x_pred = network(x_gen, t=1, r=0, cond)   # 1-NFE 前向推理传播
    │  3. V = compute_V(x_pred, actions)       # 计算基于专家数据的漂移场
    │  4. target = (x_pred + V).detach()       # 漂移后的目标值
    │  5. loss = MSE(x_pred, target)           # 训练损失计算
    ↓
loss.backward()  →  optimizer.step()
```

### 3.2 在线强化学习微调数据流 (PPO)

```
Environment (向量化环境, n_envs 个并行)
    ↓
obs_venv = {"state": np.array[n_envs, D_obs]}
    ↓
cond = {"state": Tensor[n_envs, D_obs]}   # 转移到计算设备
    ↓
PPODrifting.get_actions(cond)
    │
    │  [在 NoisyDriftingMLP 内部]:
    │  1. x0 = randn(n_envs, T_a, D_a)         # 初始噪声
    │  2. mean = network(x0, t=1, r=0, cond)    # 1-NFE 确定性输出均值
    │  3. std = noise_network(mean)              # 可学习的探索噪声标准差
    │  4. action = mean + std * randn(...)       # 随机采样动作
    │  5. log_prob = Normal(mean, std).log_prob(action)  # 计算对数概率
    ↓
action_venv = actions[:, :act_steps]     # 取前 act_steps 步的动作
    ↓
obs, reward, done = env.step(action_venv)
    ↓
PPOBuffer.add(obs, chains, reward, done)
    ↓  [经过 n_steps 收集后]
PPOBuffer.make_dataset()
    │  → GAE 优势评估计算
    │  → Returns 回报计算
    ↓
PPODrifting.loss(obs, chains, returns, values, advantages, old_logprobs)
    │  → 截断代理损失计算 (Clipped surrogate loss)
    │  → 价值函数损失计算 (Value function loss)
    │  → 熵奖励机制附加 (Entropy bonus)
    │  → 可选的 BC 行为克隆正则化约束
    ↓
loss.backward()  →  actor_optimizer.step() + critic_optimizer.step()
```

### 3.3 在线强化学习微调数据流 (GRPO — 无 Critic)

```
Environment (向量化环境)
    ↓
同源环境重置 (Homogeneous Reset): G 条从相同初始状态出发的轨迹
    ↓
对于每条轨迹 g 在 [1..G] 中:
    ├── 通过 NoisyDriftingPolicy 采样动作
    ├── 收集 (状态, 动作, 对数概率, 奖励) 数据组
    └── 计算单条轨迹的回报 R_g
    ↓
GRPOBuffer.normalize_advantages()
    │  advantages = (returns - mean(returns)) / (std(returns) + eps)
    │  零方差截断保护 (Zero-variance protection): 如果 std < 1e-6, 则强置 advantages = 0
    ↓
GRPODrifting.compute_loss(obs, actions, advantages, old_log_probs)
    │  1. curr_log_prob = NoisyDriftingPolicy.get_log_prob(obs, actions)
    │  2. ratio = exp(curr_log_prob - old_log_prob)
    │  3. surr1 = ratio * advantages
    │  4. surr2 = clamp(ratio, 1-eps, 1+eps) * advantages
    │  5. policy_loss = -min(surr1, surr2).mean()
    │  6. kl_div = analytical_kl(current_dist, ref_dist)   # 无 Critic 的解析 KL 计算!
    │  7. loss = policy_loss + beta * kl_div
    ↓
loss.backward()  →  optimizer.step()
```

---

## 4. 模型层深度剖析

### 4.1 MeanFlowMLP — 共享网络骨干

**文件位置:** `model/flow/mlp_meanflow.py`

`MeanFlowMLP` 是被 MeanFlow 和 Drifting Policy 共享的神经网络基础架构。它接收如下输入：

| 输入张量 | 形状维度 | 描述意义 |
|-------|-------|-------------|
| `x` | `[B, T_a, D_a]` | 带有噪声的动作序列 |
| `t` | `[B]` | 时间步 (0→1) |
| `r` | `[B]` | 辅助变量 (分辨率) |
| `cond` | `dict{"state": [B, T_c, D_obs]}` | 条件反馈观测数据 |

**输出：** `[B, T_a, D_a]` — 预测的位移速度/动作场。

### 4.2 DriftingPolicy

**文件位置:** `model/drifting/drifting.py`

1-NFE Drifting Policy 的核心实现：

- **`compute_V(x, y_pos, y_neg)`**: 利用基于样本间相互作用的 RBF 权重核计算漂移场
- **`loss(x1, cond)`**: 通过计算网络原始预测值与加上漂移场后的目标值之间的均方误差（MSE），实现预训练阶段的损失控制
- **`sample(cond, ...)`**: 使用固定的时间系数 `t=1.0, r=0.0` 进行单步前向传播采样推理

### 4.3 PPODrifting

**文件位置:** `model/drifting/ft_ppo/ppodrifting.py`

包含两个关键组件：
- **`NoisyDriftingMLP`**: 对 MeanFlowMLP 网络进行带有可学习探索噪声的包装机制扩展（包含使用 MLP 来预测对数方差）
- **`PPODrifting(PPOFlow)`**: 基于单步对数概率计算的 PPO 目标网络架构实现

### 4.4 GRPODrifting

**文件位置:** `model/drifting/ft_grpo/grpodrifting.py`

包含三个关键组件：
- **`_tanh_jacobian_correction(u)`**: 为了在极值边界处求解 Tanh-Normal 分布保证数据稳定的数值安全雅可比矩阵计算
- **`NoisyDriftingPolicy`**: 对于基于 Tanh-Normal 的动作分布施加合规的对数概率包装
- **`GRPODrifting`**: 无 Critic (Value-free) 模式的 GRPO 目标函数，包含使用解析解方式计算 KL 散度

---

## 5. 智能体层深度剖析

### 5.1 PreTrainAgent (基类)

**文件位置:** `agent/pretrain/train_agent.py`

它提供了具有普适性的离线预训练循环：
1. 借由 Hydra 系统 `_target_` 进行模型反射实例化
2. 基于 EMA (指数移动平均指数) 模型的状态管理
3. 专家数据集处理与按批次的数据加载与切分
4. 内部优化器和学习率衰减调度器 (LR scheduler)
5. 权重的自动保存与能够自动接续恢复训练的 Checkpoint 管理
6. 可选配：在训练周期内基于 MuJoCo 直接开启评估校验测试
7. 整合基于 WandB 的状态记录日志系统

**子类必须复写的核心抽象方法方法：**
- `get_loss(batch_data)` — 指定特定的网络前向损失计算手段
- `inference(cond)` — 基于观测输入产生结果的推理生成逻辑

### 5.2 训练智能体继承树 (Agent Hierarchy)

```
PreTrainAgent
├── TrainDriftingAgent             # 1-NFE, 基于漂移场损失
│   └── TrainDriftingDispersiveAgent   # + 弥散正则化约束
├── TrainMeanFlowAgent             # 5步欧拉流算法匹配
├── TrainShortCutAgent             # 基于自适应步数的 shortcut 流
│   ├── TrainShortCutDispersiveAgent
│   └── TrainMeanFlowDispersiveAgent
├── TrainReFlowAgent               # Rectified flow
├── TrainConsistencyAgent          # 一致性模型蒸馏
├── TrainDiffusionAgent            # 基于经典 DDPM/DDIM 的扩散模型
└── TrainGaussianAgent             # 普通的单步 Gaussian 行为克隆

TrainPPOShortCutAgent (PPO 微调智能体基准)
├── TrainPPODriftingAgent          # Drifting PPO (针对本体状态输入)
├── TrainPPOMeanFlowAgent          # MeanFlow PPO (针对本体状态输入)
├── TrainPPOFlowAgent              # 通用的流匹配策略 PPO
├── TrainPPODiffusionAgent         # Diffusion PPO
├── TrainPPOGaussianAgent          # Gaussian PPO
└── TrainPPOImgShortCutAgent       # ShortCut PPO (针对图像输入)
    ├── TrainPPOImgDriftingAgent   # Drifting PPO (针对图像输入)
    ├── TrainPPOImgMeanFlowAgent   # MeanFlow PPO (针对图像输入)
    ├── TrainPPOImgFlowAgent       # Flow PPO (针对图像输入)
    └── TrainPPOImgDiffusionAgent  # Diffusion PPO (针对图像输入)

TrainGRPODriftingAgent             # GRPO (无Critic模式，目前仅对 Drifting 架构生效)
```

---

## 6. 配置系统

所有的 DMPO 算法实验都受控于 Hydra YAML 配置文件体系，并按照以下原则进行编排组织：

```
cfg/{环境系列_env_suite}/{训练阶段_stage}/{具体任务_task}/{特定的配置表称呼_config_name}.yaml
```

**经典的配置路径案例:**
```
cfg/gym/pretrain/hopper-medium-v2/pre_drifting_mlp.yaml
cfg/gym/finetune/hopper-v2/ft_ppo_drifting_mlp.yaml
cfg/gym/finetune/hopper-v2/ft_grpo_drifting_mlp.yaml
cfg/gym/eval/hopper-v2/eval_drifting_mlp.yaml
cfg/robomimic/pretrain/can/pre_drifting_mlp_img.yaml
```

**配置文件内涉及的关键段落设定:**

| 段落节点 | 用处 | 配置项举例 |
|---------|---------|--------------|
| `model` | 指定构建特定策略神经网络体系结构的方案 | `_target_`, `mlp_dims`, `act_min/max` |
| `train` | 决定训练进程长短与批处理控制宏观参数 | `n_epochs`, `batch_size`, `learning_rate` |
| `env` | 测试/交互环境自身设置 | `n_envs`, `name`, `max_episode_steps` |
| `train_dataset` | 数据集加载方式和目标设置 | `_target_`, `dataset_path`, `horizon_steps` |
| `ema` | 滑动平均更新算法配置 | `decay` |

---

## 7. Drifting Policy 与 Mean Flow Policy — 数学与代码的对比

### 7.1 数学基础理念

**Mean Flow Policy (均值流策略)** 致力于通过构建起一个速度预测场向量 `v(x, t)` 的方式，在设定时间区间 `t ∈ [0,1]` 中，把高斯噪声 `x_0 ~ N(0,I)` 映射传输到目标分布的数据点 `x_1` 处：

```
dx/dt = v(x, t)     →   x_1 = x_0 + ∫₀¹ v(x_t, t) dt
```

在推理生成（Inference）阶段下，这个时域积分往往需被多次欧拉步进行切分并逼近运算（一般需要花费 5 次 NFE 处理时长）。

**Drifting Policy (均值漂移策略)** 将上述复杂的多次流程直接坍塌折叠在了一个步内即可完成。通过训练直接让神经网络负责输出最后步结果，并在损失网络结构里施加外部力学上对应的专家先验漂移校正（drift correction）作为学习目标：

```
x_pred = f_θ(x_0, t=1, r=0, cond)           # 单次走完前向评估传播
V = compute_V(x_pred, x_expert)              # 面向专家真实数据集的强制漂移场计算
target = (x_pred + V).detach()               # 加入场效应以后的最终强迫目标靶向点
loss = MSE(x_pred, target)                   # 自监督机制进行闭环自我学习逼近修正
```

### 7.2 代码层面全方面比照

| 各个方面环节 | MeanFlow | Drifting |
|--------|----------|---------|
| **存放核心文件** | `model/flow/meanflow.py` | `model/drifting/drifting.py` |
| **底层共用网络** | `MeanFlowMLP` | `MeanFlowMLP` (采用同底骨架) |
| **评估推理需要的 NFE 耗时** | 5 步 (多次欧拉前向推演) | 1 步 (极速一阶直出前向) |
| **损失函数设计层面** | 面向速度场匹配的 MSE 误差体系 | 面向加持纠正偏移后目标向量的极短MSE误差体系 |
| **预训练时采用的步幅参数 `t`** | 由 `t ~ U(0,1)` 统括时间段均分采样 | 固定使用一步 `t=1.0` |
| **分布解算用的参数 `r`** | 被采样亦或恒为常数进行适配 | 永远固定使用 `r=0.0` |
| **应用 PPO 的套壳外置算法** | `PPOMeanFlow` | `PPODrifting` |
| **应用 GRPO 的套壳外置算法** | — (不支持) | `GRPODrifting` |
| **调用的离线训练智能体封装** | `TrainMeanFlowAgent` | `TrainDriftingAgent` |
| **使用的在评估智能体封装** | `eval_meanflow_agent.py` | `eval_drifting_agent.py` |

### 7.3 最大不同之处代码体现解析: 针对漂移场的构建算式 (The drift field)

```python
# 该内容摘录自 DriftingPolicy 实体类内部的 compute_V() 算法节点中:
# 1. 第一步需测算全批次的预测动作项集合及其向专家动作项之间的相互欧氏距离之差
diff = x.unsqueeze(1) - y_pos.unsqueeze(0)    # [B, B, T_a, D_a]
dist_sq = (diff ** 2).sum(dim=(-1, -2))         # [B, B]

# 2. 第二步带入径向基分布函数(RBF kernel)以此给出对应权重加持
weights = torch.exp(-dist_sq / (2 * bandwidth**2))  # [B, B]

# 3. 如果设定使用自身剔除掩盖的话，将阻断自己对自己的拉扯计算项
if mask_self:
    weights.fill_diagonal_(0)

# 4. 根据权重计算归一化的、导向专家的修正纠偏漂移引力分布场
weights_norm = weights / (weights.sum(dim=1, keepdim=True) + eps)
V_pos = (weights_norm.unsqueeze(-1).unsqueeze(-1) * diff).sum(dim=1)

# 5. 最后依据外部超参配置，根据乘数将所有拉扯量施加放缩运算反馈回总体漂移量里作为返回出参
V_total = drift_coef * V_pos - neg_drift_coef * V_neg
```

---

## 8. 微调架构：PPO 与 GRPO

### 8.1 基于 PPO 算法微调流程体系

无论是 MeanFlow 还是 Drifting 模型在接入在线环境微调时，都服从隶属于同一组的抽象 PPO 实现（即共用的 `PPOFlow` 基准类体系）。他们的区别只在于：
- **网络单次对数概率求解实现**: 考虑到 Drifting 是彻底的 1-NFE 单级直出响应，只需要做单一的对应高斯演变过程便可求出；而对于采用传统多步机制的 MeanFlow 则要用一种特殊方式连算链式步骤结构下的每个中间联合条件分布过程概率。
- **关于环境交互推理步骤约束**: Drifting 必须强性限定要求强制设定参数 `inference_steps=1`。
- **马尔可夫演变数据链路大小限制**: Drifting 参数里的时序推理链永远局限在一个仅含初始噪声(Noise) 到单步动作出值(Action)的硬绑定两点链路数据。

### 8.2 基于 GRPO (仅供 Drifting Policy 使用)

GRPO(极轻量组相对强化微调)架构具备一些目前系统专为提升极速单步 Drifting 机制的优势点赋能的设计特征：
- **完全去掉外置的 Critic 结构体系** — 取消使用一个必须在网络前向内做单独评分估计计算庞大规模网络评估模块层，不仅极大降低参数占用空间大小甚至也能一劳永逸直接回避该类型估价带来的有偏引导风险失真（Value estimation bias）
- **轨迹群体级别的内生优势标准化计算处理** — 对于一次收集的数条平齐启动过程在重构的同轨迹奖励中进行组级的相对数据极化统计，并直接提取作为具有指导判断效应 Z-score 统计标准化。
- **严格遵循解析解范式的 KL 散度防偏约束估计** — 完全靠带入两个动作分布内的预设函数和均指值等数学推算替代采样，根治利用对数似然残差直接计算造成的过大概率随机数据扰动方差。
- **结合有极值边界效应约束的 Tanh-Normal 系统动作出值截断处理并匹配严格雅可比数值变换** — 有效校正高斯体系内采样输出无限泛扩展在必须对标实际夹具执行设备受机械固定运动位置限制 $[-1, 1]$ 下引发的正偏对数理论失序的问题

---

## 9. 环境集成

### 9.1 默认可支持的所有强化基准实境环境套件体系

| 集合类别 | 实施囊括任务清单 | 面向观测反馈介质类型 | 环境动作自由空间要求情况 |
|-------|-------|-------------|--------------|
| **Gym (D4RL经典基准包)** | HalfCheetah, Hopper, Walker2d, Ant, Humanoid | 独立解包分离出来的 State 状态数组向量数值 | 全维度的连续型 Continuous 动作约束支持 |
| **Franka Kitchen** | kitchen-mixed, kitchen-complete, kitchen-partial | 独立解包分离出来的 State 状态数组向量数值 | 全维度的连续型 Continuous 动作约束支持 |
| **RoboMimic** | Can, Lift, Transport, Square | 原生级 RGB 多机位画面 + 本体随动反馈状态 State | 全维度的连续型 Continuous 动作约束支持 |
| **FurnitureBench** | one_leg_low, one_leg_high, square_table | 原生级 RGB 多机位画面 + 本体随动反馈状态 State | 全维度的连续型 Continuous 动作约束支持 |
| **D3IL** | Avoid, Aligning, Pushing, Sorting, Stacking | 独立解包分离出来的 State 状态数组向量数值 | 全维度的连续型 Continuous 动作约束支持 |

### 9.2 关于实际交互中的感知数据处理 (Observation Processing)

关于只接受固定内部物理参量的模拟应用，统一封装传递的简化调用词典模式如下形式：
```python
cond = {"state": Tensor[B, D_obs]}
```

针对具备外部摄像头参与进行视觉感知混合模拟任务处理，系统会同时传输一个复合结构：
```python
cond = {
    "rgb": Tensor[B, C, H, W],       # 来自外部的 RGB 成像输入矩阵
    "state": Tensor[B, D_proprio],     # 来自机器主体的本体觉控制关联参量
}
```

值得着重指出，处理并加载图象环境关联训练策略时(`TrainPPOImgDriftingAgent` 等代理群)，在后台均已被隐式整合挂载接入以应对抗模糊漂移问题发生的画面偏移扭曲处理外挂方案（如 RandomShiftsAug），甚至还可以实现用于解决算存限制下的低资源梯次数据累计拼接叠加反馈技巧应用机制（gradient accumulation）辅助推进模型在环境内有效正常拟合。
