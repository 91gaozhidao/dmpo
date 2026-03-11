# Drifting Policy 指南

**Drifting Policy** 是一种 1-NFE（单次神经函数评估）生成模型，它使用“漂移场”（drifting field）在单次前向传递中将噪声直接映射到专家动作。它旨在保持基于流的模型的多峰建模能力的同时，实现高效率推理。

---

## 1. 数学基础

### 1.1 背景：用于流匹配的连续时间 ODE

标准的流匹配模型（例如 MeanFlow）通过常微分方程 (ODE) 定义连续概率路径：

$$\frac{dx_t}{dt} = v_\theta(x_t, t, s), \quad t \in [0, 1]$$

其中 $v_\theta$ 是以状态 $s$ 为条件的学习到的速度场，$x_0 \sim \mathcal{N}(0, I)$ 是高斯噪声。推理需要通过数值积分（例如欧拉法）求解此 ODE，产生 $K$-NFE，其中 $K \geq 1$ 是离散化步骤数。

**MeanFlow** 对此进行了改进，它学习一个平均速度 $u_\theta(x_t, t, r, s)$，直接从时间 $t$ 映射到时间 $r$：

$$x_r = x_t - (t - r) \cdot u_\theta(x_t, t, r, s)$$

这允许灵活的步数，但网络仍必须学习完整的速度场。

### 1.2 Drifting Policy：将均值漂移前置到训练阶段

Drifting Policy 采用了一种根本不同的方法，**将迭代生成过程前置**到训练阶段。它不学习推理时必须积分的速度场，而是学习一个直接映射 $f_\theta: \mathcal{N}(0, I) \times \mathcal{S} \to \mathcal{A}$，在单次前向传递中产生动作。

**训练阶段：** 给定一批专家动作 $\{a_i^*\}_{i=1}^B$ 和噪声样本 $\{z_i\}_{i=1}^B$，其中 $z_i \sim \mathcal{N}(0, I)$：

1. 计算网络预测：$\hat{a}_i = f_\theta(z_i, s_i)$，固定 $t=1, r=0$
2. 计算驱动预测向专家数据移动的漂移场 $V_{\text{total}}$：

$$V_{\text{pos},i} = \alpha_+ \sum_{j} w_{ij}^+ (a_j^* - \hat{a}_i), \quad w_{ij}^+ = \frac{\exp(-\|{\hat{a}_i - a_j^*}\|^2 / 2h^2)}{\sum_k \exp(-\|{\hat{a}_i - a_k^*}\|^2 / 2h^2)}$$

$$V_{\text{neg},i} = \alpha_- \sum_{j} w_{ij}^- (\hat{a}_i - a_j^-), \quad w_{ij}^- = \frac{\exp(-\|{\hat{a}_i - a_j^-}\|^2 / 2h^2)}{\sum_k \exp(-\|{\hat{a}_i - a_k^-}\|^2 / 2h^2)}$$

3. 构造漂移后的目标：$\tilde{a}_i = \hat{a}_i + V_{\text{pos},i} + V_{\text{neg},i}$
4. 最小化：$\mathcal{L} = \frac{1}{B}\sum_i \| f_\theta(z_i, s_i) - \text{sg}(\tilde{a}_i) \|^2$，其中 $\text{sg}(\cdot)$ 是停止梯度（stop-gradient）

**推理阶段 (1-NFE)：**
$$a = \text{clamp}\left(f_\theta(z, s)\big|_{t=1, r=0},\ [-1, 1]\right), \quad z \sim \mathcal{N}(0, I)$$

### 1.3 RBF 核与漂移场

RBF（径向基函数）核权重 $w_{ij}$ 确保了平滑且对距离敏感的交互：
- 附近的专家样本产生更强的吸引力（正向漂移）
- 附近的负样本产生更强的排斥力（负向漂移）
- 带宽 $h$ 控制交互范围

当 `mask_self=True` 时，排除自身交互 ($i = j$) 以防止出现干扰解。

---

## 2. 架构与模块依赖

```
model/drifting/
├── drifting.py                 # DriftingPolicy: 核心 1-NFE 策略
│   ├── 使用: model.flow.mlp_meanflow.MeanFlowMLP (仅状态)
│   └── 使用: model.flow.mlp_meanflow.MeanFlowViT (基于视觉)
├── ft_ppo/
│   └── ppodrifting.py          # PPODrifting: PPO 微调包装器
│       ├── NoisyDriftingMLP: 添加可学习的探索噪声
│       └── 继承: model.flow.ft_ppo.ppoflow.PPOFlow
└── ft_grpo/
    └── grpodrifting.py         # GRPODrifting: 无 Critic 的 GRPO 包装器
        ├── NoisyDriftingPolicy: 具有雅可比修正的 Tanh-Normal 策略
        └── 使用: agent.finetune.grpo.buffer.GRPOBuffer
```

**主干网络共享：** Drifting Policy 使用与 MeanFlow 相同的 `MeanFlowMLP` 和 `MeanFlowViT` 网络，确保架构一致以进行受控比较。网络接收输入 `(z, t=1.0, r=0.0, cond)` 并输出形状为 `(B, Ta, Da)` 的动作预测。

---

## 3. 关键超参数

| 参数                  | 默认值  | 描述                                                   |
| --------------------- | ------- | ------------------------------------------------------ |
| `drift_coef`          | 0.1     | 正向漂移（向专家动作移动）的比例因子 $\alpha_+$。      |
| `neg_drift_coef`      | 0.05    | 负向漂移（远离非专家样本）的比例因子 $\alpha_-$。      |
| `mask_self`           | `False` | 是否在漂移场中排除自身交互 ($i=j$)。                   |
| `bandwidth`           | 1.0     | 控制交互范围的 RBF 核带宽 $h$。                        |
| `max_denoising_steps` | 1       | **必须为 1**。在运行时强制执行；增加此值会违反 1-NFE。 |

### PPO 特定参数

| 参数                         | 默认值 | 描述                                   |
| ---------------------------- | ------ | -------------------------------------- |
| `min_logprob_denoising_std`  | 0.05   | 用于 log-prob 的最小探索噪声标准差。   |
| `max_logprob_denoising_std`  | 0.12   | 用于 log-prob 的最大探索噪声标准差。   |
| `use_time_independent_noise` | `True` | 使用与时间无关的噪声（适用于 1-NFE）。 |
| `denoising_steps`            | 1      | 必须符合 1-NFE 约束。                  |

### GRPO 特定参数

| 参数            | 默认值 | 描述                                          |
| --------------- | ------ | --------------------------------------------- |
| `group_size`    | 16     | 每组的轨迹数 $G$ ($G \geq 16$)。              |
| `kl_beta`       | 0.05   | 初始 KL 散度惩罚系数。                        |
| `kl_beta_min`   | 0.001  | 衰减后的最小 KL 惩罚。                        |
| `kl_beta_decay` | 0.995  | 每轮迭代 KL 惩罚的指数衰减率。                |
| `epsilon`       | 0.2    | 代理损失（surrogate loss）的 PPO 式裁剪范围。 |
| `init_log_std`  | -0.5   | 探索噪声的初始对数标准差。                    |

---

## 4. 可用配置

### 离线预训练

| 领域      | 配置名称               | 网络        | 命令目录                         |
| --------- | ---------------------- | ----------- | -------------------------------- |
| Gym       | `pre_drifting_mlp`     | MeanFlowMLP | `cfg/gym/pretrain/<task>/`       |
| Robomimic | `pre_drifting_mlp_img` | MeanFlowViT | `cfg/robomimic/pretrain/<task>/` |
| D3IL      | `pre_drifting_mlp`     | MeanFlowMLP | `cfg/d3il/pretrain/<task>/`      |
| Furniture | `pre_drifting_mlp`     | MeanFlowMLP | `cfg/furniture/pretrain/<task>/` |

### PPO 微调

| 领域      | 配置名称                  | 命令目录                         |
| --------- | ------------------------- | -------------------------------- |
| Gym       | `ft_ppo_drifting_mlp`     | `cfg/gym/finetune/<task>/`       |
| Robomimic | `ft_ppo_drifting_mlp_img` | `cfg/robomimic/finetune/<task>/` |
| D3IL      | `ft_ppo_drifting_mlp`     | `cfg/d3il/finetune/<task>/`      |
| Furniture | `ft_ppo_drifting_mlp`     | `cfg/furniture/finetune/<task>/` |

### GRPO 微调（无 Critic）

| 领域      | 配置名称                   | 命令目录                         |
| --------- | -------------------------- | -------------------------------- |
| Gym       | `ft_grpo_drifting_mlp`     | `cfg/gym/finetune/<task>/`       |
| Robomimic | `ft_grpo_drifting_mlp_img` | `cfg/robomimic/finetune/<task>/` |
| D3IL      | `ft_grpo_drifting_mlp`     | `cfg/d3il/finetune/<task>/`      |
| Furniture | `ft_grpo_drifting_mlp`     | `cfg/furniture/finetune/<task>/` |

---

## 5. 训练阶段

### 第 1 阶段：预训练

在专家数据上使用行为克隆（Behavior Cloning）训练基础漂移场。

**Gym（基于状态）：**
```bash
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp
```

**Robomimic（基于图像）：**
```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_drifting_mlp_img
```

### 第 2 阶段：微调

**PPO 微调：**
```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_drifting_mlp \
  base_policy_path=<PRETRAINED_CHECKPOINT_PATH>
```

**GRPO 微调（无 Critic，基于组）：**
GRPO 取消了 Critic/Value 网络。优势（Advantage）通过组内 Z-score 归一化计算：
$$A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}, \quad \sigma_R = \text{std}(\{R_1, \ldots, R_G\}), \quad \epsilon = 10^{-6}$$

当 $\sigma_R < \epsilon$ 时（例如，在稀疏奖励任务中回报全部为零），优势被设置为零以防止除以零错误。

```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_grpo_drifting_mlp \
  base_policy_path=<PRETRAINED_CHECKPOINT_PATH>
```

---

## 6. 运行时断言与安全

实现包括运行时断言以确保正确性：

1. **动作边界验证**：在 `sample()` 之后，验证动作是否位于 $[-1, 1]$ 之间。
2. **对数概率有限性**：在 PPO 和 GRPO 中，雅可比修正后检查对数概率是否为 NaN/Inf。
3. **零方差保护**：在 GRPO 中，具有相同回报的组 ($\sigma_R < 10^{-6}$) 产生零优势而不是 NaN。

---

## 7. 调优建议

1.  **稳定性**：如果动作过于抖动，请减小 `drift_coef`（例如从 0.1 减小到 0.05）。
2.  **探索**：如果模型塌缩到单个模式，请增加 `neg_drift_coef` 以将样本推开。
3.  **视觉编码器**：对于基于图像的任务，确保 `img_cond_steps` 与您的预训练设置相匹配。
4.  **推理步数**：在微调配置中始终验证 `denoising_steps=1` 和 `ft_denoising_steps=1`。
5.  **GRPO 组大小**：使用 $G \geq 16$ 进行可靠的优势估计。较小的组会增加方差。
6.  **KL 衰减**：对于稀疏奖励任务（例如 Adroit），使用较慢的衰减 (`kl_beta_decay=0.999`)。
