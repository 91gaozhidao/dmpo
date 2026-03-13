# Online RL Fine-Tuning Algorithm Analysis for Drifting Policy

## 1. 当前事实总结

### Drifting 核心机制

- **1-NFE / 1-Step Action Generation**：`a = f_θ(z)`，`z ~ N(0,I)`，单次前向传播映射
  高斯噪声到动作空间。
- **无显式 log-probability**：Drifting 没有显式密度模型。它是一个隐式生成策略
  (implicit generative policy)，通过漂移场 `V_{p,q}` 驱动样本向高质量区域移动。
- **`predict()` 可微分**：`DriftingPolicy.predict()` 没有 `@torch.no_grad()` 装饰器，
  梯度可以从动作反向传播到网络参数。这意味着 pathwise / reparameterization 风格的
  优化天然被支持。
- **漂移场更新**：Actor 更新使用 `x -> x + V_{p,q}(x)` 形式，其中正样本集 `p` 和
  负样本集 `q` 定义了漂移方向。Q-guided 方案用 critic 评分选择正样本。
- **随机性来源**：每次推理时采样不同的 `z ~ N(0,I)` 产生不同的动作。这是隐式的
  exploration 机制。

### 数据资产现实

- **离线预训练数据**：`states / actions / traj_lengths`，部分有 `images`。
  大多数 **没有 `rewards` / `terminals`**。这是 demonstration-style 数据。
- **在线环境**：可以提供真实的 `(s, a, r, s', done)` transition 数据。
- **预训练 checkpoint**：可获得。"先离线预训练，再在线 RL 微调" 是可行路径。
- **关键约束**：不能把 "offline reward-labeled dataset 必然存在" 作为算法前提。

### 仓库实现现状

- **Q-Guided Drifting**（`model/drifting/ft_qguided/`）是当前唯一实现的 drifting
  RL 微调方案。
- **PPO / GRPO**：`model/drifting/ft_ppo/` 和 `model/drifting/ft_grpo/` 目录不存在。
  仅在 checkpoint 加载兼容代码中有引用。
- **当前 Q-Guided 依赖离线 rewards**：`TrainQGuidedDriftingAgent._cache_offline_dataset()`
  和 `StitchedSequenceQLearningDataset` 要求离线数据包含 `rewards` 和 `terminals`。
  这在当前数据资产下无法满足（RoboMimic image 除外，有 fallback）。

---

## 2. 候选算法比较

### 2.1 纯 Online PPO / GAE / On-Policy Actor-Critic

**为什么看起来可能适合 drifting：**
- PPO 是成熟的在线 RL 算法，不需要离线 rewards。
- 只需在线 rollout 数据即可训练。
- 大量 continuous control 成功案例。

**与当前数据资产是否匹配：**
- ✅ 不依赖离线 rewards，完全匹配当前数据条件。

**主要优点：**
- 不需要 critic 提前训练（on-policy 自然同步更新）。
- 训练流程简单、稳定性好。
- 不需要 replay buffer。

**主要问题：**
- ❌ **PPO 强制要求显式 log-probability `log π(a|s)`**。Drifting 是一个隐式
  生成模型，没有解析密度。要为 drifting 加 log-prob，只能：
  - (a) 在 drifting 输出上叠加独立的参数化分布（如 Tanh-Normal），但这实质上
    把策略从 "drift-field-based" 变成了 "distribution-based"，严重改变了 drifting
    的核心机制。
  - (b) 用 change-of-variables 公式计算 Jacobian，但 drifting 的网络是任意
    transformer/UNet，Jacobian 计算代价极高且不稳定。
- ❌ **PPO ratio `π_new/π_old` 在隐式生成模型上没有良好定义**。
- ❌ 为适配 PPO 而注入的 log-prob 机制会扭曲 drifting 的训练动力学。
- ❌ On-policy 的样本效率较低，与 drifting 多样本评估的特性不太匹配。

**结论：** 不推荐作为主线。需要根本性地改造 drifting 来适配 PPO，代价过高。

---

### 2.2 纯 Online Reparameterized Actor-Critic (SVG / SAC-like / Pathwise)

**为什么看起来可能适合 drifting：**
- Drifting 的 `predict()` 天然可微分：`a = f_θ(z), z ~ N(0,I)`
- Pathwise gradient：`∇_θ J = E_z[∇_a Q(s, a) · ∇_θ f_θ(z)]`
- 不需要 log-probability，只需要动作对参数的梯度
- 与 drifting 的生成机制完美契合

**与当前数据资产是否匹配：**
- ✅ Critic 可以纯在线训练：从 replay buffer 的 `(s, a, r, s', done)` 学习
- ✅ 不依赖离线 rewards

**主要优点：**
- 不需要修改 drifting 核心机制
- Critic 提供 off-policy 信号，样本效率比 PPO 高
- 梯度路径简洁：`loss = -Q(s, f_θ(z))`
- 探索自然来自 `z ~ N(0,I)` 的随机性

**主要问题：**
- ⚠️ 纯 pathwise gradient `loss = -Q(s, a)` 会让策略快速坍缩到 critic 认为最优
  的单点动作，丧失 drifting 的分布特性。
- ⚠️ 没有 drifting V-field 的约束，actor 更新不再保持 "向高质量区域漂移"
  的原始语义。
- ⚠️ critic 过拟合或 Q-value 发散时，actor 可能被误导到 out-of-distribution 区域。
- ⚠️ 需要仔细调节 critic 训练速度与 actor 更新频率的平衡。

**结论：** 虽然与 drifting 的微分结构契合，但丢弃了 drifting 的 V-field 更新机制。
可以作为备选方案，但不应作为主线，因为它把 drifting 退化成了普通的
reparameterized policy。

---

### 2.3 纯 Online Q-Guided Drifting（去除离线 rewards 依赖）

**为什么看起来可能适合 drifting：**
- 完整保留 drifting 的 V-field actor 更新：
  - 采样 N 个候选动作
  - 用 critic Q 值评分
  - 选 top-K 作为正样本
  - 计算漂移场 `V_{p,q}`
  - Actor loss = `MSE(x, sg(x + V))`
- Critic 的 TD 学习只需要 `(s, a, r, s', done)`，这些可以完全来自在线 rollout

**与当前数据资产是否匹配：**
- ✅ Pretraining 阶段：只需 `states / actions`（无 rewards 要求），已有
- ✅ Online fine-tuning 阶段：critic 从在线 replay buffer 学习，reward 来自环境
- ✅ 不依赖离线 rewards

**主要优点：**
- **完整保留 drifting 核心机制**：actor 更新依然是漂移场驱动，不是 policy gradient
- **Critic 仅用于指导正样本选择**，不直接产生 actor 梯度
- **Off-policy**：replay buffer 支持高样本效率
- **现有代码 90% 可复用**：只需移除离线 rewards 依赖
- **Reference anchor**：已有 `reference_anchor_coeff` 防止过度偏离预训练分布

**主要问题：**
- ⚠️ 初始阶段 critic 质量低（没有离线数据预热），前几十次迭代的 Q 值噪声大
- ⚠️ 需要足够的在线 rollout 来填充 replay buffer 后 critic 才有意义
- ⚠️ 解决方案：增加 critic warmup 阶段的 rollout 数量，延迟 actor 更新

**结论：** **最推荐方案。** 它在 drifting 原理 + 当前数据资产 + 当前仓库结构三者
之间最匹配。

---

### 2.4 其他方向：GRPO-style (Group Relative Policy Optimization)

**为什么看起来可能适合 drifting：**
- GRPO 不需要 critic，使用 group 内 Z-score 归一化的 return 作为 advantage
- 仓库中已有 `GRPOBuffer` 实现

**与当前数据资产是否匹配：**
- ✅ 不依赖离线 rewards
- ✅ 只需在线 rollout 的 episodic return

**主要优点：**
- 不需要 critic 网络
- 概念简单

**主要问题：**
- ❌ **GRPO 仍需要 log-probability**：`GRPOBuffer` 存储 `old_log_probs`，
  GRPO 本质上是 policy gradient 变体
- ❌ 与 PPO 一样，需要为 drifting 注入人工 log-prob 机制
- ❌ 仓库中没有 `GRPODrifting` 模型实现（只有 buffer）
- ❌ On-policy 样本效率低

**结论：** 不推荐。与 PPO 有相同的 log-prob 问题。

---

## 3. 最终推荐

### 主线推荐：纯 Online Q-Guided Drifting

**推荐理由（三重匹配）：**

1. **Drifting 原理匹配**：
   - 完整保留 `x -> x + V_{p,q}(x)` 漂移场更新
   - Critic 仅用于定义"什么是好样本"（正样本选择），不直接产生 actor 梯度
   - 不需要 log-prob，不改变 drifting 的生成机制
   - 保持 drifting 作为"隐式生成策略"的本质

2. **当前数据资产匹配**：
   - Pretraining：使用 reward-less demonstration 数据（已有）
   - Fine-tuning：critic 100% 从在线 replay buffer 学习
   - 不依赖离线 rewards
   - 在线环境提供真实 reward 信号

3. **仓库结构匹配**：
   - `QGuidedDrifting` 模型层已完备（actor loss、critic loss、target critic）
   - `DictReplayBuffer` 已实现
   - `TrainQGuidedDriftingAgent` 只需修改离线数据依赖
   - 评估流程、checkpoint 保存/加载、日志记录全部可复用

### 关键问题的显式回答

**Q1: drifting 更像"可写显式 logprob 的策略"还是"隐式生成策略"？**
→ **隐式生成策略。** `a = f_θ(z)` 是一个 deterministic mapping，没有解析密度。
随机性来自输入噪声 `z`，而非参数化分布。漂移场 `V_{p,q}` 基于 kernel
归一化计算，不涉及概率密度。

**Q2: drifting 是否天然适合 PPO？**
→ **不适合。** PPO 要求 `log π(a|s)` 和 ratio `π_new/π_old`。Drifting 没有
`π(a|s)` 的解析形式。强行添加（如叠加 Tanh-Normal）会从根本上改变策略的
训练动力学，把 "drift-field-based" 退化为 "distribution-based"。

**Q3: drifting 是否更适合 pathwise actor-critic？**
→ **部分适合。** `predict()` 的可微性使 pathwise gradient 自然可行。但纯
pathwise 的 `loss = -Q(s, f_θ(z))` 丢弃了 drifting 的 V-field 语义。
如果需要保持 drifting 的核心机制，Q-guided 比纯 pathwise 更合适。

**Q4: 在没有 offline rewards 的条件下，Q-guided 能否作为主线？**
→ **可以，只要改为纯 online critic training。** Q-guided drifting 中 critic 的
唯一作用是为候选动作评分。Critic 训练只需要 `(s, a, r, s', done)`。这些
可以 100% 来自在线 replay buffer。不需要离线 rewards。

**Q5: 在"已有 pretrain checkpoint + 只有 online reward"的条件下，最合适的路线？**
→ **纯 Online Q-Guided Drifting。**
  1. 加载预训练 checkpoint 作为 actor 初始化
  2. 初始化一个随机 critic
  3. 收集在线 rollout 填充 replay buffer
  4. 先 warmup critic（只更新 critic，不更新 actor）
  5. 然后同时更新 critic 和 actor（actor 用 Q-guided drifting field）

---

## 4. 不推荐方案

| 方案 | 不推荐原因 |
|------|-----------|
| **PPO** | 强制要求 log-prob，严重破坏 drifting 核心机制 |
| **GRPO** | 同样需要 log-prob，且仓库无 drifting 适配实现 |
| **纯 Pathwise AC** | 丢弃 V-field 语义，把 drifting 退化成普通 reparameterized policy |
| **任何依赖 offline rewards 的方案** | 当前数据资产不支持 |
| **SAC** | 熵正则化依赖 log-prob，与 drifting 不兼容 |

---

## 5. 仓库落地建议

### 5.1 应该优先复用的现有模块

| 模块 | 用途 | 修改程度 |
|------|------|---------|
| `model/drifting/ft_qguided/qguided_drifting.py` | Actor + Critic 模型 | 无需修改 |
| `model/drifting/drifting.py` | Drifting 核心 | 不修改 |
| `model/common/critic.py` | Critic 网络 | 无需修改 |
| `agent/finetune/drifting/train_qguided_drifting_agent.py` | 训练循环 | 需修改：添加 online-only 模式 |
| `agent/dataset/sequence.py` | 数据集加载 | 无需修改（online-only 不使用） |

### 5.2 需要修改的模块

**`agent/finetune/drifting/train_qguided_drifting_agent.py`**：
- 添加 `online_only` 模式：当离线数据无 rewards 时自动启用
- `__init__` 中 offline_dataset 改为可选
- `run()` 中跳过 `_cache_offline_dataset()` 当 online_only 时
- `_sample_training_batch()` 在 online_only 模式下返回纯在线数据
- 在 online_only 模式下，从第 0 次迭代就开始 rollout collection
- Critic warmup 阶段只更新 critic，actor 冻结

### 5.3 不要修改的核心文件

- `model/drifting/drifting.py` — drifting 核心算法
- `model/drifting/ft_qguided/qguided_drifting.py` — Q-guided 模型层
- `model/common/critic.py` — critic 网络

---

## 6. 最小验证计划

### 验证矩阵

| 阶段 | 验证内容 | 任务 | 优先级 |
|------|---------|------|--------|
| 1 | 单元测试：online-only 模式初始化 | N/A | P0 |
| 2 | 单元测试：_sample_training_batch 纯 online 路径 | N/A | P0 |
| 3 | 单元测试：现有 tests 回归不破坏 | N/A | P0 |
| 4 | 集成测试：在线 rollout + critic warmup + actor update | gym/hopper | P1 |
| 5 | 数值稳定性：Q 值、drift norm、actor loss 监控 | gym/hopper | P1 |
| 6 | 扩展验证：其他 gym 任务 | walker2d, ant | P2 |
| 7 | 扩展验证：RoboMimic image 任务 | lift, can | P2 |
| 8 | 回归测试：旧 offline-mixed 模式仍正常 | gym/hopper | P1 |

### 验证顺序

1. **先验证**：`gym/hopper-v2` — 最稳定、最快的 MuJoCo 任务
2. **再扩展**：`gym/walker2d-v2`, `gym/ant-v2`
3. **后验证**：`robomimic/lift`, `robomimic/can`（需要 image 数据）
4. **最后**：`kitchen-*` 任务

### 关键监控指标

- `critic/loss` 应稳步下降
- `critic/q1_mean` 应合理增长（不发散）
- `actor/V_norm_mean` 应保持非零
- `actor/query_q_mean` 应随训练提升
- `avg_episode_reward` 应优于纯预训练 baseline

### Pretraining vs Fine-tuning 数据需求

| 阶段 | 需要的数据 | 来源 |
|------|-----------|------|
| Pretraining | `states, actions, traj_lengths` | 离线 demonstration |
| Fine-tuning (critic) | `(s, a, r, s', done)` | 在线 replay buffer |
| Fine-tuning (actor) | `s` (observations only) | 在线 replay buffer |
