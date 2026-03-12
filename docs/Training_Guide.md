# 全流程运行与测试指南 (Comprehensive Execution Guide)

> 面向 DMPO 框架中离线预训练、模型评估以及在线强化学习微调的保姆级分步操作说明。

---

## 目录

1. [环境安装准备](#1-环境安装准备)
2. [使用 Drifting Policy 开展预训练](#2-使用-drifting-policy-开展预训练)
3. [模型评估 (离线与在线)](#3-模型评估)
4. [PPO 微调测试](#4-ppo-微调测试)
5. [GRPO 微调测试 (无Critic架构)](#5-grpo-微调测试)
6. [配置文件参考指南](#6-配置文件参考指南)
7. [排障指南](#7-排障指南)

---

## 1. 环境安装准备

### 1.1 安装依赖

```bash
# 克隆代码仓库
git clone https://github.com/xxx/DMPO.git
cd DMPO

# 安装基础包依赖
pip install -e .

# 针对 Gym/D4RL 任务基准
pip install -e ".[gym]"

# 针对 Franka Kitchen 机械臂厨房任务
pip install -e ".[kitchen]"

# 针对 RoboMimic 机器视觉操纵任务
pip install -e ".[robomimic]"

# 针对 D3IL 任务
pip install -e ".[d3il]"

# 针对 FurnitureBench 组装任务
pip install -e ".[furniture]"
```

### 1.2 设置环境变量

```bash
# 必填项：设置你的项目工作区根目录、数据存放目录和日志存放目录
export REINFLOW_DIR=/path/to/your/workspace
export REINFLOW_DATA_DIR=$REINFLOW_DIR/data
export REINFLOW_LOG_DIR=$REINFLOW_DIR/logs
```

### 1.3 下载数据集

主程序的启动脚本 (`script/run.py`) 会在需要时尝试从 Google Drive 自动下载缺失的数据集。你也可以选择手动下载存放：

```bash
# 数据集应被统一排布存放在 $REINFLOW_DATA_DIR/{环境系列}/{具体任务}/
# 举个例子: $REINFLOW_DATA_DIR/gym/hopper-medium-v2/train.npz
```

---

## 2. 使用 Drifting Policy 开展预训练

### 2.1 启动标准化 Drifting 预训练

```bash
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp
```

**发生了什么事情：**
1. 系统会试图向 `$REINFLOW_DATA_DIR/gym/hopper-medium-v2/` 路径加载专家演示数据集。
2. 内部构建出一个采取了 `MeanFlowMLP` 网络骨干结构作为底层的 `DriftingPolicy` 模型。
3. 应用基于漂移场建立的 Loss 开始进行约 `n_epochs=40` 轮数的专家数据拟合训练。
4. 将训练产生的 Checkpoints 周期性存储至 `$REINFLOW_LOG_DIR/` 下并同步接驳 WandB 记录实验看板。
5. （如果配置了）每经过预设步数设定的 `test_freq` 个训练 epoch 后会顺便挂载起一个真实的 MuJoCo 在线环境直接当场验算效果。

**核心需要留心调节的关键性超参如下表：**
```yaml
train:
  n_epochs: 40            # 要求训练经历总的总轮次代数
  batch_size: 128         # 一次投喂计算的批次大小
  learning_rate: 1e-3     # 网络更新的学习率
model:
  drift_coef: 0.1         # 向着正专家样本做漂移对齐的诱导引力强度
  neg_drift_coef: 0.05    # 向着反面样本做推离的斥力强度
  bandwidth: 1.0          # RBF 核函数的波段宽度范围设定
  mask_self: false         # 开启后计算中排异自身相互作用力干扰
```

### 2.2 附加弥散惩罚损失的 Drifting (Drifting with Dispersive Loss)

```bash
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_dispersive_mlp
```

增加这种变体参数会在优化函数后面附加额外挂靠上一条关于数据弥漫散开（dispersive regularization）的约束力，主要用来防控输出结果坍缩在一块化不开（防止模式坍塌 mode collapse 问题产生）。

### 2.3 当前支持开启离线预训练的全量配置任务表单

| 套件 | 任务系列 | 指定的启动配置项 |
|-------|------|--------|
| Gym | Hopper | `cfg/gym/pretrain/hopper-medium-v2/pre_drifting_mlp.yaml` |
| Gym | Walker2d | `cfg/gym/pretrain/walker2d-medium-v2/pre_drifting_mlp.yaml` |
| Gym | Ant | `cfg/gym/pretrain/ant-medium-expert-v0/pre_drifting_mlp.yaml` |
| Gym | Humanoid | `cfg/gym/pretrain/humanoid-medium-v3/pre_drifting_mlp.yaml` |
| Kitchen | Mixed | `cfg/gym/pretrain/kitchen-mixed-v0/pre_drifting_mlp.yaml` |
| Kitchen | Complete | `cfg/gym/pretrain/kitchen-complete-v0/pre_drifting_mlp.yaml` |
| Kitchen | Partial | `cfg/gym/pretrain/kitchen-partial-v0/pre_drifting_mlp.yaml` |

---

## 3. 模型评估

### 3.1 对基于状态量观测环境的模型启动评估 (State-Based)

```bash
python script/run.py \
  --config-dir=cfg/gym/eval/hopper-v2 \
  --config-name=eval_drifting_mlp \
  checkpoint_path=/path/to/checkpoint.pt
```

### 3.2 对包含视觉图象环境的模型启动评估 (Image-Based)

```bash
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_drifting_mlp_img \
  checkpoint_path=/path/to/checkpoint.pt
```

### 3.3 Evaluation Agent 在后台实际的运行规律

1. 寻找传入路径并加载那个历经预训练生成的神经网络层结 Checkpoint 权重。（同时允许指定要求启用 EMA 或者传统的正常标准权重模式直接做预测加载）
2. 初始化真实目标环境对象，不加修改运行其足额 `N` 次数回合 (episodes)。
3. 在过程中统计整理各种数据（包括计算取得的每回合积攒得分，最终通关的成功通过率，还有模型输出动作序列特质属性偏好统计）。
4. 在需要时开启记录并存成短视频。
5. 推送整理上述收集到的一些打点指标进入 WandB 面板去生成可视化图表。

**至关重要的一些用于修改的指标项：**
```yaml
denoising_steps: [1]              # 由于这里专门对口 Drifting 政策因此必然死绑被设为 1
load_ema: false                   # 是否加载历史 EMA 数据权重，如果为否将直取最后的参数计算
clip_intermediate_actions: true   # 是与否硬性裁切掉跑出了许可范围的激进的异常行为幅度数据
n_eval_episodes: 50               # 单次验算准备打够多少回合次数作为分子分母
```

---

## 4. PPO 微调测试

### 4.1 对于基础物理状态集启动基于 PPO 算法微调

```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_drifting_mlp \
  base_policy_path=/path/to/pretrained_checkpoint.pt
```

### 4.2 对于自带外界视觉采样的复合模型尝试 PPO 加持微调

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_drifting_mlp_img \
  base_policy_path=/path/to/pretrained_checkpoint.pt
```

### 4.3 PPO 阶段在内部的工作流程循环说明

在整个 PPO 接管周期中，代理将不断服从这个大循环处理过程：

```
每一个指定交互迭代大步（Iteration）之中包含下面三个重要环节节点：
  1. 收集交互数据(COLLECT): 于 `n_envs` 个同步拉起的并行进程中总进过消耗 `n_steps` 次步幅
     - 通过含有噪音扩展包外衣的代理 `NoisyDriftingMLP` (基准中心点 + 变差漂移点) 输出并采样得到一系列动作行为
     - 封存这个步中遭遇过的：观察环境状态、下发行动关联锁链信息项、所获取的外部真实奖励评分、和通关或失败标签 并汇入本地专属 PPO 记忆群内 (PPO buffer)
  
  2. 数学折算评估值(COMPUTE): GAE 全局经验折合测算与奖励贴现汇算
     - 调用强化广义评估优势计算公式 (Generalized Advantage Estimation) 并通常采取默认参数 γ=0.99, λ=0.95
     - 请求内网并行附加的价值评估模块（Value function）对后续期待作自举预估打分支持
  
  3. 执行回传修改参数(UPDATE): 正式拉动梯级斜率开始进行 PPO 大步前行升级
     - 从全表收集项中反抽出来开始重复刷够一众细分周转 epoch
     - 代入处理 PPO 引以为豪避免修正脱轨发生暴毙的 "截取式克隆剪裁目标差公式 (Clipped surrogate objective)"
     - 将原本独立出的附属那颗估算神经网络 (Value function loss) 拉伸匹配目标一致逼近
     - 添加一些散逸熵励机制参数 (Entropy bonus)
     - 提供并打开可选择开启的克隆保护惩罚约束力项 (BC regularization loss)
     - 采取对梯度的异常暴涨阶段实施断腕极值暴力剪切防护策略
     - 实现应对当两回合分歧跨度（KL值突变）一旦到达阈值时能够立即进行中断停止挽救（Adaptive KL-based early stopping）
```

### 4.4 极为值得调优的微调领域 PPO 专属高阶超参大全

```yaml
train:
  n_train_itr: 1000       # 指代训练要求的全局大循环步骤总计长度
  n_steps: 500             # 单进程子环境中每次能拿出来挥霍的走动跨度要求上限
  actor_lr: 3.0e-06        # 当前 Actor 执行主体脑的学习进阶速率值 (在此精调周期这个数不要设得过大)
  critic_lr: 4.5e-4        # 给旁边作为辅导点评师用的 Critic 脑放出的宽松高学习率
  gamma: 0.99              # 当下看未来的打折比例（贴现系数 Discount factor）
  gae_lambda: 0.95         # GAE公式自带平滑缓冲参数项
  batch_size: 50000         # 给 PPO 前向大批喂投吃入的数据捆块要求量
  update_epochs: 10         # 每集数据来回循环反刍的遍数指标
  vf_coef: 0.5             # 作为辅加损失的打分网系数权量占比
  ent_coef: 0.01            # 奖赏散化与探索行动鼓励加点项
  target_kl: 0.01           # 断舍离 KL 分辨差触发防暴毙安全底裤阀值点
  use_bc_loss: true         # 告诉算法是否要带上针对专家克隆痕迹不可偏离过猛的老玩家情怀锁定损失保护项
  bc_loss_coeff: 0.05       # 用力锁多死旧数据经验的影响程度

env:
  n_envs: 40               # 最核心的多进程开多少个平行虚拟世界跑
```

---

## 5. GRPO 微调测试 (无Critic架构)

### 5.1 呼出 GRPO 启动训练方式

```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_grpo_drifting_mlp \
  base_policy_path=/path/to/pretrained_checkpoint.pt
```

### 5.2 关于只属于 GRPO 才会特有的那套系统流程闭环流转说明

GRPO (组相对策略优化 Group Relative Policy Optimization) 一把扯掉了强化系统中作为负担和偏差引入根源的评论家网络 (critic network) 的伪装：

```
在每一场微调搏击大会的循环中：
  1. 并行跑路(COLLECT): 抓出同一批初始状态起点位相同的整组总计 G 条跑道轨迹数据
     - 遵循严格的 同源同根拉平启动位要求（Homogeneous reset）: 发令枪响时所有人出生数据绝对一模一样完全一致
     - 应用具备 (Tanh-Normal) 修正的 `NoisyDriftingPolicy` 在路上撒欢狂飙采样生成步子动作输出值
     - 定期一并连同 (记录目前状态信息, 行动出牌, 对数理论概论分, 刚跑下来的即时战果金币分) 入库备案
  
  2. 极性评断化解算处理(NORMALIZE): 通过算组内的表现高低定档优势数据
     - 抽出每组所有累加结算回报成绩汇总情况 R_g = Σ γ^t r_t 分在各大跑者 G 身上
     - 当场算出这组跑酷赛里相比较平均数水平的 极级标准偏差分项归一化评分 Advantage A_g = (R_g - mean(R)) / (std(R) + ε)
     - 添加关于全部死气沉沉得分表现一样时的绝对数值锁零断底机制 Zero-variance protection: 即但凡一旦存在 std(R) < 1e-6 弱智病症显现, 就全队死拉回零处理 A_g = 0 阻止数值起爆
  
  3. 大步伐修正动作发生(UPDATE): 根据上面打完战报回传调包拉进度拉拔提升更新网络参数
     - 截取修修补补助留防止更新狂化跑挂（依然挂载着类似于PPO原版使用的截取梯度修正 Surrogate loss）
     - 执行基于公式纯理论运算算盘推出来的 KL 分化衰颓计算刑罚惩处 (绝对严禁基于外带抽签估测造成的测不准抽样扰动偏差)
     - 推演打出关于 Beta 参数项缓慢柔性降级的曲线演变算法: β ← max(β * decay, β_min)
     - 不要去调那颗根本就不存在甚至被删了的 Critic 的损失算计，全本就是一场抛开所有估测全打实战表现的无批判式 (critic-free) 行动！！
```

### 5.3 属于 GRPO 自己的定制极简微调配表大放送

```yaml
train:
  group_size: 16            # 这就是一个小组赛里面能有几位同场同盘互相对照跑竞赛的坑位规模数量 (G)
  grpo_lr: 1e-5             # 针对单独干活的执行大脑网络放出的推进步幅大小参数
  update_epochs: 4           # 对于一批提取样本进行循环滚筒提练加工更新批次次数设定值
  grpo_batch_size: 256       # 最贴近底层的微缩打包塞入参数更新的计算块量级指标
  kl_beta: 0.05              # 本轮开场刚开始的时候在 KL 估差上的底线死守严打程度参数值
  kl_beta_min: 0.001         # 老油条后期也最少得坚持兜底遵守多少规模体量的最低 KL 扣分程度底线参数设置
  kl_beta_decay: 0.995       # 打完一个大周期大迭代以后针对上面那个 Beta 实行多少个巴仙点的折扣跳水收效运算参数值
  use_homogeneous_reset: true # 重点打勾是否对齐：强扭死绑所有兄弟起跑姿势与出身状态点绝对必须一比一严格靠齐还原拉准

model:
  epsilon: 0.2               # 把 PPO 这个借过来的老方剂用于夹断裁断狂飙幅度时设置允许的夹门两端许可缝隙余量区间大小配置
  beta: 0.05                  # 模型层面上原装指定的 KL 基础初始拉偏罚单值 (model-level)
```

### 5.4 快速比照速查：GRPO 大战 PPO (优劣差异总结盘点)

| 指标 | 传统 PPO 微调体系 | 新派极简组队 GRPO 搏击流体系 |
|---------|-----|------|
| 是否带有大累赘 Critic 辅助包袱网络层 | ✅ 必须有且不能剥离 | ❌ 什么？早被踢远了 |
| 利用什么东西作为全场跑过最终战绩与下个步数预测的好坏判定 | 挂个 GAE 并引入靠人喂的 bootstrap 计算预见价值进行主观估量 | 大家公平跑一场直接按照本场 Z-score (偏离大头平均数的好坏水平) 判定不凭想象打预判分 |
| 推偏背离 KL 的误差结算求断依据方式 | 随缘抽球大体约摸算出 | 给你张解析运算卷子真算得准准的交出 |
| 容身与处理动作表现的空间输出理论类型定性 | 就是直接套路普通经典高斯曲线形态 | 高斯被加装压入边界效应制成了切变模式下的 Tanh-Normal 演化扭曲新体态 |
| 边界失真带来的雅可比防反修正打补丁 (Jacobian correction) | — (没有这等高级病症要关注) | ✅ 这招专治因为夹了数据空间引发出来的对应失衡病灶 |
| 为了拉网络运行带来的额外重装备显存压力耗费量 | 超级大耗包 (吃干同时挤压带着 actor+critic 俩大哥的重量) | 就装了一个极度精简版仅仅用于实操出勤的最轻量的微端 Actor 脑 |
| 所产生的专家效能样本汲取收口使用率与进阶速度表现 | 榨取率惊人（更高） | 会慢上半拍（因为同一场测试起跑同一起源硬规定必然带来数据高度重复浪费的问题缺陷所在） |

---

## 6. 配置文件参考指南

### 6.1 一个配置核心文件的表述基本法

放眼过去的各种 YAML 基本长得都是由这几个特定结构区块硬生生拼出来的：

```yaml
# 上半层通常都是一些顶设总参环境
env_suite: gym                     # 套装类别名字
env: hopper-medium-v2              # 选用的那个数据库环境集的名字
action_dim: 3                      # 要控制打出来的出装行动参数有几条轴多宽尺寸
horizon_steps: 4                   # 计划生成预测未来长度块的时间线步伐跨度设置
obs_dim: 11                        # 传回观测点感知的宽维尺寸大小值
cond_steps: 1                      # 本次感知作为依赖往常历史时往回捞前面要记录回推多少个作为预判反馈提供参数

# 要反射找出来实例创建的中心智代（Agent）定位坐标点指路明灯位置（也就是 Hydra 的灵魂配置锚点所在位置）
_target_: agent.pretrain.train_drifting_agent.TrainDriftingAgent

# 有了这个目标位然后接下来就要对这个框架做全盘网络规模具体配置设定值填入表块（model篇）
model:
  _target_: model.drifting.drifting.DriftingPolicy
  # ... 以下省略一万字针对对应具体模型的深度指定细节参量配置说明...

# 进入专门为了炼制调整打造出来的心电配置项区块
train:
  # ... 略去详细到变态的各种花式训练手段细节开关表单...

# （如果目前只有跑离线集需要去装去套那么一定会配齐的专属 Dataset 配置专区表）
train_dataset:
  _target_: agent.dataset.sequence.StitchedSequenceDataset
  # ... 如何找如何喂投的表...

# (同样对于部分需要启动带残余保护手段指数衰减记录挂钩的专门预录版才有的项目参数区块组 EMA篇项)
ema:
  decay: 0.995
```

### 6.2 关于在运行时不改原表强制命令行改参的临时救火绝技运用方式

Hydra 自带了一股随时霸道覆盖掉预设文件内部定死内容的暴力参数洗刷劫持命令行覆盖机制：

```bash
# 遇到跑得很臭你想当场在打命令起动时候立刻把学习率从文件设改掉的时候
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp \
  train.learning_rate=5e-4

# 又或者你想调整某些特定的结构项网络参数指标也完全可以按照层级点分拨过去进行直接暴力强制接管
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp \
  model.drift_coef=0.2 \
  model.bandwidth=0.5
```

---

## 7. 排障指南

### 7.1 常规爆雷黑产地整理清单（常见问题自救表）

| 问题大类 | 是谁在作恶导致的锅？ | 你得这么干自救（Solution） |
|-------|-------|----------|
| `KeyError: 'REINFLOW_DIR'` | 环境变量大漏设忘设置或者挂掉没了 | 当场开终端老老实实补敲上一句 `export REINFLOW_DIR=/path/to/workspace` 给它指路 |
| 一运行在算 Loss 报错位丢出了 `NaN` 死亡红字代码炸裂崩溃 | 八成是跑的太激进造成步子扯蛋引起学习率拉得过高或者是那个漂偏移量诱导引力 `drift_coef` 拉偏力度太凶导致扯断撕裂计算网链了 | 切入配置将 `learning_rate` 和 `drift_coef` 参数狂降压低 |
| 直接崩出 OOM （把显卡的显存内存吃光挤爆满溢死机了） | 想想看肯定是图像输入的图片流块在进入进行 PPO 前台狂吞跑算的大步批次 `batch_size` 块塞得太大一卡子完全吃吃喝不下引发内爆了！ | 设法强行把梯度积攒功能外包缓冲保护开启动 `grad_accumulate`，更重要的是把巨无霸饭量 `batch_size` 扣着嗓喉硬给压缩切变小量装配 |
| 第一回合初始预判期算出一个离谱非正常预报值譬如说跑出来 Ratio 并不是死死咬在规矩应该落在 1.0 的附近的时候！ | 那绝对就是对数预判算流链条上写挂了某些数学理论或者是出岔出错的实现算式跑瓢了！ | 请严厉核查审查对数理论实现上在处理推演数据动作输出链路上的时空接轨链条尺寸并同时仔细确认有关噪音扩散演算法向和上限越位的溢出错误处理是否得当等算理异常处。 |
| 跑着跑着那个 KL 惩处断层鸿沟偏差（KL divergence）数值一路高歌猛涨不受收口打断狂奔冲天！ | 显然约束这头出轨猛兽的绳索勒的不够大力没挂足 Beta 罚金，还有那催跑狂奔学习步子的跨距拉得过头飞太脱线了引发了极大的漂移错觉！ | 赶紧在设置里加量放大狂给 `kl_beta` 数值并且反手直接削掉学习跨度率指标数以强勒这即将出栏崩塌的疯马。 |

### 7.2 关于为了修补定位查错而在测试环节采取的一些小秘方特技

```bash
# 加入此等尾缀后缀将把一切本想遮遮掩掩躲藏起来执行运行后台详情过程记录通通喷爆出来全部向外倒出呈现看戏（最适用于深度跟踪摸底报错线）
python script/run.py \
  --config-dir=cfg/gym/finetune/hopper-v2 \
  --config-name=ft_ppo_drifting_mlp \
  train.verbose=true

# 用于最极速检验调测一套走通系统大闭环机制流程行跑不行死没死而不必枯耗算力大半天才等出首发异常情况暴露的小步数冒烟测试缩简急跑法
python script/run.py \
  --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
  --config-name=pre_drifting_mlp \
  train.n_epochs=2 \
  train.test_freq=1
```

### 7.3 全方位安全测验试炼阵通过手段

```bash
# 全力召唤并且全部发动进行一切现存全部功能全方位综合打包安全通过检查连套！
python -m pytest tests/ -v

# 你也可以指名道姓拉个特殊组件或者一个极易引发血战暴死单薄单体弱鸡文件点名出面单独承受全面大质询检查：
python -m pytest tests/test_drifting_policy.py -v
python -m pytest tests/test_ppo_drifting.py -v
python -m pytest tests/test_grpo_drifting.py -v
python -m pytest tests/test_grpo_buffer.py -v
python -m pytest tests/test_end_to_end_smoke.py -v
python -m pytest tests/test_static_and_config.py -v
```
