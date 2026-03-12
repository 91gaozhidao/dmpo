# Drifting Policy — 开发者魔改指南 (Developer's Cookbook)

> 一份提供动手实践操作去修改、延展并排障 Drifting Policy 具体实现代码块落地的保姆级操作指南。

---

## 目录

1. [架构总览](#1-架构总览)
2. [如何修改预训练架构 (Pre-training)](#2-如何修改预训练架构)
3. [如何修改 PPO 微调架构 (PPO Fine-tuning)](#3-如何修改-ppo-微调架构)
4. [如何修改 GRPO 微调架构 (GRPO Fine-tuning)](#4-如何修改-grpo-微调架构)
5. [如何添加一个全新训练任务环境](#5-如何添加一个全新训练任务环境)
6. [关键常量定义与设定参数罗列](#6-关键常量定义与设定参数罗列)
7. [排障指南](#7-排障指南)

---

## 1. 架构总览

在目前结构体内 Drifting Policy 是建立在这种相互继承搭建的三层复合架构体系带上面的：

```
代码构架第 3 层: 智能系统包 Agent (负责整个外循环调度和挂载运作)
  ├── agent/pretrain/train_drifting_agent.py
  ├── agent/finetune/reinflow/train_ppo_drifting_agent.py
  ├── agent/finetune/reinflow/train_ppo_drifting_img_agent.py
  └── agent/finetune/grpo/train_grpo_drifting_agent.py

代码构架第 2 层: 模型包 Model (承担计算损失比对、产出分发动作集、和 RL 各类目标函体系工作)
  ├── model/drifting/drifting.py              # 核心的计算策略本身
  ├── model/drifting/ft_ppo/ppodrifting.py    # 为 PPO 设计的专用扩展马甲层
  └── model/drifting/ft_grpo/grpodrifting.py  # 为 GRPO 设计的专用扩展马甲层

代码构架第 1 层: 底层神经网络骨干包 Network (具体的堆叠网络结构件)
  └── model/flow/mlp_meanflow.py              # MeanFlowMLP 底层组件网络
```

---

## 2. 如何修改预训练架构 (Pre-training)

### 2.1 针对底层深度的网络外围结构的微调改动

**应当要去开刀的文件位置:** `model/flow/mlp_meanflow.py`

`MeanFlowMLP` 类直接定义了所有的脑补结构。你可以这样动手直接更改架构：

```python
# model/flow/mlp_meanflow.py
class MeanFlowMLP(torch.nn.Module):
    def __init__(self, action_dim, horizon_steps, cond_dim, 
                 mlp_dims=[512, 512, 512],      # ← 直接去改动加厚或削减这些神经元规模
                 activation_type="Mish",         # ← 在这换掉使用的激活面类型
                 ...):
```

甚至你懒得动代码的，可以直接在配置参数上实现完全越级指挥替代接管去换骨架设计：

```yaml
# 在你所使用的配置文件 YAML 里面大声喊出改写：
model:
  network:
    mlp_dims: [1024, 1024, 1024]   # 把网络参数扩建撑大
    activation_type: ReLU           # 直接换掉非线性激活算子
```

### 2.2 有关底层用来计算 “漂移拉扯引力场 (Drift Field)” 核心逻辑的魔改

**应当要去开刀的文件位置:** `model/drifting/drifting.py`， 找到这个方法块 `compute_V()`

这里就是负责判定每一次生成出来的游离量将会如何受到指引向专家示范数据集牵扯过去的力场规则：

```python
# 位图: model/drifting/drifting.py, 在系统 DriftingPolicy 这个类里面

def compute_V(self, x, y_pos, y_neg=None):
    """
    入参：
    x:     [B, T_a, D_a]  - 当下推算产出的预想出招动作组
    y_pos: [B, T_a, D_a]  - 用作正面表率去指引的 (专家数据) 标准答案
    y_neg: [B, T_a, D_a]  - 用作反面教材去推开避让的打标 (这项视情况为选用入参)
    
    想要大修该段魔改这套漂引核算机制？请按以下三个方向搞：
    1. 使用另一套新的距离算术标准（现在默认采用的是 L2 的算项范式）
    2. 使用一套全新的函数套核 (目前这里放的是纯正的 RBF / 高斯曲线函数)
    3. 改掉归一化放缩的处理分配制度逻辑
    """
    # 现状所挂载使用的机制: RBF kernel 核向
    diff = x.unsqueeze(1) - y_pos.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=(-1, -2))
    weights = torch.exp(-dist_sq / (2 * self.bandwidth**2))
    
    # 手把手替换演示范例: 如果你要把核函数生切替换变成用 Laplacian 核计算方式的话大概要长这样写：
    # dist = diff.abs().sum(dim=(-1, -2))
    # weights = torch.exp(-dist / self.bandwidth)
```

### 2.3 关于损失惩罚统计函数计算公式 (Loss Function) 的大挪移

**应当要去开刀的文件位置:** `model/drifting/drifting.py`, 找到这个方法 `loss()`

```python
# 位图: model/drifting/drifting.py, 这个结构依然附着在 DriftingPolicy 这个类名册下面

def loss(self, x1, cond):
    """
    要求入参：
    x1:   [B, T_a, D_a]  - 正式的专家操作数据 (也是我们向往追赶的标杆)
    cond: dict            - 收集传来的现状观察环境变量条件池
    
    目前的 Loss 情况是这么得来的: 就是将原本预判和上了修正方向漂引轨之后的两股流向量套在均方根误差体系里算出的出项数值。
    """
    # 动作发车起步段 1: 去采第一把白噪声出来
    x_gen = torch.randn_like(x1)
    
    # 直出前推传播走一遭 2: 算一波网络 (这部分属于 1-NFE 一步到位算出)
    x_pred = self.network(x_gen, t=ones, r=zeros, cond=cond)
    
    # 引力盘算结项点 3: 算出牵制引导力 V 的力向偏转分量
    V = self.compute_V(x_pred, y_pos=x1)
    
    # 最终打把靶心定标 4: 让原本的结果和这个拉扯力度合并造就一个最后想去向往的目标
    target = (x_pred + V).detach()
    
    # 打分误差比对项 5: 就是这里最终结账算损失！你想怎么加酷刑都从这动手
    loss = F.mse_loss(x_pred, target)
    
    # 乱入插手修改做个示范: 如果我想私自夹带施加一个带有 L1 正则化的手段惩罚力度上去应该怎么写：
    # loss = loss + 0.01 * x_pred.abs().mean()
    
    return loss
```

### 2.4 关于如何调教修改“防止模式拥堵坍缩的弥漫散花打压力” (Dispersive Regularization)

**应当要去开刀的文件位置:** `agent/pretrain/train_drifting_dispersive_agent.py`

在带有预防坍容机制变体代码中添加为了追求行为丰富散花状态下做出的附加性罚点规避惩治：

```python
# 想要进行彻底大换血重构弥漫手段的打法，请对这个底层进行强制覆写劫持 get_loss() 环节:
class TrainDriftingDispersiveAgent(TrainDriftingAgent):
    def get_loss(self, batch_data):
        actions, obs = batch_data
        # 这是调用基线原本正常产生的漂移计算结果 loss 总比分
        loss = self.model.loss(x1=actions, cond=obs)
        
        # 请在这里塞进你设计好了的一整套专为拉开打散状态自研附加打压手段...
        # 给出一个自己强贴个附加工资散溢熵激励项罚扣范例文：
        # with torch.no_grad():
        #     x0 = torch.randn_like(actions)
        #     pred = self.model.network(x0, t=1, r=0, obs)
        #     entropy = -torch.mean(pred ** 2)
        # loss = loss - 0.01 * entropy
        
        return loss
```

---

## 3. 如何修改 PPO 微调架构 (PPO Fine-tuning)

### 3.1 这里是教你怎么调教替换带有探索性开荒功能的“随机撒娇干扰性白噪音”部件 (Exploration Noise)

**应当要去开刀的文件位置:** `model/drifting/ft_ppo/ppodrifting.py`, 直接找叫这个名字 `NoisyDriftingMLP` 的组件

```python
# 位图查岗: model/drifting/ft_ppo/ppodrifting.py

class NoisyDriftingMLP(torch.nn.Module):
    def forward(self, x, cond, ...):
        # 步骤点 1: 先找本体大哥那个不带虚头巴脑绝对理性计算的那个直接解（基点）
        mean = self.policy(x, t=1.0, r=0.0, cond=cond)
        
        # 步骤点 2: 这就是根据过往教训学习总结后自主给出的浮动抖动震颤极差估计幅度 (就在这替换生成这种迷幻漂散干扰的源头方案)
        log_var = self.MLP_logvar(mean)  # ← 直接换掉这种搞法发源中心脑
        std = torch.exp(0.5 * log_var).clamp(min_std, max_std)
        
        # 步骤点 3: 将上述虚无缥缈附带着随机撒欢情绪的波动掺杂进来真正去进行最终决策产生变异动作
        noise = torch.randn_like(mean)
        action = mean + std * noise
        
        # 私改范例演练: 不搞什么劳什子普通高斯抽样了，直接去换带有连续游走性质的 OU 噪音 (OU noise) 方案上来干掉经典派！
        # if hasattr(self, 'ou_state'):
        #     self.ou_state = 0.85 * self.ou_state + 0.15 * noise
        #     action = mean + std * self.ou_state
```

### 3.2 关于修改计算并求导“对数理论概率方程评价体系” (Log-Probability) 参数核心的过程

**应当要去开刀的文件位置:** `model/drifting/ft_ppo/ppodrifting.py`, 找准名为 `PPODrifting` 类的对象

整个强化学习最看重它计算反馈能算出怎样精确靠谱的对数值率。而在 1-NFE 步频环境内的 Drifting 面临的问题就是非常单纯的一次高斯抽算展开而已了：

```python
# 找家串门路径图: model/drifting/ft_ppo/ppodrifting.py, 类位落在 PPODrifting 组件处

def get_logprobs(self, cond, chains, ...):
    """
    传进来的长锁链串 chains 字段: 尺寸在 [B, 2, T_a, D_a] 这般上下  - 在 Drifting 设置下死规矩永永远远恒为一个为 2 的短链项 (即噪音输入点 + 动作产生落点)
    
    系统内核运算对数概率的步骤法理:
    1. 分离链表截取头（抽样引入噪音项源头 z ） 跟尾巴（最终拍板砸下的动作执行指令输出端 a ）
    2. 对 z 这个初始虚空混沌噪音投入进那条算力隧道 `mean = network(z, t=1, r=0, cond)` 去压出实际的推演意图本体点
    3. 调用伴飞辅助干扰计算网络一并算出来当前允许其进行偏离波动的那个标准公差方差界域值
    4. 对着实际打出来的动作落点，拿高斯常规函数公式去框出这把它落在哪个概率上：log_prob = Normal(mean, std).log_prob(a).sum()
    """
    z = chains[:, 0]  # 源端白噪声混沌端
    a = chains[:, 1]  # 落地执行出的动作行为端
    
    # 调教传导过具有抗干扰浮动变差评估包裹网络组件
    mean, std = self.actor_ft(z, cond, ...)
    
    # 标准普通版高斯散列概率算法打出该有的日志判定
    dist = Normal(mean, std)
    log_prob = dist.log_prob(a).sum(dim=(-1, -2))
    
    # 夹带干私活改造演练时间: 为了能够适应那些边界限制被机械硬件卡死受限框出的运行场地强硬要加上关于极地失序的雅可比修正力补偿（Jacobian correction）做法的话：
    # if self.use_tanh_squashing:
    #     log_prob -= (2 * (math.log(2) - a - F.softplus(-2*a))).sum(dim=(-1,-2))
```

### 3.3 魔改强退关于 PPO 后端裁切组装运算的种种 Loss 计算惩罚规矩 (PPO Loss Components)

**应当要去开刀的文件位置:** `model/flow/ft_ppo/ppoflow.py`, 就是这个名为 `loss()` 负责操刀裁决最终成败算总帐环节

原本作为一切强化流计算宗家的 `PPOFlow` (也就是你家 `PPODrifting` 的亲长辈老爹) 给我们留下了这样合成分子的总损失方案配比公式：

```python
# 全部累计结账大盘口 = pg_loss (主力网络分) + ent_coef * entropy_loss (散列罚薪) + vf_coef * v_loss (指导教练评分补偿) + bc_coeff * bc_loss (克隆老东家传统动作惩罚加项)

# 假设这里咱看不惯那个专收智商税强逼去模拟克隆老玩家旧动作防止变调的 (BC loss / behavioral cloning regularization) 老八股规矩非得进行强行改派:
# 具体定点位置: model/flow/ft_ppo/ppoflow.py, 去搜第差不多大概行数 500 行的这处前后方位节点：
if bc_loss_type == 'W2':
    bc_loss = wasserstein_distance(...)
elif bc_loss_type == 'velocity':
    bc_loss = velocity_prediction_loss(...)
# 如果老子非要在这里开辟一条属于我自己独特信仰宗门计算出来的 BC 加点折损体系的规矩那就直接往这上硬插代码写出算例：
# elif bc_loss_type == 'my_custom_bc':
#     bc_loss = my_custom_function(...)
```

### 3.4 想彻底大改重写属于你自己的那套私密 PPO 代理特聘教官推演套环法则

**应当要去开刀的文件位置:** `agent/finetune/reinflow/train_ppo_drifting_agent.py`

如果处理含有只吃单一基本环境变量请用上面那个位置，若是要挂满高清摄像头处理带有三维实物成像的话直接去找这兄弟 `agent/finetune/reinflow/train_ppo_drifting_img_agent.py` 文件下刀改写。

去这几个核心关键方法下猛注去进行暴力覆盖掉重做流程：
- `agent_update()` — 这正是掌管了如何去压着它算出并强推上去的那个 PPO 最核心拉陡坡下刀升级的执行更新推进端算式环节
- `run()` — 去把持着掌控一切推进时间大轮盘运行和转动大周期主干
- `__init__()` — 系统刚下地接生落地起航前的初始化建构、大参数强行接盘重分配安放过程的开荒期阶段

---

## 4. 如何修改 GRPO 微调架构 (GRPO Fine-tuning)

### 4.1 如何直接换成别家流派做变频动作发生分布的动作空间产出机制法门

**应当要去开刀的文件位置:** `model/drifting/ft_grpo/grpodrifting.py`, 针对类结构 `NoisyDriftingPolicy`

```python
# 位图查岗: model/drifting/ft_grpo/grpodrifting.py

class NoisyDriftingPolicy(torch.nn.Module):
    def get_log_prob(self, cond, action):
        """
        按照 Tanh-Normal 的变形双切框约束的边界控制体系输出进行推导概略评估判定。
        
        如果要换一条出路，想换成他种推算法门思路该怎么办呢：
        1. 首先去大修并接管负责打回分派输出那个 distribution 机制的底层方法源头 `get_distribution()` 将其切成你的定制品类
        2. 若换掉了基础概率分部面那么一定务必记得同步把因为拉拔极点扭曲畸变而专配的对口那个雅可比修复纠正机制（Jacobian correction）算法补丁一并重构掉
        """
        mean, std = self.forward(cond)
        dist = Normal(mean, std)
        
        # 挂上倒挡逆向拉掉 tanh 的极强力挤压框定把数据逼出在变形受到强控挤迫之前原本真正长成什么样子的本体原型
        u = torch.atanh(action.clamp(-TANH_CLIP_THRESHOLD, TANH_CLIP_THRESHOLD))
        
        # 再代入经过解套压迫的本体值连同配套修正偏斜错谬雅可比打补丁法则一并合算出概率比分值
        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= _tanh_jacobian_correction(u).sum(dim=-1)  # ← 极为要紧的核心偏差回拉修补手术执行位置
        
        return log_prob
```

### 4.2 对于防止脱轨散开防漏用的 KL Divergence 计算惩击评估标准去修改的办法

**应当要去开刀的文件位置:** `model/drifting/ft_grpo/grpodrifting.py`, 就叫这个 `GRPODrifting` 组件

```python
# 位图走你: model/drifting/ft_grpo/grpodrifting.py, 类名叫 GRPODrifting

def compute_loss(self, obs, actions, advantages, old_log_probs):
    # 现状保留用法: 因为现在使用全高斯演变的纯天然解开方式（且是在未引入压边操作约束切入的预演展区划空阶段开算的： pre-tanh space）
    # 大概长这样推导法则：KL(N(mu_curr, sigma_curr) || N(mu_ref, sigma_ref))
    kl = (
        torch.log(sigma_ref / sigma_curr)
        + (sigma_curr**2 + (mu_curr - mu_ref)**2) / (2 * sigma_ref**2)
        - 0.5
    ).sum(dim=-1).mean()
    
    # 假如有特殊古怪爱好你非不想用理论上算的这种解析直接解公式，就是想要退环境回去使用拿去实测采点乱蒙碰运气乱抽对对碰那种带有震荡偏性测法（由 logprob 直接拉相减）：
    # kl = (curr_log_prob - ref_log_prob).mean()
    
    # 或者说反向思维去跑偏挂载完全倒向反过来的那个叫倒打求参逆向评估手法（reverse KL）：
    # kl_reverse = (
    #     torch.log(sigma_curr / sigma_ref)
    #     + (sigma_ref**2 + (mu_ref - mu_curr)**2) / (2 * sigma_curr**2)
    #     - 0.5
    # ).sum(dim=-1).mean()
```

### 4.3 魔改群组相对优胜劣汰战况排位结算归并（Advantage Computation）体系

**应当要去开刀的文件位置:** `agent/finetune/grpo/buffer.py`, 这个数据中存仓名号 `GRPOBuffer` 就是了

```python
# 位图飞点: agent/finetune/grpo/buffer.py

ADVANTAGE_STD_THRESHOLD = 1e-6  # ← 这是那条全挂病死强拉救命兜底零偏差防阻断归零下限

class GRPOBuffer:
    def normalize_advantages(self):
        """
        目前的标准现状是: 在一堆跑同一起跑线的组里面去实行 Z-score (零和打平分向出格值计算算评排资)
        简单口诀优势算分法则 = (自家打回来的猎物量 - 本组所有打来的战利品平均分给线分) / (去分摊折算掉方差 + 防爆炸无限小保护数 eps)
        
        如果你有别的神仙结算战绩方式可以在这改掉它算法:
        """
        mean = self.returns.mean()
        std = self.returns.std(unbiased=False)  # 基于这帮家伙的集体总体统估盘算出总体标准公差去挂比项
        
        if std < ADVANTAGE_STD_THRESHOLD:
            # 数据彻底死掉归全归零防御强控触发警报：发现所有人这一路打下来打出来的奖金评分死鱼一样居然一点上下区别分叉都没有全是同样的一个呆比分数值！
            self.advantages = torch.zeros_like(self.returns)
        else:
            self.advantages = (self.returns - mean) / (std + ADVANTAGE_STD_THRESHOLD)
        
        # 魔改小举例: 改掉按照大锅平均切的方法，而采用按照四分位数等级切割出表现极优等和最糟等的百分比例落差做分水岭计算（Percentile-based normalization）:
        # median = self.returns.median()
        # iqr = self.returns.quantile(0.75) - self.returns.quantile(0.25)
        # self.advantages = (self.returns - median) / (iqr + eps)
```

### 4.4 有关 KL 的松绑宽宥罚数项（Beta）随着训练日久放宽衰减进度条的调整（Beta Decay Schedule）

**应当要去开刀的文件位置:** `agent/finetune/grpo/train_grpo_drifting_agent.py`

```python
# 找到这段去强行下手重编: agent/finetune/grpo/train_grpo_drifting_agent.py

def update_beta(self):
    """
    当下的做派是: 使用随进度呈比例递减乘数连乘一直挂底线的（自然指数型衰变减让法）
    其规则：每次下一场演化后的新 beta 指标口 = 提取比较出两个里面最大的值(拿原本 beta * 每回合去打骨折折损值, 和老祖宗留下底线兜底不许让步跌穿的最惨保留额面线 beta_min) 
    """
    self.beta = max(self.beta * self.beta_decay, self.beta_min)
    
    # 变脸改造套近乎展示: 我们换种温柔曲线救国的走势（拉上带使用类似于平缓坡度曲线做滑行余弦落下的降解方法 Cosine annealing）来调教！
    # progress = self.itr / self.n_train_itr
    # self.beta = self.beta_min + 0.5 * (self.beta_init - self.beta_min) * (1 + math.cos(math.pi * progress))
```

---

## 5. 如何添加一个全新训练任务环境

### 5.1 自己建一套全新的适配的配置文件群

去找个合理地段连造下面这种连体套房建制 3 份出来，各就各位布好局：

```bash
cfg/{你搞的这个环境主题名_env_suite}/pretrain/{这特定小关卡任务名_task}/pre_drifting_mlp.yaml
cfg/{你搞的这个环境主题名_env_suite}/finetune/{这特定小关卡任务名_task}/ft_ppo_drifting_mlp.yaml
cfg/{你搞的这个环境主题名_env_suite}/finetune/{这特定小关卡任务名_task}/ft_grpo_drifting_mlp.yaml
cfg/{你搞的这个环境主题名_env_suite}/eval/{这特定小关卡任务名_task}/eval_drifting_mlp.yaml
```

### 5.2 全新的关卡大冒险建立配置起步模板库抄底样章演示

```yaml
# 第一站先把属于你任务离线打点训练关卡的这个配置架设好样例文件在: cfg/my_suite/pretrain/my_task/pre_drifting_mlp.yaml
env_suite: my_suite
env: my_task
action_dim: 6                  # ← 填入你的设备机体动作发号施令控制一共有几个多大跨宽度的轴参数量
horizon_steps: 4               # ← 让它单次抛算筹码切出多大长短的时序切块
obs_dim: 20                    # ← 确认自己设备本体上面一共有多少项传感器知觉传入数据条度
cond_steps: 1

_target_: agent.pretrain.train_drifting_agent.TrainDriftingAgent

train:
  n_epochs: 40
  batch_size: 128
  learning_rate: 1e-3

model:
  _target_: model.drifting.drifting.DriftingPolicy
  network:
    _target_: model.flow.mlp_meanflow.MeanFlowMLP
    action_dim: ${action_dim}
    horizon_steps: ${horizon_steps}
    cond_dim: ${obs_dim}
    mlp_dims: [512, 512, 512]
    activation_type: Mish
  act_min: -1
  act_max: 1
  max_denoising_steps: 1
  drift_coef: 0.1
  neg_drift_coef: 0.05
  mask_self: false
  bandwidth: 1.0

train_dataset:
  _target_: agent.dataset.sequence.StitchedSequenceDataset
  dataset_path: ${oc.env:REINFLOW_DATA_DIR}/my_suite/my_task/train.npz
  horizon_steps: ${horizon_steps}
  cond_steps: ${cond_steps}

ema:
  decay: 0.995

test_in_mujoco: false
```

---

## 6. 关键常量定义与设定参数罗列

### 6.1 专属针对 Drifting Policy 的超管级配置常量大集结

| 叫什么常量名 | 惯常预制初始值 | 它到底埋伏在哪个山头文件里 | 它到底是干什么活办什么事的 |
|----------|-------|------|---------|
| `drift_coef` | 0.1 | `model/drifting/drifting.py` | 管向着正面表率靠拢靠齐时候打牵引加力拉过去的吸力强硬度 |
| `neg_drift_coef` | 0.05 | `model/drifting/drifting.py` | 负责去对那个指定成要反开逆方向当排挤异端要排斥弹飞开时的避让强度弹走力量 |
| `bandwidth` | 1.0 | `model/drifting/drifting.py` | 分派影响范围去套打 RBF 核心圈内发作生效范围的有效宽泛带 |

### 6.2 专属针对极简流 GRPO 专供配套常数配置合集

| 名号参数 | 给的什么数值 | 所在具体窝藏点位置在哪里 | 用了干啥勾当 |
|----------|-------|------|---------|
| `LOG_2` | `ln(2)` | `model/drifting/ft_grpo/grpodrifting.py` | 辅助作为 Tanh 雅可比变形挤压公式里运算需要的一把必切大刀钥匙 |
| `TANH_CLIP_THRESHOLD` | 0.999999 | `model/drifting/ft_grpo/grpodrifting.py` | 万万不可顶出头极变，防强行极限切除出乱码的护体保底稳定值 |
| `JACOBIAN_EPS` | 1e-6 | `model/drifting/ft_grpo/grpodrifting.py` | 处理带有超界溢流强拉算式中遇到除外为零这种爆破崩溃事端的底托分母填坑保护盾除险常量 |
| `ADVANTAGE_STD_THRESHOLD` | 1e-6 | `agent/finetune/grpo/buffer.py` | 为了专门防止组队打到所有一分未差的同局而生造导致计算归项全崩特设的这颗用于斩灭计算方差起爆可能发生防归零爆表的安全守门垫脚石常量 |

### 6.3 提供给正统一门 PPO 的保留常规常数列装

| 参数字眼 | 常见预付给的初始值 | 现在安详睡在哪个核心文件内部 |
|-----------|--------------|------|
| `min_sampling_denoising_std` | 多处随机常有变动 | `model/drifting/ft_ppo/ppodrifting.py` |
| `min_logprob_denoising_std` | 也是在很多变 | `model/drifting/ft_ppo/ppodrifting.py` |
| `inital_noise_scheduler_type` | 各不常一 | `model/drifting/ft_ppo/ppodrifting.py` |

> **提示排坑特别留言:** 关于参数词条名字为 `inital_noise_scheduler_type` (少了个 'i') 没错就是人家当时第一代前人留下来拼写刻意拼错写歪导致的字，但由于目前代码要保持对于上级祖宗辈所有的流 PPO 各模型家族都要统一相兼容调用向前提齐，所以不准把错字拼对强行要沿用以前定死的留个坑一直给带着这个错误拼字使用配置。

---

## 7. 排障指南

### 7.1 在 Loss 输出位上面狂奔喷发丢出 `NaN` 的诡谲惨剧

**暴毙症状特点表现:** 直接就是在狂跑训练计算进度中突发断崖直接崩满屏一地打出计算值无效失传归途的 `NaN`。

**要冲去核查拔查炸病根的地方有哪些：**
1. 跑去找: `model/drifting/drifting.py:compute_V()` — 这往往说明两边对冲数据距离太大偏了很远已经把 RBF 体系权重连乘撑大扯断运算越界导致算裂开。
2. 跑去查验: `model/drifting/ft_grpo/grpodrifting.py:_tanh_jacobian_correction()` — 说明肯定有极其夸张畸变的非正常数据越过界顶穿在极限位边线上直接造成强行剪切打变形修正把数值顶废破功失控了。
3. 把着看: `agent/finetune/grpo/buffer.py:normalize_advantages()` — 分母垫背太薄不幸撞上了大家得了一样的烂分数遭遇上了一个除于了无穷下限引发死算归零异常故障。

**该怎么用雷霆手段火线救急:** 直接硬切强开上锁夹断梯次数值切削上限功能（梯度截断 gradient clipping），又或者干练下发降低猛学猛看的那个极高猛进学习效率拉回至底数 `learning_rate` ，还不行就尝试拓宽撒网覆盖容受限度放大那个有效宽泛波段带宽范围带 `bandwidth` 上。

### 7.2 遇到了在开场的第一天比分就出现了不合理非 1.0 开始预留分岔位 Ratio

**发病迹象外征:** PPO 开始计算并展示的那第一圈预演汇报 `update_epoch=0, batch_id=0` 比分拉不开比对的时刻这个本应等价抵消的 ratio 完全没落在 1.0 的老点位处挂上钩。

**你被大代码背刺中坑了必须马上纠错处理的信号** 这里代表了完全关于怎么弄算出对数几率那一块绝对写错了在胡编乱造算法！要明白一件最起底的规矩在这第一次下第一只手推网络升级进行任何一点梯次网络步幅变动前的新旧分是铁板一样死死钉死的绝绝对对必然一样等比例重同数字呈现。

**立刻火线前往勘探纠除大病区:** `model/drifting/ft_ppo/ppodrifting.py:get_logprobs()` — 务必死盯打头看看对于动作解算那两套链路抽包索取环节下标有没有指歪瞎取到不存在或非预定空缺层以及在对待推演搞那个附加随机浮标生成这套流程有没有做到严格按照既往预制轨迹复现稳在实控手里而非彻底撒欢随机乱点谱子不可追迹。

### 7.3 在执行 GRPO 的情况下突然惊现 KL 误差猛然崩裂狂飘不受制狂涨爆炸案发

**突发病症危疾的特点:** KL 那个预测的惩罚数值没边没际狂顶冲高无能管住，代表着此时在受测跑出圈子的这个家伙表现出的离脱叛道作死狂奔离专家或者基线越来越远一溜走失追不回来！

**强压控制治乱手段施为:**
1. 直接用重典拉死扣强上加筹加大在惩治散逸出逃上不宽贷那个严管把牢惩戒尺度 `kl_beta` (加上个越发死紧的枷锁控制住他乱奔逃向) 
2. 放慢减停压低目前这种疯狂不羁过拉大高步频狂进极大的猛进试训步调控制好小步走率 `grpo_lr` (让其稍微压慢改变脱出方向的变质节奏)
3. 开绝招一劳永逸引入当场达到预警戒尺指标阀值马上自动刹车停止的防御护城功能：
   ```yaml
   train:
     target_kl: 0.01
     lr_schedule: adaptive_kl
   ```

### 7.4 面向平时大板观测各种表针数值查验监控预演判断是否处于稳定安康安全指标预判指南一览：

重点需要你去盯着在 WandB 开着大盘去看的这几项关键体征指征：

| 名字和表项称号 | 老老实实本本分分的安全舒适生存域区应该稳在哪 | 快要濒发心绞痛死病问题发作时它长成什么个样子 |
|--------|---------------|------------|
| `noise_std` | 0.01–1.0 范围之内 | 当缩在这几乎快躺底向零位看齐快没了 (彻底完全在不思进取放弃乱串了无路可出停止尝试) 或者反向一路长高越界大于长出 >5 甚至更多了以后往天上乱冲 (根本失控成疯乱飘一弃崩掉状态下) |
| `approx_kl` | 0.001–0.05 之间晃荡游移 | 但凡突然爆长出去狂飞超离 >0.1 之上了甚至更多的情况发生的话 (那完了这个行为偏离和原本的老根已经越划越跑断远偏离拉脱逃不可收敛离彻底失败报销差不远了) |
| `clipfrac` | 控制处在大概只占 0.05–0.3 的规模分度比例 | 如果超出了 >0.5 或者狂涨的话 (证明全靠这个最后的一刀切防护盾天天吃着在顶缸狂死咬防被拉崩剪掉实在发生得特么太经常了) |
| `ratio` | 贴稳靠在差不多 0.8–1.2 之间起落打来回浮动震荡状态 | 下去降落贴底线狂跌或者跌越出了 <0.5 的底盘抑或高涨破顶突破压过 >2.0 的这极度非稳定抽风乱抽颠簸病象 |
| `entropy_loss` | 没有定轨，各随其家自己去定义走势变化差异 | 如果看到这根曲线呈现极度反平毫无作为而且一直只是单吊直下死水微滑一去不退地进行单纯直线下落不转圈 → 直接就给宣判预判死局由于还没怎么学会啥东西就发生这种极端急收场过早丧失可能性的退局判定 (早夭衰亡提前打卡放弃了自己探索欲表现出的躺平收敛) |
