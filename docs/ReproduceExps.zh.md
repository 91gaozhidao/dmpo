# 实验复现友好指南

欢迎！本指南将引导您完成实验的设置与运行。

指南分为清晰的几个步骤：获取数据集、下载检查点、运行实验以及一些实用提示。让我们尽可能让这个过程变得顺畅。


## 1. 获取并准备预训练数据集
**我应该阅读这一部分吗？**
* 如果您希望复现我们的预训练结果，或者训练 FQL 等离线 RL 智能体，请阅读此部分并先下载预训练数据。
* 如果您想添加自己的环境，从头开始训练模型，然后进行微调，此部分对于理解本仓库也很有帮助。
* 如果您只想使用在线 RL 方法微调我们预训练好的检查点，可以放心跳过此部分。

### OpenAI Gym: 基于状态的运动数据集

对于 OpenAI Gym，您有两个数据集选项：D4RL 版本（我们的首选）和 DPPO 版本（较旧且不推荐）。以下是处理方法：

#### D4RL 版本 (推荐)

- **选项 1: 自动获取处理后的数据**
  - 在预训练命令中使用 `use_d4rl_dataset=True`，即可轻松下载 `train.npz` 和 `normalization.npz` 文件。
  - `walker2d-medium-v2` 示例：
    ```bash
    python script/run.py --config-dir=cfg/gym/pretrain/walker2d-medium-v2 --config-name=pre_reflow_mlp
    ```
  - 更多详情请参见第 3.1 节。

- **选项 2: 手动处理数据**
  - 此方法仅适用于 `hopper`、`walker2d` 和 `ant`。
  - 步骤：
    1. 从 Hugging Face 下载原始 `.hdf5` 文件：
       ```bash
       wget https://huggingface.co/datasets/imone/D4RL/resolve/main/hopper_medium-v2.hdf5
       wget https://huggingface.co/datasets/imone/D4RL/resolve/main/walker2d_medium-v2.hdf5 
       wget https://huggingface.co/datasets/imone/D4RL/resolve/main/ant_medium_expert-v2.hdf5
       ```
    2. 查看 `.hdf5` 文件内部：
       ```bash
       # 检查 .hdf5 文件的数据结构
       python data_process/read_hdf5.py --file_path=<PATH_TO_YOUR_HDF5> 
       ```
    3. 转换为 `.npz` 并归一化专家演示：
       ```bash
       # 将 .hdf5 转换为 .npz 并将演示缩放到 [-1,1]
       python data_process/hdf5_to_npz.py --data_path=<PATH_TO_YOUR_HDF5>  
       # 在同一文件夹下输出 description.log 和 normalization.npz
       # 可选：调整 --max_episodes（默认：全部）和 --val_split（默认：0.0）
       ```
    4. 检查新的 `.npz` 文件：
       ```bash
       # 探索处理后的 .npz 或阅读 description.log
       python data_process/read_npz.py --data_path=normalization.npz
       ```
    5. 将文件移动到 `/data/gym/<TASK_NAME>`。

#### DPPO 版本 (不推荐)

- 使用 `use_d4rl_dataset=False` 自动下载，但为了保持一致性，我们建议坚持使用 D4RL。
- 示例：
  ```bash
  python script/run.py --config-dir=cfg/gym/pretrain/walker2d-medium-v2 --config-name=pre_reflow_mlp device=cuda:0 sim_device=cuda:0
  ```
- **提醒：** 
  - 仅在需要完全匹配 DPPO 的结果时使用该数据集。
  - 为了与基于 D4RL 的更广泛研究保持兼容，请使用 D4RL 版本。
  - 如果两者都用，请分开放置，并在命令中更新 `train_dataset_path` 和 `normalization_path`。

### Franka Kitchen: 基于状态的多任务操作数据集

- 通过 DPPO 的方法使用 D4RL 的数据集（完整、混合或部分观测）。
- 运行预训练命令，数据会自动下载并归一化——非常简单！详见第 3.1 节。

### Robomimic: 基于像素的操作数据集

- 依赖于 DPPO 简化后的 Robomimic 像素数据集。
- 预训练命令会自动处理下载和归一化。详见第 3.1 节。
- **注意：** DPPO 的数据比官方 Robomimic 数据更简单、更小。在官方数据上训练需要更大的模型和更多时间，但会提升预训练的成功率。

---

## 2. 下载预训练检查点

只需运行微调脚本，检查点就会自动下载。


## 3. 运行实验
由于我们支持的实验和算法非常多，本文档不会记录所有可能的命令。我们仅展示一些示例，其他设置遵循我们列出的命令模式。

### 3.1 预训练

预训练使用离线数据集训练策略，这些检查点将用于微调。
如果您想使用我们预训练好的检查点，可以跳过此部分，但您也可以按照我们的指南训练自己的模型。
以下是某些环境在不同设置下的运作方式。您可以根据需要随时更改模型类、环境和任务名称。

#### OpenAI Gym

- **在 `walker2d` 中预训练 DDPM 策略：**
  ```bash
  # 为 walker2d 训练 DDPM 策略，支持 GPU 并在过程中定期测试
  python script/run.py --config-dir=cfg/gym/pretrain/walker2d-medium-v2 --config-name=pre_diffusion_mlp device=cuda:0 sim_device=cuda:0 test_in_mujoco=True
  # `device`: 用于计算的 GPU；`sim_device`: 用于渲染的 GPU（如果没有 EGL 支持则设为 null）
  # `test_in_mujoco=True`: 每隔 `test_freq` 步测试一次策略
  ```

- **离线预训练 `Humanoid-v3` 的 1-ReFlow 策略：**
  ```bash
  # 离线训练 1-ReFlow 策略——非常适合网络不稳定的情况
  python script/run.py --config-dir=cfg/gym/pretrain/humanoid-medium-v3 --config-name=pre_reflow_mlp wandb.offline_mode=True
  # `wandb.offline_mode=True`: 如果无法在线同步，则将日志保持在本地
  ```

- **在 `Robomimic square` 中预训练 Shortcut 策略：**
  ```bash
  # 训练 Shortcut 策略，去噪步数高达 20（使用 2 的幂：1, 2, 4, 8, 16）
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_mlp denoising_steps=20
  # `denoising_steps`: 蒸馏的最大步数；使用小于该值的 2 的幂
  ```

#### Franka Kitchen 和 Robomimic

- 命令与 Gym 类似，只需更换配置路径：
  - Franka Kitchen: `--config-dir=cfg/gym/pretrain/kitchen-mixed-v0`
  - Robomimic: `--config-dir=cfg/robomimic/pretrain/transport`
- 更新 `--config-name` 以匹配这些目录中的配置文件名。

### 3.2 微调

微调通过在线 RL 改进预训练好的策略。参考以下示例。

- **使用 DPPO 在 Franka Kitchen 中微调 DDPM 策略：**
  ```bash
  python script/run.py --config-dir=cfg/gym/finetune/kitchen-partial-v0 --config-name=ft_ppo_diffusion_mlp seed=3407
  ```

- **使用 DPPO 在 Robomimic 中微调视觉输入 DDIM 策略：**
  ```bash
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_diffusion_mlp_img
  ```

- **使用 ReinFlow 在 OpenAI Gym 中微调 1-ReFlow 策略：**
  ```bash
  python script/run.py --config-dir=cfg/gym/finetune/ant-v2 --config-name=ft_ppo_reflow_mlp_img min_std=0.08 max_std=0.16 train.ent_coef=0.03 
  ```

- **使用 ReinFlow 在 Robomimic 中微调视觉输入 Shortcut 策略：**
  ```bash
  python script/run.py --config-dir=cfg/gym/finetune/ant-v2 --config-name=ft_ppo_reflow_mlp_img denoising_steps=1 ft_denoising_steps=1 
  ```

- **在 OpenAI Gym 中使用 PPO 微调 Drifting Policy (1-NFE)：**
  ```bash
  python script/run.py --config-dir=cfg/gym/finetune/hopper-v2 --config-name=ft_ppo_drifting_mlp base_policy_path=<PRETRAINED_DRIFTING_CHECKPOINT>
  ```

- **在 OpenAI Gym 中使用 GRPO 微调 Drifting Policy (1-NFE, 无 Critic)：**
  ```bash
  python script/run.py --config-dir=cfg/gym/finetune/hopper-v2 --config-name=ft_grpo_drifting_mlp base_policy_path=<PRETRAINED_DRIFTING_CHECKPOINT>
  ```

- **在 Robomimic 中使用 PPO 微调基于图像的 Drifting Policy (1-NFE)：**
  ```bash
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_drifting_mlp_img base_policy_path=<PRETRAINED_DRIFTING_CHECKPOINT>
  ```

- **在 Robomimic 中使用 GRPO 微调基于图像的 Drifting Policy (1-NFE, 无 Critic)：**
  ```bash
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_grpo_drifting_mlp_img base_policy_path=<PRETRAINED_DRIFTING_CHECKPOINT>
  ```

#### 故障排除

- **训练崩溃了？尝试恢复它：**
  ```bash
  # 崩溃后从上次中断的地方继续
  ENV_NAME=walker2d-v2 ALG_NAME=difussion \
  python script/run.py --config-dir=cfg/gym/finetune/${ENV_NAME} --config-name=ft_ppo_${ALG_NAME}_mlp \
  resume_path=CHECKPOINT_THAT_FAILED.pt
  ```
  - 这将使您再继续训练 `train.n_train_itr` 轮，从检查点中恢复所有内容。
  - 如果遇到提示配置中没有 `resume_path` 的错误，直接添加该参数即可。
  - 目前我们仅支持从 ReinFlow 获得的预训练或微调检查点，以及通过 `agent/finetune/reinflow/train_ppo_diffusion_img_agent.py` 或 `agent/finetune/reinflow/train_ppo_diffusion_agent.py` 训练的 DPPO 检查点。

- **想要更改预训练策略？**
  只需在命令中指定即可：
  ```bash
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp \
  base_policy_path=NEW_PRETRAINED_POLICY.pt \
  ```

- **想要使用在不同专家数据上训练的策略？**
  别忘了将归一化路径更新为您的新专家数据。归一化不匹配会导致您的曲线看起来非常奇怪！
  ```bash
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp \
  device=cuda:0 sim_device=cuda:1 \
  base_policy_path=NEW_PRETRAINED_POLICY.pt \
  normalization_path=PATH_TO_YOUR_NEW_EXPERT_DATA_DIRECTORY/normalization.npz \
  ```

- **如何在训练开始前指定 WandB 名称？（进行参数搜索时）**
  ```bash
  # 命名您的运行并将其记录在后台
  nohup python script/run.py --config-dir=cfg/gym/finetune/kitchen-complete-v0 --config-name=ft_ppo_shortcut_mlp device=cuda:0 sim_device=cuda:0 denoising_steps=1 ft_denoising_steps=1 seed=3407 \
  name=THE_NAME_I_LIKE_seed3407 \
  > ./ft_kitchen_complete_v0_shortcut_denoise_step_1_seed3407.log 2>&1 &
  ```

- **我的 GPU 不够快，想跑个通宵，该怎么办？**
  最稳定的方法是在后台运行并指定输出到日志文件。
  ```bash
  # 在后台运行保存日志到自定义文件
  nohup python script/run.py --config-dir=cfg/gym/finetune/kitchen-complete-v0 --config-name=ft_ppo_shortcut_mlp > ./MY_CUSTOM_LLOG_FILE.log 2>&1 &
  ```


#### 微调 Flow 策略的快速提示

- **允许失败（有时）：**
  - 使用足够的 rollout 步数来混合成功和失败的情况。太短的轨迹可能会误导 Critic 做出错误的决策。
- **调整 Critic 预热：**
  - 根据初始策略的表现调整预热迭代次数甚至初始化方式。


### 3.3 评估

使用此命令在不同去噪步数下评估预训练策略：
```bash
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_reflow_mlp_img base_policy_path=PATH_TO_THE_POLICY_TO_EVALUATE denoising_step_list=[1,2,4,5,8,16,32,64,128] load_ema=False
```
- **您将得到：** 六张图表，显示回合奖励、成功率、回合长度、推理频率、持续时间和最佳奖励，带阴影区域表示标准差。
- **提示：**
  - 预训练策略设为 `load_ema=True`；微调策略设为 `False`。
  - 无法设置 `denoising_step_list`？将其添加到配置文件中。
  - 清理机器上的其他进程以获得准确的计时。
- **输出：** 保存一张 `.png` 图表和供以后使用的数据文件。示例：<img src="../sample_figs/denoise_step.png" alt="Evaluation Output" width="60%">

**警告：** 如果您使用 ReinFlow 训练了流匹配策略，通常在微调期间我们会对去噪动作进行裁剪。因此，我们建议您在评估脚本中开启 `self.clip_intermediate_actions=True`。否则奖励可能会下降。


### 3.4 敏感性分析
* 更改 1-ReFlow 策略的采样分布。

目前我们支持使用 Beta 分布或 Logit Normal 分布。
只需在预训练命令中更改 `model.sample_t_type=beta` 或 `model.sample_t_type=logitnormal` 即可。

我们提供了在 Square 环境中使用 [Beta](https://drive.google.com/file/d/1d2oDejHLzkzSOuryvDKBZbJlVFBeo3ZQ/view?usp=drive_link) 和 [Logit Normal](https://drive.google.com/file/d/1W-RH23OZMpVW5Ijo2ZmsYIlgVTsXuSfD/view?usp=drive_link) 分布训练的 1-ReFlow 检查点的 Google Drive 链接。


## 4. 充分利用本仓库

### WandB 日志

- **没联网？没关系：**
  - 使用 `wandb.offline_mode=True` 并在以后使用 `wandb sync <your_wandb_offline_file>` 进行同步。
  - 您也可以批量同步离线数据，例如同步 2025/05/10 的所有离线 WandB 运行，在终端运行：
    `for dir in ./wandb_offline/wandb/offline-run-20250510*; do wandb sync "${dir}"; done`
- **跳过 WandB 日志以节省内存：**
  - 设置 `wandb=null` 将停止创建 WandB 日志。
- **我设置了 wandb=null 但后来想找回这些数据：**
  - 别担心。即使 `wandb=null`，我们仍会自动将所有运行的训练记录保存到 `.pkl` 文件。您随时可以使用 `util/pkl2wandb.py` 恢复 WandB 运行记录。

### 内存管理

- **CUDA 显存溢出 (OOM)：**
  - 减少 `env.n_envs` 并相应增加 `train.n_steps` 以保持总步数一致 (`n_envs x n_steps x act_steps`)。
  - 使 `train.n_steps` 成为 `env.max_episode_steps / act_steps` 的倍数。
  - Furniture-Bench 在单个 GPU 上使用 IsaacGym。

### 切换策略或恢复

- **自定义预训练策略：**
  - 在命令中设置 `base_policy_path=<path>`。
- **恢复训练：**
  - 在命令中添加 `resume_path=<checkpoint>`（确保配置中存在 `resume_path: null`）。

### 使用 MuJoCo 渲染

- **我想在 GPU 上渲染得更快**
  - 设置 `sim_device=<gpu_id>` 以实现快速渲染。
  - 没有 EGL？使用 `sim_device=null` 进行较慢的 osmesa 渲染。

### 查看结果

- **Furniture-Bench:** 设置 `env.specific.headless=False` 且 `env.n_envs=1`。
- **D3IL:** 使用 `+env.render=True`、`env.n_envs=1`、`train.render.num=1` 以及 `script/test_d3il_render.py`。
- **Robomimic:** 通过 `env.save_video=True`、`train.render.freq=<iterations>`、`train.render.num=<num_videos>` 记录视频。
