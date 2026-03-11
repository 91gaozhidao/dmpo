
## 添加您自己的数据集或环境


### 预训练数据

预训练脚本位于：
- **Diffusion:** [`agent/pretrain/train_diffusion_agent.py`](agent/pretrain/train_diffusion_agent.py)
- **MeanFlow:** [`agent/pretrain/train_meanflow_agent.py`](agent/pretrain/train_meanflow_agent.py)
- **Drifting Policy:** [`agent/pretrain/train_drifting_agent.py`](agent/pretrain/train_drifting_agent.py)

预训练数据集 [加载器](agent/dataset/sequence.py) 假设使用包含 numpy 数组 `states`、`actions`、`images`（如果使用像素；img_h = img_w 且为 8 的倍数）和 `traj_lengths` 的 npz 文件，其中 `states` 和 `actions` 的形状为 num_total_steps x obs_dim/act_dim，`images` 为 num_total_steps x C（如果有多张图像则拼接）x H x W，而 `traj_lengths` 是用于跨 num_total_steps 索引的一维数组。

对于 OpenAI Gym 和 Franka Kitchen 任务，您可以从 [D4RL 数据集](https://huggingface.co/datasets/imone/D4RL/tree/main) 下载原始数据集，然后运行 `python data_process/hdf5_to_npz_wrapped.py --data_path=<PATH_TO_YOUR_OFFLINE_RL_DATASET>` 将原始 hdf5 转换为同一目录下的归一化 train.npz 和 normalization.npz 文件。

要检查 train.npz 文件的内容和范围，请运行：
```
python data_process/read_npz.py --data_path=<PATH_TO_YOUR_OFFLINE_RL_DATASET_DIR>/train.npz
```


### 观测历史

在我们的实验中，我们没有使用来自先前时间步的任何观测（状态或像素），但该功能已实现。您可以在预训练中设置 `cond_steps=<num_state_obs_step>`（以及 `img_cond_steps=<num_img_obs_step>`，不大于 `cond_steps`），并在微调新预训练的策略时设置相同的值。

### 配置 Drifting Policy

要为新环境添加 Drifting Policy 支持，请按照现有模板创建三个 YAML 配置文件：

1. **预训练配置** (`pre_drifting_mlp.yaml` 或 `pre_drifting_mlp_img.yaml`):
   - 设置 `_target_: agent.pretrain.train_drifting_agent.TrainDriftingAgent`
   - 使用 `model._target_: model.drifting.drifting.DriftingPolicy`
   - 设置 `max_denoising_steps: 1` (1-NFE 约束)
   - Drifting 特有参数：`drift_coef`, `neg_drift_coef`, `mask_self`

2. **PPO 微调配置** (`ft_ppo_drifting_mlp.yaml`):
   - 设置 `_target_: agent.finetune.reinflow.train_ppo_drifting_agent.TrainPPODriftingAgent`
   - 使用 `model._target_: model.drifting.ft_ppo.ppodrifting.PPODrifting`
   - 设置 `denoising_steps: 1` 和 `ft_denoising_steps: 1`
   - 设置 `use_time_independent_noise: true` (适用于 1-NFE)

3. **GRPO 微调配置** (`ft_grpo_drifting_mlp.yaml`):
   - 设置 `_target_: agent.finetune.grpo.train_grpo_drifting_agent.TrainGRPODriftingAgent`
   - 使用 `model._target_: model.drifting.ft_grpo.grpodrifting.GRPODrifting`
   - 不需要 Critic 网络；设置 `group_size >= 16`
   - 配置 KL 衰减：`kl_beta`, `kl_beta_min`, `kl_beta_decay`

有关详细的数学背景和参数说明，请参阅 [`docs/Drifting_Guide.md`](docs/Drifting_Guide.md)。

### 微调环境

我们遵循 Gym 格式与环境进行交互。矢量化环境在 [make_async](env/gym_utils/__init__.py#L10)（在父级微调智能体类 [此处](agent/finetune/train_agent.py#L38-L39) 调用）中初始化。目前的实现不是最简洁的，因为我们尝试使其与 Gym、Robomimic、Furniture-Bench 和 D3IL 环境兼容，但也应该很容易修改以允许使用其他环境。我们使用 [multi_step](env/gym_utils/wrapper/multi_step.py) 包装器进行历史观测和多环境步动作执行。我们还使用特定于环境的包装器，如 [robomimic_lowdim](env/gym_utils/wrapper/robomimic_lowdim.py) 和 [furniture](env/gym_utils/wrapper/furniture.py)，用于观测/动作归一化等。如果需要，您可以实现新的环境包装器。
