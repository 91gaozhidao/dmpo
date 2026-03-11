

## Adding your own dataset or environment


### Pre-training data

Pre-training scripts are at:
- **Diffusion:** [`agent/pretrain/train_diffusion_agent.py`](agent/pretrain/train_diffusion_agent.py)
- **MeanFlow:** [`agent/pretrain/train_meanflow_agent.py`](agent/pretrain/train_meanflow_agent.py)
- **Drifting Policy:** [`agent/pretrain/train_drifting_agent.py`](agent/pretrain/train_drifting_agent.py)

The pre-training dataset [loader](agent/dataset/sequence.py) assumes a npz file containing numpy arrays `states`, `actions`, `images` (if using pixel; img_h = img_w and a multiple of 8) and `traj_lengths`, where `states` and `actions` have the shape of num_total_steps x obs_dim/act_dim, `images` num_total_steps x C (concatenated if multiple images) x H x W, and `traj_lengths` is a 1-D array for indexing across num_total_steps.

For OpenAI Gym and Franka Kitchen tasks, you can download raw datasets from [D4RL datasets](https://huggingface.co/datasets/imone/D4RL/tree/main), and then run `python data_process/hdf5_to_npz_wrapped.py --data_path=<PATH_TO_YOUR_OFFLINE_RL_DATASET>` to convert raw hdf5 to normalized train.npz and normalization.npz files in the same directory. 

To inspect the contents and ranges of the train.npz file, run 
```
python data_process/read_npz.py --data_path=<PATH_TO_YOUR_OFFLINE_RL_DATASET_DIR>/train.npz
```


### Observation history

In our experiments we did not use any observation from previous timesteps (state or pixel), but it is implemented. You can set `cond_steps=<num_state_obs_step>` (and `img_cond_steps=<num_img_obs_step>`, no larger than `cond_steps`) in pre-training, and set the same when fine-tuning the newly pre-trained policy.

### Configuring Drifting Policy

To add Drifting Policy support for a new environment, create three YAML configuration files following the existing templates:

1. **Pretrain config** (`pre_drifting_mlp.yaml` or `pre_drifting_mlp_img.yaml`):
   - Set `_target_: agent.pretrain.train_drifting_agent.TrainDriftingAgent`
   - Use `model._target_: model.drifting.drifting.DriftingPolicy`
   - Set `max_denoising_steps: 1` (1-NFE constraint)
   - Drifting-specific params: `drift_coef`, `neg_drift_coef`, `mask_self`

2. **PPO finetune config** (`ft_ppo_drifting_mlp.yaml`):
   - Set `_target_: agent.finetune.reinflow.train_ppo_drifting_agent.TrainPPODriftingAgent`
   - Use `model._target_: model.drifting.ft_ppo.ppodrifting.PPODrifting`
   - Set `denoising_steps: 1` and `ft_denoising_steps: 1`
   - Set `use_time_independent_noise: true` (appropriate for 1-NFE)

3. **GRPO finetune config** (`ft_grpo_drifting_mlp.yaml`):
   - Set `_target_: agent.finetune.grpo.train_grpo_drifting_agent.TrainGRPODriftingAgent`
   - Use `model._target_: model.drifting.ft_grpo.grpodrifting.GRPODrifting`
   - No Critic network needed; set `group_size >= 16`
   - Configure KL decay: `kl_beta`, `kl_beta_min`, `kl_beta_decay`

For detailed mathematical background and parameter descriptions, see [`docs/Drifting_Guide.md`](docs/Drifting_Guide.md).

### Fine-tuning environment

We follow the Gym format for interacting with the environments. The vectorized environments are initialized at [make_async](env/gym_utils/__init__.py#L10) (called in the parent fine-tuning agent class [here](agent/finetune/train_agent.py#L38-L39)). The current implementation is not the cleanest as we tried to make it compatible with Gym, Robomimic, Furniture-Bench, and D3IL environments, but it should be easy to modify and allow using other environments. We use [multi_step](env/gym_utils/wrapper/multi_step.py) wrapper for history observations and multi-environment-step action execution. We also use environment-specific wrappers such as [robomimic_lowdim](env/gym_utils/wrapper/robomimic_lowdim.py) and [furniture](env/gym_utils/wrapper/furniture.py) for observation/action normalization, etc. You can implement a new environment wrapper if needed.
