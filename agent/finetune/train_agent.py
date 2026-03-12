# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
"""
Parent fine-tuning agent class.
"""

import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import logging
import wandb
import random

log = logging.getLogger(__name__)
from env.gym_utils import make_async


class TrainAgent:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Wandb (revised by ReinFlow authors)
        self.use_wandb = cfg.wandb is not None
        if self.use_wandb:
            # Check if offline mode is enabled (e.g., via config or environment variable)
            offline_mode = cfg.wandb.get("offline_mode", False)  # Add this to your config if desired
            if offline_mode:
                wandb_dir = cfg.wandb.get("dir", "./wandb_offline")  # Local directory for offline logs
            else:
                wandb_dir = cfg.wandb.get("dir", "./wandb")
            os.makedirs(wandb_dir, exist_ok=True)  # Ensure the directory exists
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode="offline" if offline_mode else "online",  # Switch to offline mode
                dir=wandb_dir,  # Specify local directory for offline logs
            )
            # Get the exact subfolder for this run
            run_id = wandb.run.id  # Unique ID for the run
            run_dir = wandb.run.dir  # Full path to the run's directory
            if offline_mode:
                log.info(f"Wandb running in offline mode. Logs will be saved to {os.path.dirname(run_dir)}")
                log.info(f"Run ID: {run_id}")
            else:
                log.info(f"Wandb running online. Run ID: {run_id}")

        train_cfg = cfg.get("train", {})
        env_cfg = cfg.env
        env_name = env_cfg.get("name", env_cfg.get("env_name", cfg.get("env_name")))
        env_type = env_cfg.get("env_type", cfg.get("env_suite", None))
        shape_meta = cfg.get("shape_meta", None)
        use_image_obs = env_cfg.get(
            "use_image_obs",
            shape_meta is not None and "rgb" in shape_meta.obs,
        )
        robomimic_env_cfg_path = cfg.get("robomimic_env_cfg_path", None)
        if robomimic_env_cfg_path is None and env_type == "robomimic" and env_name is not None:
            env_meta_name = f"{env_name}-img.json" if use_image_obs else f"{env_name}.json"
            candidate_path = os.path.join("cfg", "robomimic", "env_meta", env_meta_name)
            if os.path.exists(candidate_path):
                robomimic_env_cfg_path = candidate_path

        # Make vectorized env
        if env_name is None:
            raise ValueError("Could not resolve env name from the fine-tuning config.")
        self.env_name = env_name
        self.use_image_obs = use_image_obs
        self.venv = make_async(
            env_name,
            env_type=env_type,
            num_envs=env_cfg.n_envs,
            asynchronous=True,
            max_episode_steps=env_cfg.max_episode_steps,
            wrappers=env_cfg.get("wrappers", None),
            robomimic_env_cfg_path=robomimic_env_cfg_path,
            shape_meta=shape_meta,
            use_image_obs=use_image_obs,
            render=env_cfg.get("render", False),
            render_offscreen=env_cfg.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            **env_cfg.specific if "specific" in env_cfg else {},
        )
        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(env_cfg.n_envs)]
            )  # otherwise parallel envs might have the same initial states!
            # isaacgym environments do not need seeding
        self.n_envs = env_cfg.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = env_cfg.max_episode_steps
        self.reset_at_iteration = env_cfg.get("reset_at_iteration", True)
        self.save_full_observations = env_cfg.get("save_full_observations", False)
        self.furniture_sparse_reward = (
            env_cfg.specific.get("sparse_reward", False)
            if "specific" in env_cfg
            else False
        )  # furniture specific, for best reward calculation

        # Batch size for gradient update
        self.batch_size: int = train_cfg.get("batch_size", cfg.get("batch_size"))

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)

        # Training params
        self.itr = 0
        self.n_train_itr = train_cfg.get("n_train_itr", cfg.get("n_train_itr"))
        self.val_freq = train_cfg.get("val_freq", cfg.get("val_freq", 0))
        self.force_train = train_cfg.get("force_train", False)
        self.n_steps = train_cfg.get("n_steps", cfg.get("n_steps"))
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else env_cfg.get("best_reward_threshold_for_success", 1)
        )
        self.max_grad_norm = train_cfg.get("max_grad_norm", None)

        # Logging, rendering, checkpoints
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.result_path = os.path.join(self.logdir, "result.pkl")
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_trajs = train_cfg.get("save_trajs", False)
        self.log_freq = train_cfg.get("log_freq", cfg.get("log_freq", 1))
        self.save_model_freq = train_cfg.get(
            "save_model_freq",
            cfg.get("save_model_freq", 1),
        )
        render_cfg = train_cfg.get("render", None)
        self.render_freq = (
            render_cfg.get("freq", 1) if render_cfg is not None else 1
        )
        self.n_render = (
            render_cfg.get("num", cfg.get("render_num", 0))
            if render_cfg is not None
            else cfg.get("render_num", 0)
        )
        self.render_video = env_cfg.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
        self.traj_plotter = (
            hydra.utils.instantiate(train_cfg.plotter)
            if "plotter" in train_cfg
            else None
        )

    def run(self):
        pass

    def save_model(self):
        """
        saves model to disk; no ema
        """
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
        }  # right now `model` includes weights for `network`, `actor`, `actor_ft`. Weights for `network` is redundant, and we can use `actor` weights as the base policy (earlier denoising steps) and `actor_ft` weights as the fine-tuned policy (later denoising steps) during evaluation.
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")
        log.info(f"Evaluation results saved to {self.result_path}")

    def load(self, itr):
        """
        loads model from disk
        """
        loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
        data = torch.load(loadpath)

        self.itr = data["itr"]
        self.model.load_state_dict(data["model"])

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # convert to OrderedDict if obs_venv is a list of dict
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
