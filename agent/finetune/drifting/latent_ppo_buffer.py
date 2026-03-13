# MIT License
#
# Copyright (c) 2026 ReinFlow Authors
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

from __future__ import annotations

import logging

import numpy as np
import torch

from util.reward_scaling import RunningRewardScaler

log = logging.getLogger(__name__)


class PPODriftingLatentBuffer:
    def __init__(
        self,
        n_steps,
        n_envs,
        horizon_steps,
        act_steps,
        action_dim,
        save_full_observation,
        furniture_sparse_reward,
        best_reward_threshold_for_success,
        reward_scale_running,
        gamma,
        gae_lambda,
        reward_scale_const,
        device,
    ):
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.horizon_steps = int(horizon_steps)
        self.act_steps = int(act_steps)
        self.action_dim = int(action_dim)
        self.save_full_observation = save_full_observation
        self.furniture_sparse_reward = furniture_sparse_reward
        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        self.reward_scale_running = reward_scale_running
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale_const = reward_scale_const
        self.device = torch.device(device)
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(
                num_envs=n_envs,
                gamma=gamma,
            )
        self.reset()

    def reset(self, initial_firsts=None):
        self.obs_trajs = {}
        self.latent_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.horizon_steps, self.action_dim),
            dtype=np.float32,
        )
        self.action_trajs = np.zeros_like(self.latent_trajs)
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs), dtype=np.float32)
        if initial_firsts is not None:
            self.firsts_trajs[0] = np.asarray(initial_firsts, dtype=np.float32)
        self.value_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.logprobs_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.next_value_overrides = np.full(
            (self.n_steps, self.n_envs),
            np.nan,
            dtype=np.float32,
        )
        self.last_values = np.zeros((self.n_envs,), dtype=np.float32)
        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.returns_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

    def _ensure_obs_storage(self, obs_venv):
        for key, value in obs_venv.items():
            if key not in self.obs_trajs:
                self.obs_trajs[key] = np.zeros(
                    (self.n_steps, self.n_envs) + tuple(value.shape[1:]),
                    dtype=value.dtype,
                )

    def add(
        self,
        step,
        obs_venv,
        latent_venv,
        action_venv,
        reward_venv,
        terminated_venv,
        truncated_venv,
        value_venv,
        logprob_venv,
        next_value_override_venv=None,
    ):
        self._ensure_obs_storage(obs_venv)
        done_venv = np.asarray(terminated_venv | truncated_venv, dtype=np.float32)
        for key, value in obs_venv.items():
            self.obs_trajs[key][step] = np.asarray(value)
        self.latent_trajs[step] = np.asarray(latent_venv, dtype=np.float32)
        self.action_trajs[step] = np.asarray(action_venv, dtype=np.float32)
        self.reward_trajs[step] = np.asarray(reward_venv, dtype=np.float32)
        self.terminated_trajs[step] = np.asarray(terminated_venv, dtype=np.float32)
        self.firsts_trajs[step + 1] = done_venv
        self.value_trajs[step] = np.asarray(value_venv, dtype=np.float32)
        self.logprobs_trajs[step] = np.asarray(logprob_venv, dtype=np.float32)
        if next_value_override_venv is not None:
            self.next_value_overrides[step] = np.asarray(
                next_value_override_venv,
                dtype=np.float32,
            )

    def set_last_values(self, last_value_venv):
        self.last_values = np.asarray(last_value_venv, dtype=np.float32)

    def normalize_reward(self):
        if self.reward_scale_running:
            reward_transpose = self.running_reward_scaler(
                reward=self.reward_trajs.T,
                first=self.firsts_trajs[:-1].T,
            )
            self.reward_trajs = reward_transpose.T

    def update_adv_returns(self):
        next_values = np.zeros_like(self.value_trajs)
        if self.n_steps > 1:
            next_values[:-1] = self.value_trajs[1:]
        next_values[-1] = self.last_values

        override_mask = ~np.isnan(self.next_value_overrides)
        next_values[override_mask] = self.next_value_overrides[override_mask]

        lastgaelam = np.zeros((self.n_envs,), dtype=np.float32)
        for step in reversed(range(self.n_steps)):
            non_terminal = 1.0 - self.terminated_trajs[step]
            delta = (
                self.reward_trajs[step] * self.reward_scale_const
                + self.gamma * next_values[step] * non_terminal
                - self.value_trajs[step]
            )
            lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
            self.advantages_trajs[step] = lastgaelam
        self.returns_trajs = self.advantages_trajs + self.value_trajs

    def update(self):
        self.normalize_reward()
        self.update_adv_returns()

    def iter_minibatches(
        self,
        batch_size,
        update_epochs,
        device,
        shuffle=True,
        normalize_advantages=True,
    ):
        total_batch = self.n_steps * self.n_envs
        batch_size = min(int(batch_size), total_batch)
        flat_obs = {
            key: value.reshape((total_batch,) + value.shape[2:])
            for key, value in self.obs_trajs.items()
        }
        flat_latents = self.latent_trajs.reshape(
            total_batch, self.horizon_steps, self.action_dim
        )
        flat_actions = self.action_trajs.reshape(
            total_batch, self.horizon_steps, self.action_dim
        )
        flat_returns = self.returns_trajs.reshape(total_batch)
        flat_values = self.value_trajs.reshape(total_batch)
        flat_advantages = self.advantages_trajs.reshape(total_batch)
        flat_logprobs = self.logprobs_trajs.reshape(total_batch)

        if normalize_advantages and total_batch > 1:
            adv_mean = flat_advantages.mean()
            adv_std = flat_advantages.std() + 1e-8
            flat_advantages = (flat_advantages - adv_mean) / adv_std

        indices = np.arange(total_batch)
        for _ in range(update_epochs):
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, total_batch, batch_size):
                batch_indices = indices[start : start + batch_size]
                obs_batch = {
                    key: torch.from_numpy(value[batch_indices]).float().to(device)
                    for key, value in flat_obs.items()
                }
                yield {
                    "obs": obs_batch,
                    "latents": torch.from_numpy(flat_latents[batch_indices]).float().to(
                        device
                    ),
                    "actions": torch.from_numpy(flat_actions[batch_indices]).float().to(
                        device
                    ),
                    "returns": torch.from_numpy(flat_returns[batch_indices]).float().to(
                        device
                    ),
                    "values": torch.from_numpy(flat_values[batch_indices]).float().to(
                        device
                    ),
                    "advantages": torch.from_numpy(
                        flat_advantages[batch_indices]
                    ).float().to(device),
                    "logprobs": torch.from_numpy(
                        flat_logprobs[batch_indices]
                    ).float().to(device),
                }

    def get_explained_var(self):
        values = self.value_trajs.reshape(-1)
        returns = self.returns_trajs.reshape(-1)
        var_y = np.var(returns)
        if var_y == 0:
            return np.nan
        return 1 - np.var(returns - values) / var_y

    def summarize_episode_reward(self):
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(self.firsts_trajs[:, env_ind] == 1)[0]
            for idx in range(len(env_steps) - 1):
                start = env_steps[idx]
                end = env_steps[idx + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        if len(episodes_start_end) == 0:
            log.info("[WARNING] No episode completed within the iteration!")
            return {
                "num_episode_finished": 0,
                "avg_episode_reward": 0.0,
                "avg_best_reward": 0.0,
                "success_rate": 0.0,
                "avg_episode_length": 0.0,
            }

        reward_slices = [
            self.reward_trajs[start : end + 1, env_ind]
            for env_ind, start, end in episodes_start_end
        ]
        episode_reward = np.array(
            [np.sum(reward_traj) for reward_traj in reward_slices],
            dtype=np.float32,
        )
        if self.furniture_sparse_reward:
            episode_best_reward = episode_reward
        else:
            episode_best_reward = np.array(
                [np.max(reward_traj) / self.act_steps for reward_traj in reward_slices],
                dtype=np.float32,
            )
        episode_lengths = np.array(
            [end - start + 1 for _, start, end in episodes_start_end],
            dtype=np.float32,
        ) * self.act_steps
        return {
            "num_episode_finished": len(reward_slices),
            "avg_episode_reward": float(np.mean(episode_reward)),
            "avg_best_reward": float(np.mean(episode_best_reward)),
            "success_rate": float(
                np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            ),
            "avg_episode_length": float(np.mean(episode_lengths)),
        }
