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

import os
import pickle
from collections import deque
import logging

import hydra
import numpy as np
import torch
import wandb

from agent.finetune.train_agent import TrainAgent
from util.scheduler import CosineAnnealingWarmupRestarts
from util.timer import Timer

log = logging.getLogger(__name__)


class DictReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.obs = {}
        self.next_obs = {}
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.terminated = deque(maxlen=capacity)

    def __len__(self):
        return len(self.actions)

    def _init_key(self, store, key):
        if key not in store:
            store[key] = deque(maxlen=self.capacity)

    def add(self, obs, action, next_obs, reward, terminated):
        for key, value in obs.items():
            self._init_key(self.obs, key)
            self.obs[key].append(np.array(value, copy=True))
        for key, value in next_obs.items():
            self._init_key(self.next_obs, key)
            self.next_obs[key].append(np.array(value, copy=True))
        self.actions.append(np.array(action, copy=True))
        self.rewards.append(float(reward))
        self.terminated.append(float(terminated))

    def sample(self, batch_size: int, device: torch.device):
        replace = len(self) < batch_size
        indices = np.random.choice(len(self), size=batch_size, replace=replace)
        obs = {
            key: torch.from_numpy(
                np.stack([self.obs[key][i] for i in indices], axis=0)
            ).float().to(device)
            for key in self.obs
        }
        next_obs = {
            key: torch.from_numpy(
                np.stack([self.next_obs[key][i] for i in indices], axis=0)
            ).float().to(device)
            for key in self.next_obs
        }
        actions = torch.from_numpy(
            np.stack([self.actions[i] for i in indices], axis=0)
        ).float().to(device)
        rewards = torch.from_numpy(
            np.array([self.rewards[i] for i in indices], dtype=np.float32)
        ).float().to(device)
        terminated = torch.from_numpy(
            np.array([self.terminated[i] for i in indices], dtype=np.float32)
        ).float().to(device)
        return obs, actions, next_obs, rewards, terminated


class TrainQGuidedDriftingAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dataset_offline = hydra.utils.instantiate(cfg.offline_dataset)
        self.gamma = cfg.train.gamma
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr
        self.target_ema_rate = cfg.train.target_ema_rate
        self.scale_reward_factor = cfg.train.scale_reward_factor
        self.buffer_size = cfg.train.buffer_size
        self.offline_batch_ratio = cfg.train.offline_batch_ratio
        self.offline_only_iters = cfg.train.offline_only_iters
        self.updates_per_itr = cfg.train.updates_per_itr
        self.n_steps_eval = cfg.train.n_steps_eval
        self.offline_cache_batch_size = cfg.train.get("offline_cache_batch_size", 2048)

        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

    def _cache_offline_dataset(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset_offline,
            batch_size=min(len(self.dataset_offline), self.offline_cache_batch_size),
            drop_last=False,
        )

        obs_chunks = {}
        next_obs_chunks = {}
        action_chunks = []
        reward_chunks = []
        terminated_chunks = []

        for batch in dataloader:
            actions, conditions, rewards, terminated = batch
            action_chunks.append(actions.cpu().numpy())
            reward_chunks.append(rewards.cpu().numpy().reshape(-1))
            terminated_chunks.append(terminated.cpu().numpy().reshape(-1))

            for key, value in conditions.items():
                if key.startswith("next_"):
                    next_obs_chunks.setdefault(key[5:], []).append(value.cpu().numpy())
                else:
                    obs_chunks.setdefault(key, []).append(value.cpu().numpy())

        offline_data = {
            "obs": {
                key: np.concatenate(value, axis=0)
                for key, value in obs_chunks.items()
            },
            "next_obs": {
                key: np.concatenate(value, axis=0)
                for key, value in next_obs_chunks.items()
            },
            "actions": np.concatenate(action_chunks, axis=0),
            "rewards": np.concatenate(reward_chunks, axis=0),
            "terminated": np.concatenate(terminated_chunks, axis=0),
        }
        return offline_data

    def _sample_offline_batch(self, offline_data, batch_size: int):
        replace = len(offline_data["actions"]) < batch_size
        indices = np.random.choice(
            len(offline_data["actions"]), size=batch_size, replace=replace
        )
        obs = {
            key: torch.from_numpy(value[indices]).float().to(self.device)
            for key, value in offline_data["obs"].items()
        }
        next_obs = {
            key: torch.from_numpy(value[indices]).float().to(self.device)
            for key, value in offline_data["next_obs"].items()
        }
        actions = torch.from_numpy(offline_data["actions"][indices]).float().to(
            self.device
        )
        rewards = torch.from_numpy(offline_data["rewards"][indices]).float().to(
            self.device
        )
        terminated = torch.from_numpy(
            offline_data["terminated"][indices]
        ).float().to(self.device)
        return obs, actions, next_obs, rewards, terminated

    def _concat_obs(self, obs_a, obs_b):
        return {key: torch.cat([obs_a[key], obs_b[key]], dim=0) for key in obs_a}

    def _sample_training_batch(self, offline_data, replay_buffer: DictReplayBuffer):
        if len(replay_buffer) == 0 or self.offline_batch_ratio >= 1.0:
            return self._sample_offline_batch(offline_data, self.batch_size)

        n_offline = int(round(self.batch_size * self.offline_batch_ratio))
        n_online = self.batch_size - n_offline
        if n_online <= 0:
            return self._sample_offline_batch(offline_data, self.batch_size)

        offline_batch = self._sample_offline_batch(offline_data, n_offline)
        online_batch = replay_buffer.sample(n_online, self.device)
        obs = self._concat_obs(offline_batch[0], online_batch[0])
        actions = torch.cat([offline_batch[1], online_batch[1]], dim=0)
        next_obs = self._concat_obs(offline_batch[2], online_batch[2])
        rewards = torch.cat([offline_batch[3], online_batch[3]], dim=0)
        terminated = torch.cat([offline_batch[4], online_batch[4]], dim=0)
        return obs, actions, next_obs, rewards, terminated

    def _obs_to_tensor(self, obs_venv):
        return {
            key: torch.from_numpy(value).float().to(self.device)
            for key, value in obs_venv.items()
        }

    def _single_env_obs(self, obs_venv, env_index: int):
        return {
            key: obs_venv[key][env_index]
            for key in obs_venv
        }

    def _collect_rollout(
        self,
        prev_obs_venv,
        replay_buffer: DictReplayBuffer,
        cnt_train_step: int,
    ):
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        # Seed the initial episode boundary: 1 on first rollout (fresh reset),
        # or carry over done flags from the previous rollout.
        if getattr(self, "done_venv", None) is None:
            firsts_trajs[0] = 1
        else:
            firsts_trajs[0] = self.done_venv
        reward_trajs = np.zeros((self.n_steps, self.n_envs))

        for step in range(self.n_steps):
            with torch.no_grad():
                cond = self._obs_to_tensor(prev_obs_venv)
                samples = self.model(cond=cond, deterministic=False).cpu().numpy()

            action_venv = samples[:, : self.act_steps]
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            done_venv = terminated_venv | truncated_venv
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_venv

            for env_ind in range(self.n_envs):
                next_obs = (
                    info_venv[env_ind]["final_obs"]
                    if truncated_venv[env_ind] and "final_obs" in info_venv[env_ind]
                    else self._single_env_obs(obs_venv, env_ind)
                )
                replay_buffer.add(
                    obs=self._single_env_obs(prev_obs_venv, env_ind),
                    action=action_venv[env_ind],
                    next_obs=next_obs,
                    reward=reward_venv[env_ind] * self.scale_reward_factor,
                    terminated=terminated_venv[env_ind],
                )

            prev_obs_venv = obs_venv
            cnt_train_step += self.n_envs * self.act_steps

        # Persist done flags for the next rollout's firsts_trajs[0]
        self.done_venv = done_venv
        metrics = self._summarize_rollout_metrics(firsts_trajs, reward_trajs)
        return prev_obs_venv, cnt_train_step, metrics

    def _summarize_rollout_metrics(self, firsts_trajs, reward_trajs):
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        if len(episodes_start_end) == 0:
            return {
                "num_episode_finished": 0,
                "avg_episode_reward": 0.0,
                "avg_best_reward": 0.0,
                "success_rate": 0.0,
                "avg_traj_length": 0.0,
            }

        reward_trajs_split = [
            reward_trajs[start : end + 1, env_ind]
            for env_ind, start, end in episodes_start_end
        ]
        episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])
        episode_best_reward = np.array(
            [np.max(reward_traj) / self.act_steps for reward_traj in reward_trajs_split]
        )
        episode_lengths = np.array(
            [end - start + 1 for _, start, end in episodes_start_end]
        ) * self.act_steps

        return {
            "num_episode_finished": len(reward_trajs_split),
            "avg_episode_reward": float(np.mean(episode_reward)),
            "avg_best_reward": float(np.mean(episode_best_reward)),
            "success_rate": float(
                np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            ),
            "avg_traj_length": float(np.mean(episode_lengths)),
        }

    def _update_once(self, batch):
        obs_b, actions_b, next_obs_b, rewards_b, terminated_b = batch

        critic_loss, critic_metrics = self.model.loss_critic(
            obs_b,
            next_obs_b,
            actions_b,
            rewards_b,
            terminated_b,
            self.gamma,
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()

        if self.itr >= self.n_critic_warmup_itr:
            actor_loss, actor_metrics = self.model.loss_actor(obs_b)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.actor.parameters(), self.max_grad_norm
                )
            self.actor_optimizer.step()
        else:
            actor_loss = torch.tensor(0.0, device=self.device)
            actor_metrics = {"actor/loss": 0.0}

        self.model.update_target_critic(self.target_ema_rate)
        return actor_loss, critic_loss, actor_metrics, critic_metrics

    def evaluate(self):
        prev_obs_venv = self.reset_env_all()
        firsts_trajs = np.zeros((self.n_steps_eval + 1, self.n_envs))
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps_eval, self.n_envs))

        for step in range(self.n_steps_eval):
            with torch.no_grad():
                cond = self._obs_to_tensor(prev_obs_venv)
                samples = self.model(cond=cond, deterministic=True).cpu().numpy()
            action_venv = samples[:, : self.act_steps]
            obs_venv, reward_venv, terminated_venv, truncated_venv, _ = self.venv.step(
                action_venv
            )
            done_venv = terminated_venv | truncated_venv
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_venv
            prev_obs_venv = obs_venv

        return self._summarize_rollout_metrics(firsts_trajs, reward_trajs)

    def run(self):
        offline_data = self._cache_offline_dataset()
        replay_buffer = DictReplayBuffer(self.buffer_size)

        timer = Timer()
        run_results = []
        cnt_train_step = 0

        while self.itr < self.n_train_itr:
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()

            if eval_mode:
                eval_metrics = self.evaluate()
                actor_loss = 0.0
                critic_loss = 0.0
                actor_metrics = {}
                critic_metrics = {}
                rollout_metrics = eval_metrics
            else:
                if (
                    self.itr == 0
                    or self.reset_at_iteration
                    or not hasattr(self, "prev_obs_venv")
                ):
                    self.prev_obs_venv = self.reset_env_all()
                    self.done_venv = None  # signal fresh reset to _collect_rollout

                rollout_metrics = {
                    "num_episode_finished": 0,
                    "avg_episode_reward": 0.0,
                    "avg_best_reward": 0.0,
                    "success_rate": 0.0,
                    "avg_traj_length": 0.0,
                }
                if self.itr >= self.offline_only_iters:
                    (
                        self.prev_obs_venv,
                        cnt_train_step,
                        rollout_metrics,
                    ) = self._collect_rollout(
                        self.prev_obs_venv,
                        replay_buffer,
                        cnt_train_step,
                    )

                actor_loss = 0.0
                critic_loss = 0.0
                actor_metrics = {}
                critic_metrics = {}
                for _ in range(self.updates_per_itr):
                    batch = self._sample_training_batch(offline_data, replay_buffer)
                    (
                        actor_loss_t,
                        critic_loss_t,
                        actor_metrics,
                        critic_metrics,
                    ) = self._update_once(batch)
                    actor_loss = float(actor_loss_t.item())
                    critic_loss = float(critic_loss_t.item())

                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()

            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0:
                time_cost = timer()
                run_results[-1]["time"] = time_cost

                if eval_mode:
                    log_payload = {
                        "success rate - eval": rollout_metrics["success_rate"],
                        "avg episode reward - eval": rollout_metrics["avg_episode_reward"],
                        "avg best reward - eval": rollout_metrics["avg_best_reward"],
                        "num episode - eval": rollout_metrics["num_episode_finished"],
                        "avg traj length - eval": rollout_metrics["avg_traj_length"],
                    }
                    if self.use_wandb:
                        wandb.log(log_payload, step=self.itr, commit=False)
                    run_results[-1].update(log_payload)
                else:
                    log_payload = {
                        "total env step": cnt_train_step,
                        "loss - actor": actor_loss,
                        "loss - critic": critic_loss,
                        "avg episode reward - train": rollout_metrics["avg_episode_reward"],
                        "avg best reward - train": rollout_metrics["avg_best_reward"],
                        "success rate - train": rollout_metrics["success_rate"],
                        "num episode - train": rollout_metrics["num_episode_finished"],
                        "avg traj length - train": rollout_metrics["avg_traj_length"],
                    }
                    log_payload.update(actor_metrics)
                    log_payload.update(critic_metrics)
                    if self.use_wandb:
                        wandb.log(log_payload, step=self.itr, commit=True)
                    run_results[-1].update(log_payload)

                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)

            self.itr += 1
