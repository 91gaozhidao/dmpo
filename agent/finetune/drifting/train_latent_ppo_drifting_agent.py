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
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from agent.finetune.drifting.latent_ppo_buffer import PPODriftingLatentBuffer
from agent.finetune.train_agent import TrainAgent
from util.scheduler import CosineAnnealingWarmupRestarts
from util.timer import Timer

log = logging.getLogger(__name__)


class TrainLatentPPODriftingAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.gamma = cfg.train.gamma
        self.gae_lambda = cfg.train.get("gae_lambda", 0.95)
        self.clip_coef = cfg.train.get("clip_coef", 0.2)
        self.ent_coef = cfg.train.get("ent_coef", 0.01)
        self.vf_coef = cfg.train.get("vf_coef", 0.5)
        self.target_kl = cfg.train.get("target_kl", None)
        self.update_epochs = cfg.train.get("update_epochs", 10)
        self.n_critic_warmup_itr = cfg.train.get("n_critic_warmup_itr", 0)
        self.reward_scale_running = cfg.train.get("reward_scale_running", True)
        self.reward_scale_const = cfg.train.get("reward_scale_const", 1.0)
        self.normalize_advantages = cfg.train.get("normalize_advantages", True)
        self.grad_accumulate = max(1, int(cfg.train.get("grad_accumulate", 1)))
        self.n_steps_eval = cfg.train.get("n_steps_eval", self.n_steps)

        self.actor_optimizer = torch.optim.AdamW(
            self.model.latent_policy.parameters(),
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

        self.buffer = PPODriftingLatentBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            device=self.device,
        )
        self.done_venv = None

    def _obs_to_tensor(self, obs_venv):
        return {
            key: torch.from_numpy(value).float().to(self.device)
            for key, value in obs_venv.items()
        }

    def _stack_obs_list(self, obs_list):
        return {
            key: np.stack([obs[key] for obs in obs_list], axis=0)
            for key in obs_list[0]
        }

    def _get_truncated_value_overrides(self, info_venv, truncated_venv):
        override_values = np.full((self.n_envs,), np.nan, dtype=np.float32)
        truncated_indices = [
            env_idx
            for env_idx in range(self.n_envs)
            if truncated_venv[env_idx] and "final_obs" in info_venv[env_idx]
        ]
        if len(truncated_indices) == 0:
            return override_values

        final_obs_batch = self._stack_obs_list(
            [info_venv[env_idx]["final_obs"] for env_idx in truncated_indices]
        )
        with torch.no_grad():
            final_values = (
                self.model.value(self._obs_to_tensor(final_obs_batch)).cpu().numpy()
            )
        override_values[truncated_indices] = final_values
        return override_values

    def _collect_rollout(self, prev_obs_venv, cnt_train_step):
        initial_firsts = (
            np.ones((self.n_envs,), dtype=np.float32)
            if self.done_venv is None
            else np.asarray(self.done_venv, dtype=np.float32)
        )
        self.buffer.reset(initial_firsts=initial_firsts)

        for step in range(self.n_steps):
            with torch.no_grad():
                cond = self._obs_to_tensor(prev_obs_venv)
                policy_out = self.model.get_actions(cond=cond, deterministic=False)

            full_action_venv = policy_out["actions"].cpu().numpy()
            action_venv = full_action_venv[:, : self.act_steps]
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )

            next_value_override = self._get_truncated_value_overrides(
                info_venv,
                truncated_venv,
            )
            self.buffer.add(
                step=step,
                obs_venv=prev_obs_venv,
                latent_venv=policy_out["latents"].cpu().numpy(),
                action_venv=full_action_venv,
                reward_venv=reward_venv,
                terminated_venv=terminated_venv,
                truncated_venv=truncated_venv,
                value_venv=policy_out["value"].cpu().numpy(),
                logprob_venv=policy_out["logprob"].cpu().numpy(),
                next_value_override_venv=next_value_override,
            )
            prev_obs_venv = obs_venv
            cnt_train_step += self.n_envs * self.act_steps

        with torch.no_grad():
            last_value = self.model.value(self._obs_to_tensor(prev_obs_venv)).cpu().numpy()
        self.buffer.set_last_values(last_value)
        self.buffer.update()
        self.done_venv = np.asarray(terminated_venv | truncated_venv, dtype=np.float32)
        rollout_metrics = self.buffer.summarize_episode_reward()
        rollout_metrics["explained_variance"] = float(self.buffer.get_explained_var())
        return prev_obs_venv, cnt_train_step, rollout_metrics

    def _step_optimizers(self, allow_actor_update):
        actor_grad_norm = 0.0
        critic_grad_norm = 0.0

        if allow_actor_update:
            actor_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.latent_policy.parameters(),
                    float("inf"),
                )
            )
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.latent_policy.parameters(),
                    self.max_grad_norm,
                )
            self.actor_optimizer.step()

        critic_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(),
                float("inf"),
            )
        )
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(),
                self.max_grad_norm,
            )
        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        return actor_grad_norm, critic_grad_norm

    def _ppo_update(self):
        metrics = {
            "train/policy_loss": [],
            "train/value_loss": [],
            "train/entropy": [],
            "train/approx_kl": [],
            "train/clipfrac": [],
            "train/ratio_mean": [],
            "train/latent_std_mean": [],
            "train/latent_std_min": [],
            "train/latent_std_max": [],
        }
        allow_actor_update = self.itr >= self.n_critic_warmup_itr
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        batch_counter = 0
        early_stop = False
        actor_grad_norm = 0.0
        critic_grad_norm = 0.0

        for minibatch in self.buffer.iter_minibatches(
            batch_size=self.batch_size,
            update_epochs=self.update_epochs,
            device=self.device,
            normalize_advantages=self.normalize_advantages,
        ):
            batch_counter += 1
            eval_out = self.model.evaluate_latents(
                cond=minibatch["obs"],
                z=minibatch["latents"],
            )
            new_logprob = eval_out["logprob"]
            entropy = eval_out["entropy"]
            new_value = eval_out["value"]
            latent_std = eval_out["latent_std"]

            log_ratio = new_logprob - minibatch["logprobs"]
            ratio = log_ratio.exp()
            unclipped = -minibatch["advantages"] * ratio
            clipped = -minibatch["advantages"] * torch.clamp(
                ratio,
                1.0 - self.clip_coef,
                1.0 + self.clip_coef,
            )
            policy_loss = torch.max(unclipped, clipped).mean()
            value_loss = 0.5 * F.mse_loss(new_value, minibatch["returns"])
            entropy_mean = entropy.mean()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clipfrac = (
                (torch.abs(ratio - 1.0) > self.clip_coef).float().mean()
            )

            if self.target_kl is not None and approx_kl.item() > self.target_kl:
                early_stop = True
                break

            actor_objective = (
                policy_loss - self.ent_coef * entropy_mean
                if allow_actor_update
                else torch.zeros_like(policy_loss)
            )
            total_loss = (
                actor_objective + self.vf_coef * value_loss
            ) / self.grad_accumulate
            total_loss.backward()

            metrics["train/policy_loss"].append(float(policy_loss.item()))
            metrics["train/value_loss"].append(float(value_loss.item()))
            metrics["train/entropy"].append(float(entropy_mean.item()))
            metrics["train/approx_kl"].append(float(approx_kl.item()))
            metrics["train/clipfrac"].append(float(clipfrac.item()))
            metrics["train/ratio_mean"].append(float(ratio.mean().item()))
            metrics["train/latent_std_mean"].append(float(latent_std.mean().item()))
            metrics["train/latent_std_min"].append(float(latent_std.min().item()))
            metrics["train/latent_std_max"].append(float(latent_std.max().item()))

            if batch_counter % self.grad_accumulate == 0:
                actor_grad_norm, critic_grad_norm = self._step_optimizers(
                    allow_actor_update=allow_actor_update
                )

        performed_backward_steps = len(metrics["train/policy_loss"])
        if batch_counter % self.grad_accumulate != 0 and performed_backward_steps > 0:
            actor_grad_norm, critic_grad_norm = self._step_optimizers(
                allow_actor_update=allow_actor_update
            )
        elif early_stop:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)

        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        reduced_metrics = {
            key: (float(np.mean(values)) if len(values) > 0 else 0.0)
            for key, values in metrics.items()
        }
        reduced_metrics["train/actor_grad_norm"] = actor_grad_norm
        reduced_metrics["train/critic_grad_norm"] = critic_grad_norm
        reduced_metrics["train/early_stop_due_to_kl"] = float(early_stop)
        reduced_metrics["train/actor_updates_enabled"] = float(allow_actor_update)
        return reduced_metrics

    def _summarize_eval_metrics(self, firsts_trajs, reward_trajs):
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for idx in range(len(env_steps) - 1):
                start = env_steps[idx]
                end = env_steps[idx + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        if len(episodes_start_end) == 0:
            return {
                "num_episode_finished": 0,
                "avg_episode_reward": 0.0,
                "avg_best_reward": 0.0,
                "success_rate": 0.0,
                "avg_episode_length": 0.0,
            }

        reward_slices = [
            reward_trajs[start : end + 1, env_ind]
            for env_ind, start, end in episodes_start_end
        ]
        episode_reward = np.array([np.sum(reward) for reward in reward_slices])
        if self.furniture_sparse_reward:
            episode_best_reward = episode_reward
        else:
            episode_best_reward = np.array(
                [np.max(reward) / self.act_steps for reward in reward_slices]
            )
        episode_lengths = np.array(
            [end - start + 1 for _, start, end in episodes_start_end]
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

    def evaluate(self):
        prev_obs_venv = self.reset_env_all()
        firsts_trajs = np.zeros((self.n_steps_eval + 1, self.n_envs), dtype=np.float32)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps_eval, self.n_envs), dtype=np.float32)

        for step in range(self.n_steps_eval):
            with torch.no_grad():
                cond = self._obs_to_tensor(prev_obs_venv)
                actions = self.model(cond=cond, deterministic=True).cpu().numpy()
            action_venv = actions[:, : self.act_steps]
            obs_venv, reward_venv, terminated_venv, truncated_venv, _ = self.venv.step(
                action_venv
            )
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv
            prev_obs_venv = obs_venv

        return self._summarize_eval_metrics(firsts_trajs, reward_trajs)

    def run(self):
        timer = Timer()
        run_results = []
        cnt_train_step = 0

        while self.itr < self.n_train_itr:
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()

            if eval_mode:
                rollout_metrics = self.evaluate()
                update_metrics = {}
            else:
                if (
                    self.itr == 0
                    or self.reset_at_iteration
                    or not hasattr(self, "prev_obs_venv")
                ):
                    self.prev_obs_venv = self.reset_env_all()
                    self.done_venv = None
                self.prev_obs_venv, cnt_train_step, rollout_metrics = self._collect_rollout(
                    self.prev_obs_venv,
                    cnt_train_step,
                )
                update_metrics = self._ppo_update()

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
                        "avg traj length - eval": rollout_metrics["avg_episode_length"],
                    }
                    if self.use_wandb:
                        wandb.log(log_payload, step=self.itr, commit=False)
                else:
                    log_payload = {
                        "total env step": cnt_train_step,
                        "avg episode reward - train": rollout_metrics["avg_episode_reward"],
                        "avg best reward - train": rollout_metrics["avg_best_reward"],
                        "success rate - train": rollout_metrics["success_rate"],
                        "num episode - train": rollout_metrics["num_episode_finished"],
                        "avg traj length - train": rollout_metrics["avg_episode_length"],
                        "explained variance - train": rollout_metrics["explained_variance"],
                        "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                        "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                    }
                    log_payload.update(update_metrics)
                    if self.use_wandb:
                        wandb.log(log_payload, step=self.itr, commit=True)

                run_results[-1].update(log_payload)
                with open(self.result_path, "wb") as file_obj:
                    pickle.dump(run_results, file_obj)

            self.itr += 1
