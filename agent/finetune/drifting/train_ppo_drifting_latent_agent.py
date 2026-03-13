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

"""
On-policy PPO trainer for Latent-PPO Drifting.

Collects on-policy rollouts using the latent z-policy over the frozen drifting
generator, then updates the z-policy and V(s) critic using clipped PPO with GAE.
"""

import os
import pickle
import logging
from typing import Optional

import numpy as np
import torch
import wandb

from agent.finetune.train_agent import TrainAgent
from agent.finetune.drifting.ppo_drifting_latent_buffer import PPODriftingLatentBuffer
from util.scheduler import CosineAnnealingWarmupRestarts
from util.timer import Timer

log = logging.getLogger(__name__)


class TrainPPODriftingLatentAgent(TrainAgent):
    """On-policy PPO agent that trains a latent z-policy over a frozen drifting generator."""

    def __init__(self, cfg):
        super().__init__(cfg)

        # PPO hyperparameters
        self.gamma = cfg.train.gamma
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr
        self.update_epochs: int = cfg.train.update_epochs
        self.target_kl: Optional[float] = cfg.train.target_kl
        self.ent_coef: float = cfg.train.get("ent_coef", 0.01)
        self.vf_coef: float = cfg.train.get("vf_coef", 0.5)
        self.n_steps_eval = cfg.train.get("n_steps_eval", 500)

        # Reward scaling
        self.reward_scale_running: bool = cfg.train.get("reward_scale_running", False)
        if self.reward_scale_running:
            from util.reward_scaling import RunningRewardScaler
            self.running_reward_scaler = RunningRewardScaler(self.n_envs)
        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1.0)

        # Optimizers for latent_policy and critic (generator is frozen)
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

    def run(self):
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))

        while self.itr < self.n_train_itr:
            # Prepare video paths
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Eval or train mode
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            # Keep generator always in eval mode
            self.model.generator.eval()
            last_itr_eval = eval_mode

            # Reset envs as needed
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv

            # Allocate rollout storage
            n_rollout_steps = self.n_steps_eval if eval_mode else self.n_steps
            obs_trajs = {
                "state": np.zeros(
                    (n_rollout_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            z_trajs = np.zeros(
                (n_rollout_steps, self.n_envs, self.horizon_steps, self.action_dim)
            )
            samples_trajs = np.zeros(
                (n_rollout_steps, self.n_envs, self.horizon_steps, self.action_dim)
            )
            reward_trajs = np.zeros((n_rollout_steps, self.n_envs))
            terminated_trajs = np.zeros((n_rollout_steps, self.n_envs))
            value_trajs = np.zeros((n_rollout_steps, self.n_envs))
            logprob_trajs = np.zeros((n_rollout_steps, self.n_envs))

            # ── Collect rollout ──
            for step in range(n_rollout_steps):
                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(val).float().to(self.device)
                        for key, val in prev_obs_venv.items()
                    }
                    if eval_mode:
                        action = self.model(cond=cond, deterministic=True)
                        output_venv = action.cpu().numpy()
                        z_np = np.zeros_like(output_venv)
                        val_np = np.zeros(self.n_envs)
                        lp_np = np.zeros(self.n_envs)
                    else:
                        action, z, logprob_z, value = self.model.sample_with_latent(
                            cond=cond, deterministic=False,
                        )
                        output_venv = action.cpu().numpy()
                        z_np = z.cpu().numpy()
                        val_np = value.cpu().numpy()
                        lp_np = logprob_z.cpu().numpy()

                action_venv = output_venv[:, :self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv

                obs_trajs["state"][step] = prev_obs_venv["state"]
                z_trajs[step] = z_np
                samples_trajs[step] = output_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                value_trajs[step] = val_np
                logprob_trajs[step] = lp_np
                firsts_trajs[step + 1] = done_venv

                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # ── Episode reward summary ──
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))

            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start:end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(r) for r in reward_trajs_split]
                )
                if self.furniture_sparse_reward:
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [np.max(r) / self.act_steps for r in reward_trajs_split]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # ── PPO Update ──
            if not eval_mode:
                with torch.no_grad():
                    # Reward scaling
                    if self.reward_scale_running:
                        reward_trajs_T = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_T.T

                    # Bootstrap last value
                    last_cond = {
                        key: torch.from_numpy(val).float().to(self.device)
                        for key, val in obs_venv.items()
                    }
                    last_values = self.model.critic(last_cond).squeeze(-1).cpu().numpy()
                    last_dones = terminated_trajs[-1]

                    # GAE
                    advantages = np.zeros_like(reward_trajs)
                    lastgaelam = 0.0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            next_values = last_values
                            next_nonterminal = 1.0 - last_dones
                        else:
                            next_values = value_trajs[t + 1]
                            next_nonterminal = 1.0 - terminated_trajs[t]
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * next_values * next_nonterminal
                            - value_trajs[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
                        )
                    returns = advantages + value_trajs

                # Flatten to (n_steps * n_envs)
                total_steps = self.n_steps * self.n_envs
                obs_k = {
                    "state": torch.from_numpy(
                        obs_trajs["state"][:self.n_steps].reshape(
                            total_steps, self.n_cond_step, self.obs_dim
                        )
                    ).float().to(self.device)
                }
                z_k = torch.from_numpy(
                    z_trajs[:self.n_steps].reshape(
                        total_steps, self.horizon_steps, self.action_dim
                    )
                ).float().to(self.device)
                returns_k = torch.from_numpy(
                    returns.reshape(total_steps)
                ).float().to(self.device)
                values_k = torch.from_numpy(
                    value_trajs[:self.n_steps].reshape(total_steps)
                ).float().to(self.device)
                advantages_k = torch.from_numpy(
                    advantages.reshape(total_steps)
                ).float().to(self.device)
                logprobs_k = torch.from_numpy(
                    logprob_trajs[:self.n_steps].reshape(total_steps)
                ).float().to(self.device)

                # PPO update epochs
                clipfracs = []
                for update_epoch in range(self.update_epochs):
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)

                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]

                        obs_b = {"state": obs_k["state"][inds_b]}
                        z_b = z_k[inds_b]
                        returns_b = returns_k[inds_b]
                        values_b = values_k[inds_b]
                        advantages_b = advantages_k[inds_b]
                        logprobs_b = logprobs_k[inds_b]

                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            std,
                        ) = self.model.loss(
                            obs_b,
                            z_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                        )
                        clipfracs.append(clipfrac)

                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.latent_policy.parameters(),
                                    self.max_grad_norm,
                                )
                            self.actor_optimizer.step()
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.critic.parameters(),
                                self.max_grad_norm,
                            )
                        self.critic_optimizer.step()

                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variance
                y_pred = values_k.cpu().numpy()
                y_true = returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Update learning rates
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log and save
            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | "
                        f"avg episode reward {avg_episode_reward:8.4f} | "
                        f"avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | "
                        f"loss {loss:8.4f} | pg {pg_loss:8.4f} | "
                        f"v {v_loss:8.4f} | ent {-entropy_loss:8.4f} | "
                        f"reward {avg_episode_reward:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss.item(),
                                "pg loss": pg_loss.item(),
                                "value loss": v_loss.item(),
                                "entropy": (-entropy_loss).item(),
                                "latent std": std.item() if torch.is_tensor(std) else std,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
