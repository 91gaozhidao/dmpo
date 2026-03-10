# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Continuous GRPO (Group Relative Policy Optimization) fine-tuning agent
for Drifting Policy.

GRPO eliminates the Critic/Value network entirely. For each training
iteration, G trajectories are sampled from the same initial state
(homogeneous reset). The advantages are computed via intra-group Z-score
normalization of episodic returns, and the policy is updated with a
clipped surrogate loss plus a KL penalty against a frozen reference policy.

Environments that do not support strict homogeneous reset (seed-based)
fall back to treating the batch of parallel trajectories as a single group.
"""

import os
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm
import numpy as np
import torch
from agent.finetune.train_agent import TrainAgent
from agent.finetune.grpo.buffer import GRPOBuffer
from model.drifting.ft_grpo.grpodrifting import GRPODrifting


class TrainGRPODriftingAgent(TrainAgent):
    """
    GRPO fine-tuning agent for Drifting Policy.

    Key differences from PPO-based agents:
    - No Critic network: advantages are computed from group returns.
    - Group sampling: G trajectories from the same initial state.
    - KL penalty against a frozen reference policy replaces the value loss.
    - Only the actor parameters are optimized.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: GRPODrifting

        # GRPO-specific configuration
        self.group_size = cfg.train.get("group_size", 8)
        self.grpo_lr = cfg.train.get("grpo_lr", 1e-4)
        self.update_epochs = cfg.train.get("update_epochs", 4)
        self.grpo_batch_size = cfg.train.get("grpo_batch_size", 256)
        self.max_episode_steps_grpo = cfg.env.max_episode_steps
        self.use_homogeneous_reset = cfg.train.get("use_homogeneous_reset", True)

        # KL penalty schedule: start high to prevent early divergence
        self.beta = cfg.train.get("kl_beta", 0.05)
        self.beta_min = cfg.train.get("kl_beta_min", 0.001)
        self.beta_decay = cfg.train.get("kl_beta_decay", 0.995)
        self.model.beta = self.beta

        # Gradient clipping
        self.max_grad_norm = cfg.train.get("max_grad_norm", 1.0)

        # Optimizer: only actor parameters (no critic)
        self.optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=self.grpo_lr,
            weight_decay=1e-6,
        )

        # GRPO buffer
        self.buffer = GRPOBuffer(
            group_size=self.group_size,
            device=self.device,
        )

        # Logging
        self.verbose = cfg.train.get("verbose", False)

        log.info(
            f"Initialized GRPO Drifting agent: group_size={self.group_size}, "
            f"lr={self.grpo_lr}, beta={self.beta}, epsilon={self.model.epsilon}, "
            f"update_epochs={self.update_epochs}"
        )

    def collect_group_trajectories(self):
        """
        Collect G trajectories from the same initial state.

        If use_homogeneous_reset is True, all G trajectories share the
        same environment seed for deterministic initial state. Otherwise,
        the G parallel environments are treated as a single group
        (approximate GRPO).
        """
        self.buffer.clear()
        self.model.eval()

        if self.use_homogeneous_reset:
            # Strict homogeneous reset: same seed for all G trajectories
            seed = np.random.randint(0, 100000)

            for g in range(self.group_size):
                # Reset environment to the same initial state
                self.venv.seed([seed] * self.n_envs)
                obs_venv = self.reset_env_all()

                traj_obs = []
                traj_acts = []
                traj_log_probs = []
                episodic_returns = np.zeros(self.n_envs)
                done_flags = np.zeros(self.n_envs, dtype=bool)

                for step in range(self.n_steps):
                    with torch.no_grad():
                        cond = {
                            "state": torch.from_numpy(obs_venv["state"])
                            .float()
                            .to(self.device)
                        }
                        actions, log_probs = self.model.actor.sample_action(cond)

                    action_np = actions[:, :self.act_steps].cpu().numpy()
                    next_obs, rewards, terminated, truncated, info = self.venv.step(
                        action_np
                    )

                    # Store transitions for non-done environments
                    traj_obs.append(
                        torch.from_numpy(obs_venv["state"]).float()
                    )
                    traj_acts.append(actions.cpu())
                    traj_log_probs.append(log_probs.cpu())

                    episodic_returns += rewards * (~done_flags)
                    done_flags |= terminated | truncated
                    obs_venv = next_obs

                    if done_flags.all():
                        break

                # Store each environment's trajectory separately
                obs_stack = torch.stack(traj_obs, dim=0)   # (T, n_envs, ...)
                act_stack = torch.stack(traj_acts, dim=0)   # (T, n_envs, ...)
                lp_stack = torch.stack(traj_log_probs, dim=0)  # (T, n_envs)

                for env_idx in range(self.n_envs):
                    self.buffer.add_trajectory(
                        obs_seq=obs_stack[:, env_idx],
                        act_seq=act_stack[:, env_idx],
                        log_prob_seq=lp_stack[:, env_idx],
                        episodic_return=float(episodic_returns[env_idx]),
                    )
        else:
            # Approximate GRPO: treat parallel environments as one group
            obs_venv = self.reset_env_all()

            traj_obs = []
            traj_acts = []
            traj_log_probs = []
            episodic_returns = np.zeros(self.n_envs)
            done_flags = np.zeros(self.n_envs, dtype=bool)

            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    actions, log_probs = self.model.actor.sample_action(cond)

                action_np = actions[:, :self.act_steps].cpu().numpy()
                next_obs, rewards, terminated, truncated, info = self.venv.step(
                    action_np
                )

                traj_obs.append(
                    torch.from_numpy(obs_venv["state"]).float()
                )
                traj_acts.append(actions.cpu())
                traj_log_probs.append(log_probs.cpu())

                episodic_returns += rewards * (~done_flags)
                done_flags |= terminated | truncated
                obs_venv = next_obs

                if done_flags.all():
                    break

            obs_stack = torch.stack(traj_obs, dim=0)
            act_stack = torch.stack(traj_acts, dim=0)
            lp_stack = torch.stack(traj_log_probs, dim=0)

            for env_idx in range(self.n_envs):
                self.buffer.add_trajectory(
                    obs_seq=obs_stack[:, env_idx],
                    act_seq=act_stack[:, env_idx],
                    log_prob_seq=lp_stack[:, env_idx],
                    episodic_return=float(episodic_returns[env_idx]),
                )

        self.buffer.summarize_episode_reward()

    def update_policy(self):
        """
        Perform GRPO policy update using collected group trajectories.

        Computes group-normalized advantages and runs multiple epochs
        of minibatch gradient descent on the clipped surrogate + KL loss.
        No critic is trained.
        """
        self.model.train()

        all_obs, all_actions, all_old_lps, all_advantages = (
            self.buffer.make_dataset(device=self.device)
        )
        N = all_actions.shape[0]

        total_metrics = {}
        num_updates = 0
        last_loss = 0.0

        for epoch in range(self.update_epochs):
            # Shuffle data
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.grpo_batch_size):
                end = min(start + self.grpo_batch_size, N)
                idx = perm[start:end]

                batch_obs = {"state": all_obs["state"][idx]}
                batch_actions = all_actions[idx]
                batch_advantages = all_advantages[idx]
                batch_old_lps = all_old_lps[idx]

                loss, metrics = self.model.compute_loss(
                    batch_obs, batch_actions, batch_advantages, batch_old_lps
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.actor.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()

                last_loss = loss.item()

                # Accumulate metrics
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
                num_updates += 1

        # Average metrics
        avg_metrics = {
            k: v / max(num_updates, 1) for k, v in total_metrics.items()
        }
        avg_metrics["loss"] = last_loss
        avg_metrics["num_updates"] = num_updates

        return avg_metrics

    def update_beta(self):
        """Decay the KL penalty coefficient with a floor."""
        self.beta = max(self.beta * self.beta_decay, self.beta_min)
        self.model.beta = self.beta

    def run(self):
        """
        Main GRPO training loop.

        Each iteration:
        1. Collect G group trajectories (homogeneous or approximate reset)
        2. Compute group-normalized advantages (no critic)
        3. Update policy with clipped surrogate + KL penalty
        4. Decay KL penalty coefficient
        """
        log.info(
            f"Starting GRPO Drifting Policy fine-tuning "
            f"(group_size={self.group_size}, no Critic)"
        )

        train_itr_pbar = tqdm(
            total=self.n_train_itr,
            desc="GRPO Training Iterations",
            unit="itr",
            dynamic_ncols=True,
            ascii=True,
            initial=self.itr,
        )

        while self.itr < self.n_train_itr:
            # 1. Collect group trajectories
            self.collect_group_trajectories()

            # 2. Policy update (no critic)
            metrics = self.update_policy()

            # 3. Decay KL penalty
            self.update_beta()

            # 4. Logging
            if self.verbose or self.itr % self.log_freq == 0:
                log.info(
                    f"GRPO itr={self.itr}/{self.n_train_itr}, "
                    f"avg_reward={self.buffer.avg_episode_reward:.2f}, "
                    f"success_rate={self.buffer.success_rate*100:.1f}%, "
                    f"policy_loss={metrics.get('policy_loss', 0):.4f}, "
                    f"kl_div={metrics.get('kl_div', 0):.4f}, "
                    f"approx_kl={metrics.get('approx_kl', 0):.2e}, "
                    f"clipfrac={metrics.get('clipfrac', 0):.3f}, "
                    f"beta={self.beta:.4f}, "
                    f"log_std={metrics.get('log_std', 0):.3f}"
                )

            if self.use_wandb:
                import wandb
                wandb.log(
                    {
                        "grpo/avg_reward": self.buffer.avg_episode_reward,
                        "grpo/success_rate": self.buffer.success_rate,
                        "grpo/policy_loss": metrics.get("policy_loss", 0),
                        "grpo/kl_div": metrics.get("kl_div", 0),
                        "grpo/approx_kl": metrics.get("approx_kl", 0),
                        "grpo/clipfrac": metrics.get("clipfrac", 0),
                        "grpo/ratio": metrics.get("ratio", 0),
                        "grpo/beta": self.beta,
                        "grpo/log_std": metrics.get("log_std", 0),
                        "grpo/num_trajectories": len(self.buffer.group_returns),
                    },
                    step=self.itr,
                )

            train_itr_pbar.set_postfix({
                "reward": f"{self.buffer.avg_episode_reward:.2f}",
                "success": f"{self.buffer.success_rate*100:.1f}%",
                "kl": f"{metrics.get('approx_kl', 0):.2e}",
                "beta": f"{self.beta:.3f}",
            })
            train_itr_pbar.update(1)

            # 5. Save checkpoint
            if self.itr % self.save_model_freq == 0:
                self.save_model()

            self.itr += 1

        train_itr_pbar.close()
        self.save_model()
        log.info("GRPO Drifting Policy fine-tuning completed!")
