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
DPRO (Direct Preference Reward Optimization) fine-tuning for Drifting Policy.

Drifting Policy naturally supports preference-based learning through its
positive and negative drift fields. DPRO directly optimizes the policy using
trajectory preferences without a separate reward model or critic network.

The optimization leverages compute_V(x, y_pos, y_neg):
    target = x + (V_pos - V_neg)
    loss = MSE(x, target)

where y_pos and y_neg are high-reward and low-reward trajectories collected
from online environment interaction.
"""

import os
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from agent.finetune.train_agent import TrainAgent
from model.drifting.drifting import DriftingPolicy


class DPROBuffer:
    """
    Buffer for DPRO that stores trajectories with their cumulative returns.
    Provides methods to construct positive/negative sample pairs based on 
    return ranking.
    """

    def __init__(self, capacity, horizon_steps, action_dim, obs_dim, cond_steps, device):
        self.capacity = capacity
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps

        # Storage
        self.actions = []
        self.observations = []
        self.returns = []

        # Episode-level tracking
        self.current_episode_rewards = []
        self.current_episode_actions = []
        self.current_episode_obs = []
        self.episode_rewards_all = []
        self.avg_episode_reward = 0.0
        self.success_rate = 0.0

    def reset(self):
        """Reset the buffer."""
        self.actions = []
        self.observations = []
        self.returns = []
        self.current_episode_rewards = []
        self.current_episode_actions = []
        self.current_episode_obs = []

    def add_transition(self, obs, action, reward):
        """
        Add a single transition to the current episode.

        Args:
            obs: dict with 'state' key, (n_envs, cond_steps, obs_dim)
            action: (n_envs, horizon_steps, action_dim)
            reward: (n_envs,)
        """
        self.current_episode_actions.append(action)
        self.current_episode_obs.append(obs)
        self.current_episode_rewards.append(reward)

    def finalize_episodes(self):
        """
        Finalize collected episodes: compute cumulative returns and store
        action-observation-return triplets.
        """
        if len(self.current_episode_rewards) == 0:
            return

        # Stack episode data
        rewards = np.stack(self.current_episode_rewards, axis=0)  # (T, n_envs)
        n_envs = rewards.shape[1]

        # Compute cumulative return per environment
        episode_returns = rewards.sum(axis=0)  # (n_envs,)

        self.episode_rewards_all.extend(episode_returns.tolist())
        self.avg_episode_reward = np.mean(self.episode_rewards_all[-100:]) if self.episode_rewards_all else 0.0

        # For each environment, store the last action-obs pair with its return
        for env_idx in range(n_envs):
            for t in range(len(self.current_episode_actions)):
                action = self.current_episode_actions[t][env_idx]
                obs_state = self.current_episode_obs[t]["state"][env_idx]
                ret = episode_returns[env_idx]

                self.actions.append(action)
                self.observations.append(obs_state)
                self.returns.append(ret)

        # Trim to capacity
        if len(self.actions) > self.capacity:
            excess = len(self.actions) - self.capacity
            self.actions = self.actions[excess:]
            self.observations = self.observations[excess:]
            self.returns = self.returns[excess:]

        # Reset episode buffers
        self.current_episode_rewards = []
        self.current_episode_actions = []
        self.current_episode_obs = []

    def get_preference_pairs(self, batch_size, percentile_threshold=50.0):
        """
        Construct positive/negative sample pairs based on return ranking.

        Actions from trajectories with returns above the percentile threshold
        are treated as y_pos; those below as y_neg.

        Args:
            batch_size: Number of pairs to return
            percentile_threshold: Percentile cutoff for pos/neg split

        Returns:
            obs: (batch_size, cond_steps, obs_dim) tensor
            y_pos: (batch_size, horizon_steps, action_dim) tensor 
            y_neg: (batch_size, horizon_steps, action_dim) tensor
        """
        if len(self.returns) < 2 * batch_size:
            return None, None, None

        returns_arr = np.array(self.returns)
        threshold = np.percentile(returns_arr, percentile_threshold)

        # Split indices
        pos_indices = np.where(returns_arr >= threshold)[0]
        neg_indices = np.where(returns_arr < threshold)[0]

        if len(pos_indices) < batch_size or len(neg_indices) < batch_size:
            return None, None, None

        # Sample
        pos_sample_idx = np.random.choice(pos_indices, size=batch_size, replace=True)
        neg_sample_idx = np.random.choice(neg_indices, size=batch_size, replace=True)

        obs_batch = torch.stack(
            [torch.as_tensor(self.observations[i], dtype=torch.float32) for i in pos_sample_idx]
        ).to(self.device)
        y_pos_batch = torch.stack(
            [torch.as_tensor(self.actions[i], dtype=torch.float32) for i in pos_sample_idx]
        ).to(self.device)
        y_neg_batch = torch.stack(
            [torch.as_tensor(self.actions[i], dtype=torch.float32) for i in neg_sample_idx]
        ).to(self.device)

        return obs_batch, y_pos_batch, y_neg_batch

    def summarize_episode_reward(self):
        """Compute summary statistics from completed episodes."""
        if self.episode_rewards_all:
            recent = self.episode_rewards_all[-100:]
            self.avg_episode_reward = np.mean(recent)
            self.success_rate = np.mean([1.0 if r > 0 else 0.0 for r in recent])


class TrainDPRODriftingAgent(TrainAgent):
    """
    DPRO fine-tuning agent for Drifting Policy.

    Uses online environment interaction to collect trajectories, ranks them
    by cumulative return, and constructs preference pairs for direct
    preference-based policy optimization via the drifting field.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: DriftingPolicy

        # DPRO-specific configuration
        self.dpro_batch_size = cfg.train.get("dpro_batch_size", 256)
        self.dpro_update_interval = cfg.train.get("dpro_update_interval", 1000)
        self.dpro_gradient_steps = cfg.train.get("dpro_gradient_steps", 10)
        self.dpro_percentile = cfg.train.get("dpro_percentile", 50.0)
        self.dpro_buffer_capacity = cfg.train.get("dpro_buffer_capacity", 100000)
        self.dpro_lr = cfg.train.get("dpro_lr", 1e-4)

        # Initialize optimizer for the model
        self.optimizer = torch.optim.AdamW(
            self.model.network.parameters(),
            lr=self.dpro_lr,
            weight_decay=1e-6,
        )

        # Initialize DPRO buffer
        self.dpro_buffer = DPROBuffer(
            capacity=self.dpro_buffer_capacity,
            horizon_steps=self.model.horizon_steps,
            action_dim=self.model.action_dim,
            obs_dim=self.model.obs_dim,
            cond_steps=cfg.cond_steps if hasattr(cfg, 'cond_steps') else 1,
            device=self.device,
        )

        log.info(
            f"Initialized DPRO Drifting agent: batch_size={self.dpro_batch_size}, "
            f"update_interval={self.dpro_update_interval}, "
            f"gradient_steps={self.dpro_gradient_steps}"
        )

    def dpro_update(self, verbose=True):
        """
        Perform DPRO gradient update using preference pairs.

        Constructs positive/negative pairs from the buffer, computes the
        drifting field, and updates the policy to align with preferences.
        """
        total_loss = 0.0
        num_updates = 0

        for _ in range(self.dpro_gradient_steps):
            obs, y_pos, y_neg = self.dpro_buffer.get_preference_pairs(
                batch_size=self.dpro_batch_size,
                percentile_threshold=self.dpro_percentile,
            )
            if obs is None:
                if verbose:
                    log.warning("Not enough data in DPRO buffer for preference pairs")
                return 0.0

            cond = {"state": obs}

            # Generate current predictions from noise
            z = torch.randn(
                self.dpro_batch_size,
                self.model.horizon_steps,
                self.model.action_dim,
                device=self.device
            )
            t = torch.ones(self.dpro_batch_size, device=self.device)
            r = torch.zeros(self.dpro_batch_size, device=self.device)
            x = self.model.network(z, t, r, cond)

            # Compute drifting field with positive and negative targets
            V_total = self.model.compute_V(x, y_pos, y_neg, mask_self=False)

            # DPRO target: x + V_total
            target = (x + V_total).detach()

            # Loss: drive the network output toward the drifted target
            loss = F.mse_loss(x, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_updates += 1

        avg_loss = total_loss / max(num_updates, 1)
        if verbose:
            log.info(f"DPRO update: avg_loss={avg_loss:.6f}, num_updates={num_updates}")

        return avg_loss

    def run(self):
        """
        Main training loop for DPRO fine-tuning of Drifting Policy.

        Collects trajectories from the environment, periodically constructs
        preference pairs based on return ranking, and performs DPRO gradient
        updates.
        """
        log.info("Starting DPRO Drifting Policy fine-tuning")

        self.dpro_buffer.reset()
        self.prepare_run()

        total_env_steps = 0
        steps_since_update = 0

        train_itr_pbar = tqdm(
            total=self.n_train_itr,
            desc="DPRO Training Iterations",
            unit="itr",
            dynamic_ncols=True,
            ascii=True,
            initial=self.itr,
        )

        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.model.eval()
            self.reset_env()

            # Data collection phase
            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(self.prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    samples = self.model.sample(cond)
                    action_samples = samples.trajectories

                action_venv = action_samples[:, :self.act_steps].cpu().numpy()
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)

                self.dpro_buffer.add_transition(
                    self.prev_obs_venv, action_samples.cpu().numpy(), reward_venv
                )

                self.prev_obs_venv = obs_venv
                total_env_steps += self.n_envs * self.act_steps
                steps_since_update += self.n_envs * self.act_steps

            # Finalize episodes and compute returns
            self.dpro_buffer.finalize_episodes()
            self.dpro_buffer.summarize_episode_reward()

            # Periodic DPRO update
            if steps_since_update >= self.dpro_update_interval:
                self.model.train()
                dpro_loss = self.dpro_update(verbose=self.verbose)
                steps_since_update = 0

                if self.verbose:
                    log.info(
                        f"DPRO itr={self.itr}, env_steps={total_env_steps}, "
                        f"dpro_loss={dpro_loss:.6f}, "
                        f"avg_reward={self.dpro_buffer.avg_episode_reward:.2f}, "
                        f"success_rate={self.dpro_buffer.success_rate*100:.1f}%"
                    )

            # Logging and checkpointing
            self.log()
            self.save_model()

            if self.itr % self.log_freq == 0:
                train_itr_pbar.set_postfix({
                    'reward': f'{self.dpro_buffer.avg_episode_reward:.2f}',
                    'success': f'{self.dpro_buffer.success_rate*100:.1f}%',
                })
            train_itr_pbar.update(1)

            self.itr += 1

        train_itr_pbar.close()
        log.info("DPRO Drifting Policy fine-tuning completed!")
