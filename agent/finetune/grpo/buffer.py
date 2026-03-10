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
Group-level experience buffer for Continuous GRPO.

Stores G trajectories per group, each sharing the same initial state.
Computes group-normalized advantages via Z-score normalization of
episodic returns — no Critic/Value network required.
"""

import logging
import torch
import numpy as np

log = logging.getLogger(__name__)

# Threshold below which group returns are considered identical (no preference)
ADVANTAGE_STD_THRESHOLD = 1e-6


class GRPOBuffer:
    """
    Buffer for GRPO that stores group-level trajectories and computes
    advantages via intra-group Z-score normalization.

    Each group consists of G trajectories sampled from the same initial
    state. The advantage for trajectory i in a group is:
        A_i = (R_i - mean(R)) / (std(R) + eps)
    where R are the episodic returns within the group.
    """

    def __init__(self, group_size: int, device: torch.device = None):
        """
        Args:
            group_size: Number of trajectories per group (G)
            device: Torch device for tensor operations
        """
        self.group_size = group_size
        self.device = device or torch.device("cpu")
        self.clear()

        # Episode-level tracking for logging
        self.episode_rewards_all = []
        self.avg_episode_reward = 0.0
        self.success_rate = 0.0

    def clear(self):
        """Reset all trajectory storage for a new collection round."""
        self.obs = []           # List of (T_i, cond_steps, obs_dim) tensors
        self.actions = []       # List of (T_i, horizon_steps, action_dim) tensors
        self.old_log_probs = [] # List of (T_i,) tensors
        self.group_returns = [] # List of scalar episodic returns

    def add_trajectory(self, obs_seq, act_seq, log_prob_seq, episodic_return):
        """
        Add one complete trajectory to the buffer.

        Args:
            obs_seq: (T, cond_steps, obs_dim) observation sequence tensor
            act_seq: (T, horizon_steps, action_dim) action sequence tensor
            log_prob_seq: (T,) log-probability sequence tensor
            episodic_return: scalar cumulative return for this trajectory
        """
        self.obs.append(obs_seq)
        self.actions.append(act_seq)
        self.old_log_probs.append(log_prob_seq)
        self.group_returns.append(episodic_return)

        # Track for logging
        self.episode_rewards_all.append(episodic_return)

    def compute_group_advantages(self):
        """
        Compute GRPO advantages via intra-group Z-score normalization.

        Groups trajectories by group_size and normalizes returns within
        each group independently. If the total number of trajectories is
        not a multiple of group_size, the remaining are normalized as a
        single group.

        Returns:
            advantages: (num_trajectories,) tensor of normalized advantages
        """
        returns = torch.tensor(self.group_returns, dtype=torch.float32)
        num_traj = len(returns)

        advantages = torch.zeros(num_traj)

        # Process complete groups
        num_complete_groups = num_traj // self.group_size
        for g in range(num_complete_groups):
            start = g * self.group_size
            end = start + self.group_size
            group_returns = returns[start:end]

            mean_r = group_returns.mean()
            std_r = group_returns.std(unbiased=False)
            if std_r < ADVANTAGE_STD_THRESHOLD:
                # Zero-variance protection: when all returns in a group are
                # identical (e.g. sparse reward with all-zero returns), set
                # advantages to zero to prevent division-by-zero errors.
                log.debug(
                    f"Group {g}: zero-variance detected (std={std_r:.2e}), "
                    f"setting advantages to 0.0"
                )
                advantages[start:end] = 0.0
            else:
                advantages[start:end] = (group_returns - mean_r) / (std_r + ADVANTAGE_STD_THRESHOLD)

        # Handle remainder (if any)
        remainder_start = num_complete_groups * self.group_size
        if remainder_start < num_traj:
            remainder_returns = returns[remainder_start:]
            mean_r = remainder_returns.mean()
            std_r = remainder_returns.std(unbiased=False)
            if std_r < ADVANTAGE_STD_THRESHOLD:
                # Zero-variance protection for remainder group
                log.debug(
                    f"Remainder group: zero-variance detected (std={std_r:.2e}), "
                    f"setting advantages to 0.0"
                )
                advantages[remainder_start:] = 0.0
            else:
                advantages[remainder_start:] = (
                    (remainder_returns - mean_r) / (std_r + ADVANTAGE_STD_THRESHOLD)
                )

        return advantages

    def make_dataset(self, device=None):
        """
        Flatten all trajectories into a single dataset for batch updates.

        Broadcasts trajectory-level advantages to step-level so each
        transition inherits the advantage of its parent trajectory.

        Args:
            device: Target device (defaults to self.device)

        Returns:
            all_obs: dict with 'state' key, (N, cond_steps, obs_dim)
            all_actions: (N, horizon_steps, action_dim)
            all_old_log_probs: (N,)
            all_advantages: (N,)
        """
        device = device or self.device
        advantages = self.compute_group_advantages()

        all_obs_list = []
        all_actions_list = []
        all_log_probs_list = []
        all_advs_list = []

        for i in range(len(self.obs)):
            traj_len = len(self.obs[i])
            all_obs_list.append(self.obs[i])
            all_actions_list.append(self.actions[i])
            all_log_probs_list.append(self.old_log_probs[i])
            # Broadcast trajectory-level advantage to every step
            all_advs_list.append(advantages[i].expand(traj_len))

        all_obs_tensor = torch.cat(all_obs_list, dim=0).to(device)
        all_actions = torch.cat(all_actions_list, dim=0).to(device)
        all_old_log_probs = torch.cat(all_log_probs_list, dim=0).to(device)
        all_advantages = torch.cat(all_advs_list, dim=0).to(device)

        all_obs = {"state": all_obs_tensor}
        return all_obs, all_actions, all_old_log_probs, all_advantages

    def summarize_episode_reward(self):
        """Compute summary statistics from collected trajectories."""
        if self.episode_rewards_all:
            recent = self.episode_rewards_all[-100:]
            self.avg_episode_reward = np.mean(recent)
            self.success_rate = np.mean(
                [1.0 if r > 0 else 0.0 for r in recent]
            )
