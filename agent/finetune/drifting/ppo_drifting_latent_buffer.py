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
On-policy PPO buffer for Latent-PPO Drifting.

Stores per-step rollout data: (obs, z, action, reward, done, value, logprob_z).
Computes GAE advantages and returns for PPO updates.
"""

import numpy as np
import torch


class PPODriftingLatentBuffer:
    """On-policy buffer with GAE for latent-PPO drifting.

    Data layout (all stored as numpy arrays):
        obs_state: (n_steps, n_envs, cond_steps, obs_dim)
        z:         (n_steps, n_envs, horizon_steps, action_dim)
        actions:   (n_steps, n_envs, horizon_steps, action_dim)
        rewards:   (n_steps, n_envs)
        dones:     (n_steps, n_envs)  -- terminated flags
        values:    (n_steps, n_envs)
        logprobs:  (n_steps, n_envs)
        firsts:    (n_steps + 1, n_envs)  -- episode start flags for reward accounting

    Optionally stores 'rgb' observations for image-conditioned policies.
    """

    def __init__(
        self,
        n_steps,
        n_envs,
        cond_steps,
        obs_dim,
        horizon_steps,
        action_dim,
        device,
        gamma=0.99,
        gae_lambda=0.95,
        use_image_obs=False,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.cond_steps = cond_steps
        self.obs_dim = obs_dim
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_image_obs = use_image_obs

        self.reset()

    def reset(self):
        """Allocate storage arrays for a new rollout."""
        self.obs_state = np.zeros(
            (self.n_steps, self.n_envs, self.cond_steps, self.obs_dim)
        )
        self.z = np.zeros(
            (self.n_steps, self.n_envs, self.horizon_steps, self.action_dim)
        )
        self.actions = np.zeros(
            (self.n_steps, self.n_envs, self.horizon_steps, self.action_dim)
        )
        self.rewards = np.zeros((self.n_steps, self.n_envs))
        self.dones = np.zeros((self.n_steps, self.n_envs))
        self.values = np.zeros((self.n_steps, self.n_envs))
        self.logprobs = np.zeros((self.n_steps, self.n_envs))
        self.firsts = np.zeros((self.n_steps + 1, self.n_envs))

        if self.use_image_obs:
            self.obs_rgb = []  # list of (n_envs, T, C, H, W) arrays
        self.ptr = 0

    def add(
        self,
        obs,
        z,
        action,
        reward,
        done,
        value,
        logprob,
    ):
        """Add one time-step of rollout data.

        Args:
            obs: dict with 'state' (n_envs, cond_steps, obs_dim) and optionally 'rgb'
            z: (n_envs, horizon_steps, action_dim)
            action: (n_envs, horizon_steps, action_dim)
            reward: (n_envs,)
            done: (n_envs,)
            value: (n_envs,)
            logprob: (n_envs,)
        """
        t = self.ptr
        self.obs_state[t] = obs["state"]
        self.z[t] = z
        self.actions[t] = action
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self.logprobs[t] = logprob

        if self.use_image_obs and "rgb" in obs:
            self.obs_rgb.append(obs["rgb"])

        self.ptr += 1

    def compute_gae(self, last_values, last_dones, reward_scale_const=1.0):
        """Compute GAE advantages and returns.

        Args:
            last_values: (n_envs,) V(s_{T+1}) for bootstrap
            last_dones: (n_envs,) terminal flags for last step
            reward_scale_const: constant reward multiplier

        Returns:
            advantages: (n_steps, n_envs)
            returns: (n_steps, n_envs)
        """
        advantages = np.zeros_like(self.rewards)
        lastgaelam = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
                next_nonterminal = 1.0 - last_dones
            else:
                next_values = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t] * reward_scale_const
                + self.gamma * next_values * next_nonterminal
                - self.values[t]
            )
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
            )

        returns = advantages + self.values
        return advantages, returns

    def get_tensors(self, advantages, returns):
        """Flatten (n_steps, n_envs) -> (n_steps * n_envs) and convert to tensors.

        Returns a dict of tensors ready for mini-batch PPO updates.
        """
        total = self.n_steps * self.n_envs
        result = {
            "obs_state": torch.from_numpy(
                self.obs_state.reshape(total, self.cond_steps, self.obs_dim)
            ).float().to(self.device),
            "z": torch.from_numpy(
                self.z.reshape(total, self.horizon_steps, self.action_dim)
            ).float().to(self.device),
            "actions": torch.from_numpy(
                self.actions.reshape(total, self.horizon_steps, self.action_dim)
            ).float().to(self.device),
            "returns": torch.from_numpy(returns.reshape(total)).float().to(self.device),
            "values": torch.from_numpy(
                self.values.reshape(total)
            ).float().to(self.device),
            "advantages": torch.from_numpy(
                advantages.reshape(total)
            ).float().to(self.device),
            "logprobs": torch.from_numpy(
                self.logprobs.reshape(total)
            ).float().to(self.device),
        }

        if self.use_image_obs and self.obs_rgb:
            rgb_arr = np.stack(self.obs_rgb, axis=0)  # (n_steps, n_envs, T, C, H, W)
            result["obs_rgb"] = torch.from_numpy(
                rgb_arr.reshape(total, *rgb_arr.shape[2:])
            ).float().to(self.device)

        return result
