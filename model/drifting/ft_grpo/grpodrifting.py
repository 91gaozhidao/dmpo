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
Continuous GRPO (Group Relative Policy Optimization) for Drifting Policy.

GRPO eliminates the Critic/Value network entirely. Instead, it estimates
advantages via group-internal Z-score normalization of episodic returns
collected from G trajectories sampled from the same initial state. A frozen
reference policy provides KL divergence regularization to prevent excessive
policy drift.

For Drifting Policy (1-NFE), the deterministic network output serves as
the Gaussian mean, and a learnable log-std parameter provides exploration.
"""

import copy
import logging
import torch
import torch.nn as nn
from torch.distributions import Normal
from model.drifting.drifting import DriftingPolicy

log = logging.getLogger(__name__)


class NoisyDriftingPolicy(nn.Module):
    """
    Stochastic wrapper around the deterministic Drifting Policy for GRPO.

    The 1-NFE Drifting Policy output is treated as the Gaussian mean.
    An independent learnable log-std parameter provides exploration noise,
    enabling proper log-probability and KL divergence computation required
    by GRPO.
    """

    def __init__(self, drifting_policy: DriftingPolicy, action_dim: int,
                 horizon_steps: int, init_log_std: float = -0.5):
        """
        Args:
            drifting_policy: Pretrained DriftingPolicy instance
            action_dim: Dimension of action space
            horizon_steps: Number of steps in trajectory horizon
            init_log_std: Initial value for log standard deviation
        """
        super().__init__()
        self.drifting_policy = drifting_policy
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        # Independent learnable log-std for the full action space
        self.log_std = nn.Parameter(
            torch.full((horizon_steps * action_dim,), init_log_std)
        )

    def forward(self, cond: dict):
        """
        Compute deterministic action mean and exploration std.

        Args:
            cond: dict with 'state' key for observations

        Returns:
            mean: (B, Ta, Da) deterministic action from Drifting Policy
            std: (Ta*Da,) exploration standard deviation
        """
        mean = self.drifting_policy.sample(cond).trajectories
        std = self.log_std.exp()
        return mean, std

    def get_distribution(self, cond: dict):
        """
        Build the action distribution for the given observations.

        Args:
            cond: dict with 'state' key

        Returns:
            dist: Normal distribution over flattened actions
            mean: (B, Ta, Da) raw action mean
        """
        mean, std = self.forward(cond)
        B = mean.shape[0]
        mean_flat = mean.view(B, -1)  # (B, Ta*Da)
        dist = Normal(mean_flat, std.expand(B, -1))
        return dist, mean

    def get_log_prob(self, cond: dict, action):
        """
        Compute log-probability of actions under the current policy.

        Args:
            cond: dict with 'state' key
            action: (B, Ta, Da) actions

        Returns:
            log_prob: (B,) per-sample log-probability
        """
        dist, _ = self.get_distribution(cond)
        B = action.shape[0]
        action_flat = action.view(B, -1)  # (B, Ta*Da)
        log_prob = dist.log_prob(action_flat).sum(dim=-1)  # (B,)
        return log_prob

    def sample_action(self, cond: dict):
        """
        Sample an action and compute its log-probability.

        Args:
            cond: dict with 'state' key

        Returns:
            action: (B, Ta, Da) sampled action
            log_prob: (B,) log-probability of sampled action
        """
        dist, mean = self.get_distribution(cond)
        action_flat = dist.sample()  # (B, Ta*Da)
        log_prob = dist.log_prob(action_flat).sum(dim=-1)  # (B,)
        B = mean.shape[0]
        action = action_flat.view(B, self.horizon_steps, self.action_dim)
        action = action.clamp(*self.drifting_policy.act_range)
        return action, log_prob


class GRPODrifting(nn.Module):
    """
    Continuous GRPO objective for Drifting Policy.

    Core GRPO features:
    - No Critic/Value network: advantages are computed via group-internal
      Z-score normalization of episodic returns.
    - Frozen reference policy for KL penalty to constrain policy drift.
    - Clipped surrogate policy loss (same form as PPO clip).

    The GRPO loss is:
        L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + beta * KL
    where KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
    """

    def __init__(self, actor: NoisyDriftingPolicy,
                 beta: float = 0.01,
                 epsilon: float = 0.2,
                 act_min: float = -1.0,
                 act_max: float = 1.0):
        """
        Args:
            actor: NoisyDriftingPolicy to optimize
            beta: KL penalty coefficient
            epsilon: PPO-style clipping range
            act_min: Minimum action value for clipping
            act_max: Maximum action value for clipping
        """
        super().__init__()
        self.actor = actor
        self.act_min = act_min
        self.act_max = act_max

        # Frozen reference policy (deep copy with no gradients)
        self.actor_ref = copy.deepcopy(actor)
        for param in self.actor_ref.parameters():
            param.requires_grad = False

        self.beta = beta
        self.epsilon = epsilon

    def compute_loss(self, obs, actions, advantages, old_log_probs):
        """
        Compute the Continuous GRPO loss.

        Args:
            obs: dict with 'state' key, (B, To, Do) observations
            actions: (B, Ta, Da) sampled actions
            advantages: (B,) group-normalized advantages
            old_log_probs: (B,) log-probabilities from the sampling policy

        Returns:
            loss: scalar GRPO loss
            metrics: dict with diagnostic values
        """
        # Current policy log-probability
        current_log_probs = self.actor.get_log_prob(obs, actions)

        # Reference policy log-probability (for KL penalty)
        with torch.no_grad():
            ref_log_probs = self.actor_ref.get_log_prob(obs, actions)

        # Policy ratio: pi_theta(a|s) / pi_old(a|s)
        ratio = torch.exp(current_log_probs - old_log_probs)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
        ) * advantages
        policy_loss = -torch.min(surr1, surr2)

        # KL penalty: approximate KL(pi_ref || pi_theta)
        # Using the unbiased estimator: (pi_ref/pi_theta) - log(pi_ref/pi_theta) - 1
        log_ratio_ref = ref_log_probs - current_log_probs
        kl_div = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

        # Total GRPO loss
        loss = (policy_loss + self.beta * kl_div).mean()

        # Diagnostic metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            clipfrac = (
                (ratio - 1.0).abs() > self.epsilon
            ).float().mean().item()

        metrics = {
            "policy_loss": policy_loss.mean().item(),
            "kl_div": kl_div.mean().item(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "ratio": ratio.mean().item(),
            "log_std": self.actor.log_std.mean().item(),
        }

        return loss, metrics
