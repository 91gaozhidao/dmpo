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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from model.drifting.drifting import DriftingPolicy

log = logging.getLogger(__name__)

# Module-level constants for numerical stability
LOG_2 = np.log(2)
TANH_CLIP_THRESHOLD = 0.999999
JACOBIAN_EPS = 1e-6


def _tanh_jacobian_correction(u):
    """
    Compute the log-determinant of the Jacobian for the tanh squashing.

    Uses the numerically stable formula:
        log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2*u))

    Args:
        u: (*, D) unbounded pre-tanh actions

    Returns:
        correction: (*, D) per-dimension Jacobian correction
    """
    return 2 * (LOG_2 - u - F.softplus(-2 * u))


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
        Compute log-probability of actions under the Tanh-Normal policy.

        Applies inverse tanh (atanh) to recover unbounded actions, then
        computes Normal log-probability with Jacobian correction:
            log pi(a) = log N(atanh(a); mu, sigma) - log(1 - a^2 + eps)

        Args:
            cond: dict with 'state' key
            action: (B, Ta, Da) actions in (-1, 1)

        Returns:
            log_prob: (B,) per-sample log-probability
        """
        dist, _ = self.get_distribution(cond)
        B = action.shape[0]
        action_flat = action.view(B, -1)  # (B, Ta*Da)

        # Inverse tanh to recover unbounded action
        action_clipped = torch.clamp(action_flat, -TANH_CLIP_THRESHOLD, TANH_CLIP_THRESHOLD)
        u = torch.atanh(action_clipped)

        # Normal log-prob with Jacobian correction for tanh squashing
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1 - action_clipped.pow(2) + JACOBIAN_EPS)
        log_prob = log_prob.sum(dim=-1)  # (B,)
        return log_prob

    def get_action_and_log_prob(self, cond: dict):
        """
        Sample an action via Tanh-Normal and return the corrected log-probability.

        Uses reparameterized sampling (rsample) for gradient flow.

        Args:
            cond: dict with 'state' key

        Returns:
            action: (B, Ta, Da) sampled action in [-1, 1]
            log_prob: (B,) Jacobian-corrected log-probability
        """
        dist, mean = self.get_distribution(cond)
        u = dist.rsample()  # (B, Ta*Da)
        action_flat = torch.tanh(u)  # bounded to [-1, 1]

        # Jacobian correction (numerically stable):
        # log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2*u))
        log_prob = dist.log_prob(u)
        log_prob -= _tanh_jacobian_correction(u)
        log_prob = log_prob.sum(dim=-1)  # (B,)

        B = mean.shape[0]
        action = action_flat.view(B, self.horizon_steps, self.action_dim)
        return action, log_prob

    def sample_action(self, cond: dict):
        """
        Sample an action via Tanh-Normal for environment interaction.

        Uses non-reparameterized sampling (sample) since this is typically
        called under torch.no_grad() during trajectory collection.

        Args:
            cond: dict with 'state' key

        Returns:
            action: (B, Ta, Da) sampled action in [-1, 1]
            log_prob: (B,) Jacobian-corrected log-probability
        """
        dist, mean = self.get_distribution(cond)
        u = dist.sample()  # (B, Ta*Da)
        action_flat = torch.tanh(u)  # bounded to [-1, 1]

        # Jacobian correction (numerically stable)
        log_prob = dist.log_prob(u)
        log_prob -= _tanh_jacobian_correction(u)
        log_prob = log_prob.sum(dim=-1)  # (B,)

        B = mean.shape[0]
        action = action_flat.view(B, self.horizon_steps, self.action_dim)
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
    where KL is the analytical KL divergence KL(N_curr || N_ref)
    between the pre-tanh Gaussian distributions of the current and
    reference policies.
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
        Compute the Continuous GRPO loss with analytical KL divergence.

        Uses analytical KL(N_curr || N_ref) for the pre-tanh Gaussian
        distributions, eliminating sampling variance in the KL penalty.

        Args:
            obs: dict with 'state' key, (B, To, Do) observations
            actions: (B, Ta, Da) sampled actions
            advantages: (B,) group-normalized advantages
            old_log_probs: (B,) log-probabilities from the sampling policy

        Returns:
            loss: scalar GRPO loss
            metrics: dict with diagnostic values
        """
        # Current policy distribution parameters and log-probability
        current_mean, current_std = self.actor(obs)
        current_log_probs = self.actor.get_log_prob(obs, actions)

        # Reference policy distribution parameters
        with torch.no_grad():
            ref_mean, ref_std = self.actor_ref(obs)

        # Policy ratio: pi_theta(a|s) / pi_old(a|s)
        ratio = torch.exp(current_log_probs - old_log_probs)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
        ) * advantages
        policy_loss = -torch.min(surr1, surr2)

        # Analytical KL divergence: KL(N_curr || N_ref)
        # For diagonal Gaussian distributions with independent dimensions
        B = current_mean.shape[0]
        curr_mean_flat = current_mean.view(B, -1)   # (B, Ta*Da)
        ref_mean_flat = ref_mean.view(B, -1)         # (B, Ta*Da)
        curr_std_expanded = current_std.expand(B, -1)  # (B, Ta*Da)
        ref_std_expanded = ref_std.expand(B, -1)       # (B, Ta*Da)

        var_curr = curr_std_expanded.pow(2)
        var_ref = ref_std_expanded.pow(2)

        kl_div = (
            torch.log(ref_std_expanded / curr_std_expanded)
            + (var_curr + (curr_mean_flat - ref_mean_flat).pow(2)) / (2 * var_ref)
            - 0.5
        )
        kl_div_sum = kl_div.sum(dim=-1)  # Sum over action dimensions

        # Total GRPO loss
        loss = (policy_loss + self.beta * kl_div_sum).mean()

        # Diagnostic metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            clipfrac = (
                (ratio - 1.0).abs() > self.epsilon
            ).float().mean().item()

        metrics = {
            "policy_loss": policy_loss.mean().item(),
            "kl_div": kl_div_sum.mean().item(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "ratio": ratio.mean().item(),
            "log_std": self.actor.log_std.mean().item(),
        }

        return loss, metrics
