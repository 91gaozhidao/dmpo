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
Latent-PPO Drifting: on-policy RL via a learned latent policy over a frozen
drifting generator.

Architecture:
    z ~ pi_phi(z | s)          (diagonal Gaussian, same shape as noise input)
    a = f_theta(s, z)          (frozen drifting generator, 1-NFE)

The PPO ratio is computed entirely in the latent z-space, sidestepping the
problem that drifting's implicit action distribution has no tractable
log p(a|s).  The generator f_theta is frozen in the first version so that
the drifting pretrain quality is preserved while validating that
latent-space RL can work.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from model.drifting.ft_qguided.qguided_drifting import (
    load_drifting_policy_from_checkpoint,
    _is_compatible_drifting_buffer_key,
)
from model.drifting.drifting import DriftingPolicy

log = logging.getLogger(__name__)


class LatentPPODrifting(nn.Module):
    """Latent-PPO wrapper: frozen drifting generator + learned z-policy + V(s) critic.

    During rollout:
        1. z-policy outputs mu(s), log_std(s)
        2. z = mu + std * eps  (reparameterized)
        3. action = f_theta(s, z)  (frozen generator)
        4. Store (obs, z, action, reward, done, value, old_logprob_z)

    During PPO update:
        - Recompute logprob_z from z-policy given stored z
        - Standard clipped PPO loss + entropy bonus on z-policy
        - V(s) critic updated with MSE on GAE returns
    """

    def __init__(
        self,
        device,
        latent_policy,
        critic,
        act_dim,
        horizon_steps,
        act_steps,
        act_min,
        act_max,
        obs_dim,
        cond_steps,
        actor_policy_path=None,
        seed=0,
        use_ema_checkpoint=True,
        clip_ploss_coef=0.2,
        clip_vloss_coef=None,
        # Generator can also be passed directly (for testing)
        generator=None,
        policy=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_steps = act_steps
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        self.act_min = act_min
        self.act_max = act_max
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_vloss_coef = clip_vloss_coef

        # ── Build or load the frozen drifting generator ──
        if generator is not None:
            # Directly provided (e.g., for testing)
            self.generator = generator.to(self.device)
        elif policy is not None:
            # Build from a backbone network config (Hydra-instantiated)
            if isinstance(policy, DriftingPolicy):
                self.generator = policy.to(self.device)
            else:
                self.generator = DriftingPolicy(
                    network=policy,
                    device=self.device,
                    horizon_steps=horizon_steps,
                    action_dim=act_dim,
                    act_min=act_min,
                    act_max=act_max,
                    obs_dim=obs_dim,
                    max_denoising_steps=1,
                    seed=seed,
                ).to(self.device)
        else:
            raise ValueError(
                "LatentPPODrifting requires either 'generator' or 'policy' to build "
                "the drifting generator backbone."
            )

        # Load pretrained weights if path given
        if actor_policy_path:
            loaded = load_drifting_policy_from_checkpoint(
                actor_policy_path,
                device=self.device,
                use_ema=use_ema_checkpoint,
            )
            missing, unexpected = self.generator.load_state_dict(
                loaded.state_dict(), strict=False,
            )
            incompatible_missing = [
                k for k in missing if not _is_compatible_drifting_buffer_key(k)
            ]
            incompatible_unexpected = [
                k for k in unexpected if not _is_compatible_drifting_buffer_key(k)
            ]
            if incompatible_missing or incompatible_unexpected:
                raise ValueError(
                    "Failed to load drifting generator weights. "
                    f"Missing: {incompatible_missing}, Unexpected: {incompatible_unexpected}"
                )
            log.info("Loaded pretrained drifting generator from %s", actor_policy_path)

        # Freeze generator
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

        # ── Latent policy (trainable) ──
        self.latent_policy = latent_policy.to(self.device)

        # ── Value critic (trainable) ──
        self.critic = critic.to(self.device)

        self._report_params()

    def _report_params(self):
        gen_params = sum(p.numel() for p in self.generator.parameters()) / 1e6
        lat_params = sum(p.numel() for p in self.latent_policy.parameters()) / 1e6
        cri_params = sum(p.numel() for p in self.critic.parameters()) / 1e6
        log.info(
            "LatentPPODrifting params: Generator %.3fM (frozen) | "
            "LatentPolicy %.3fM | Critic %.3fM",
            gen_params, lat_params, cri_params,
        )

    # ──────────────────────────────────────────────────────────────
    #  Inference
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        cond: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample actions for environment interaction.

        Args:
            cond: observation dict with 'state' key (and optionally 'rgb').
            deterministic: if True, use z = mu(s) (no sampling noise).

        Returns:
            action: (B, horizon_steps, action_dim)
        """
        mu, log_std = self.latent_policy(cond)
        if deterministic:
            z = mu
        else:
            std = log_std.exp()
            z = mu + std * torch.randn_like(std)
        action = self.generator.predict(cond=cond, z=z, clip_actions=True)
        return action

    # ──────────────────────────────────────────────────────────────
    #  Rollout helpers (called by the agent during data collection)
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_with_latent(
        self,
        cond: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return the latent z plus log-prob and value.

        Returns:
            action: (B, horizon_steps, action_dim)
            z: (B, horizon_steps, action_dim)
            logprob_z: (B,) summed log-prob of z under the latent policy
            value: (B,) V(s) from the critic
        """
        mu, log_std = self.latent_policy(cond)
        std = log_std.exp()
        z = mu + std * torch.randn_like(std)

        # Log-prob of the sampled z under the diagonal Gaussian
        logprob_z = self._compute_logprob(mu, log_std, z)

        action = self.generator.predict(cond=cond, z=z, clip_actions=True)
        value = self.critic(cond).squeeze(-1)

        return action, z, logprob_z, value

    # ──────────────────────────────────────────────────────────────
    #  PPO loss computation
    # ──────────────────────────────────────────────────────────────

    def get_logprobs_entropy(
        self,
        cond: Dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-compute log-prob and entropy for stored z under current policy.

        Args:
            cond: observation dict.
            z: (B, horizon_steps, action_dim) stored latent samples.

        Returns:
            logprob: (B,) summed log-prob of z
            entropy: scalar mean entropy of the latent distribution
            std: scalar mean std for logging
        """
        mu, log_std = self.latent_policy(cond)
        logprob = self._compute_logprob(mu, log_std, z)

        std = log_std.exp()
        # Entropy of diagonal Gaussian: 0.5 * ln(2*pi*e) * D + sum(log_std)
        # Per-element: 0.5 * (1 + ln(2*pi)) + log_std
        entropy = (0.5 * (1.0 + 2.506628274631) + log_std).mean()
        return logprob, entropy, std.mean()

    def loss(
        self,
        obs: Dict[str, torch.Tensor],
        z: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, torch.Tensor]:
        """Compute clipped PPO loss in latent space + value loss.

        Returns:
            pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio_mean, std
        """
        # Recompute logprobs and entropy under current policy
        new_logprobs, entropy, std = self.get_logprobs_entropy(obs, z)

        # PPO clipped surrogate
        log_ratio = new_logprobs - old_logprobs
        ratio = log_ratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -advantages_normalized * ratio
        pg_loss2 = -advantages_normalized * torch.clamp(
            ratio, 1.0 - self.clip_ploss_coef, 1.0 + self.clip_ploss_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        new_values = self.critic(obs).squeeze(-1)
        if self.clip_vloss_coef is not None:
            v_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss = torch.max(
                F.mse_loss(new_values, returns),
                F.mse_loss(v_clipped, returns),
            )
        else:
            v_loss = F.mse_loss(new_values, returns)

        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy

        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl,
            ratio.mean().item(),
            std,
        )

    # ──────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_logprob(
        mu: torch.Tensor,
        log_std: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sum of element-wise log-probs of z under N(mu, diag(std^2)).

        Args:
            mu: (B, H, D)
            log_std: (B, H, D)
            z: (B, H, D)

        Returns:
            logprob: (B,) summed over all dimensions
        """
        std = log_std.exp()
        var = std * std
        log_scale = log_std
        # log N(z; mu, sigma^2) = -0.5 * [(z-mu)^2/var + log(2*pi) + 2*log_std]
        element_logprob = (
            -0.5 * ((z - mu) ** 2 / (var + 1e-8) + 1.8378770664093453 + 2 * log_scale)
        )
        return element_logprob.reshape(z.shape[0], -1).sum(dim=-1)
