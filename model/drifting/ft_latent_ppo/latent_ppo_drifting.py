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
Latent-space PPO wrapper for drifting policies.

The drifting generator stays frozen and keeps the 1-NFE action mapping
    a = f_theta(s, z).
PPO is applied only to an explicit latent policy pi_phi(z | s).
"""

from __future__ import annotations

import logging
import math
from typing import Dict

import einops
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import RandomShiftsAug, SpatialEmb
from model.drifting.drifting import DriftingPolicy
from model.drifting.ft_qguided.qguided_drifting import (
    _is_compatible_drifting_buffer_key,
    load_drifting_policy_from_checkpoint,
)

log = logging.getLogger(__name__)


def _build_mlp_backbone(
    dim_list,
    activation_type="Mish",
    use_layernorm=False,
    residual_style=False,
):
    if len(dim_list) < 2:
        raise ValueError("dim_list must contain at least input and output dims")
    if residual_style and (len(dim_list) - 3) % 2 != 0:
        dim_list = list(dim_list) + [dim_list[-1]]
    model_cls = ResidualMLP if residual_style else MLP
    return model_cls(
        dim_list,
        activation_type=activation_type,
        out_activation_type=activation_type,
        use_layernorm=use_layernorm,
        use_layernorm_final=use_layernorm,
    )


class _LatentPolicyHeadBase(nn.Module):
    def __init__(
        self,
        horizon_steps: int,
        action_dim: int,
        min_std: float = 0.05,
        max_std: float = 2.0,
    ):
        super().__init__()
        if horizon_steps <= 0 or action_dim <= 0:
            raise ValueError("horizon_steps and action_dim must be positive")
        if min_std <= 0 or max_std <= 0 or min_std > max_std:
            raise ValueError(
                "Expected 0 < min_std <= max_std, "
                f"but got min_std={min_std}, max_std={max_std}."
            )
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.latent_dim = self.horizon_steps * self.action_dim
        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)

    def _build_distribution(self, features: torch.Tensor):
        batch_size = features.shape[0]
        mu = self.mu_head(features).view(
            batch_size, self.horizon_steps, self.action_dim
        )
        raw_log_std = self.log_std_head(features).view(
            batch_size, self.horizon_steps, self.action_dim
        )
        log_std = raw_log_std.clamp(self.min_log_std, self.max_log_std)
        std = log_std.exp()
        dist = Independent(Normal(mu, std), 2)
        return dist, mu, std

    def distribution(self, cond: Dict[str, torch.Tensor]):
        features = self.forward_encoder(cond)
        return self._build_distribution(features)

    def sample(self, cond: Dict[str, torch.Tensor], deterministic: bool = False):
        dist, mu, std = self.distribution(cond)
        z = mu if deterministic else dist.rsample()
        return {
            "latents": z,
            "logprob": dist.log_prob(z),
            "entropy": dist.entropy(),
            "latent_mean": mu,
            "latent_std": std,
        }

    def evaluate(self, cond: Dict[str, torch.Tensor], z: torch.Tensor):
        dist, mu, std = self.distribution(cond)
        return {
            "logprob": dist.log_prob(z),
            "entropy": dist.entropy(),
            "latent_mean": mu,
            "latent_std": std,
        }


class LatentPolicyHeadLowdim(_LatentPolicyHeadBase):
    def __init__(
        self,
        cond_dim,
        horizon_steps,
        action_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        min_std=0.05,
        max_std=2.0,
    ):
        super().__init__(
            horizon_steps=horizon_steps,
            action_dim=action_dim,
            min_std=min_std,
            max_std=max_std,
        )
        hidden_dims = list(mlp_dims)
        if len(hidden_dims) == 0:
            raise ValueError("mlp_dims must be non-empty for LatentPolicyHeadLowdim")
        self.trunk = _build_mlp_backbone(
            [cond_dim] + hidden_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            residual_style=residual_style,
        )
        hidden_dim = hidden_dims[-1]
        self.mu_head = nn.Linear(hidden_dim, self.latent_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.latent_dim)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def forward_encoder(self, cond: Dict[str, torch.Tensor]):
        state = cond["state"]
        batch_size = state.shape[0]
        return self.trunk(state.view(batch_size, -1))


class LatentPolicyHeadImage(_LatentPolicyHeadBase):
    def __init__(
        self,
        backbone,
        cond_dim,
        horizon_steps,
        action_dim,
        mlp_dims,
        img_cond_steps=1,
        spatial_emb=128,
        visual_feature_dim=128,
        dropout=0.0,
        augment=False,
        num_img=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        min_std=0.05,
        max_std=2.0,
    ):
        super().__init__(
            horizon_steps=horizon_steps,
            action_dim=action_dim,
            min_std=min_std,
            max_std=max_std,
        )
        self.backbone = backbone
        self.cond_dim = cond_dim
        self.img_cond_steps = img_cond_steps
        self.num_img = num_img
        self.use_spatial_emb = spatial_emb > 0
        if self.use_spatial_emb:
            if self.num_img > 1:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                trunk_input_dim = spatial_emb * self.num_img + cond_dim
            else:
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                trunk_input_dim = spatial_emb + cond_dim
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
            trunk_input_dim = visual_feature_dim + cond_dim
        self.augment = augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)

        hidden_dims = list(mlp_dims)
        if len(hidden_dims) == 0:
            raise ValueError("mlp_dims must be non-empty for LatentPolicyHeadImage")
        self.trunk = _build_mlp_backbone(
            [trunk_input_dim] + hidden_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            residual_style=residual_style,
        )
        hidden_dim = hidden_dims[-1]
        self.mu_head = nn.Linear(hidden_dim, self.latent_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.latent_dim)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def forward_encoder(self, cond: Dict[str, torch.Tensor]):
        state = cond["state"]
        batch_size = state.shape[0]
        flat_state = state.view(batch_size, -1)
        rgb = cond["rgb"][:, -self.img_cond_steps :].float()

        if self.num_img > 1:
            _, t_rgb, _, h, w = rgb.shape
            rgb = rgb.reshape(batch_size, t_rgb, self.num_img, 3, h, w)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
            rgb_inputs = [rgb[:, img_idx] for img_idx in range(self.num_img)]
            if self.augment and self.training:
                rgb_inputs = [self.aug(image) for image in rgb_inputs]
            feats = [self.backbone(image) for image in rgb_inputs]
            visual_feat = torch.cat(
                [
                    self.compress1(feats[0], flat_state),
                    self.compress2(feats[1], flat_state),
                ],
                dim=-1,
            )
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
            if self.augment and self.training:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)
            if self.use_spatial_emb:
                visual_feat = self.compress(feat, flat_state)
            else:
                visual_feat = self.compress(feat.flatten(1, -1))

        return self.trunk(torch.cat([visual_feat, flat_state], dim=-1))


class LatentPPODrifting(nn.Module):
    def __init__(
        self,
        device,
        policy,
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
    ):
        super().__init__()
        self.device = torch.device(device)
        self.action_dim = int(act_dim)
        self.horizon_steps = int(horizon_steps)
        self.act_steps = int(act_steps)
        self.act_min = act_min
        self.act_max = act_max
        self.obs_dim = int(obs_dim)
        self.cond_steps = int(cond_steps)
        self.use_ema_checkpoint = use_ema_checkpoint

        if self.act_steps > self.horizon_steps:
            raise ValueError("act_steps must be <= horizon_steps")

        if isinstance(policy, DriftingPolicy):
            self.actor = policy.to(self.device)
        else:
            self.actor = DriftingPolicy(
                network=policy,
                device=self.device,
                horizon_steps=self.horizon_steps,
                action_dim=self.action_dim,
                act_min=self.act_min,
                act_max=self.act_max,
                obs_dim=self.obs_dim,
                max_denoising_steps=1,
                seed=seed,
            ).to(self.device)

        if actor_policy_path:
            loaded_actor = load_drifting_policy_from_checkpoint(
                actor_policy_path,
                device=self.device,
                use_ema=self.use_ema_checkpoint,
            )
            missing_keys, unexpected_keys = self.actor.load_state_dict(
                loaded_actor.state_dict(),
                strict=False,
            )
            incompatible_missing = [
                key for key in missing_keys if not _is_compatible_drifting_buffer_key(key)
            ]
            incompatible_unexpected = [
                key
                for key in unexpected_keys
                if not _is_compatible_drifting_buffer_key(key)
            ]
            if incompatible_missing or incompatible_unexpected:
                raise ValueError(
                    "Failed to copy the drifting actor into LatentPPODrifting. "
                    f"Missing keys: {incompatible_missing}. "
                    f"Unexpected keys: {incompatible_unexpected}."
                )
            if missing_keys or unexpected_keys:
                log.info(
                    "Ignoring compatible drifting buffer mismatch while copying "
                    "actor checkpoint. Missing keys: %s. Unexpected keys: %s.",
                    missing_keys,
                    unexpected_keys,
                )
        else:
            log.warning(
                "LatentPPODrifting is starting from the provided drifting backbone "
                "without loading an actor_policy_path checkpoint."
            )

        self.latent_policy = latent_policy.to(self.device)
        self.critic = critic.to(self.device)

        for param in self.actor.parameters():
            param.requires_grad = False
        self.actor.eval()

        self.report_network_params()

    def train(self, mode: bool = True):
        super().train(mode)
        self.actor.eval()
        return self

    def report_network_params(self):
        log.info(
            "Number of network parameters: Total %.3f M | Frozen actor %.3f M | "
            "Latent policy %.3f M | Critic %.3f M",
            sum(p.numel() for p in self.parameters()) / 1e6,
            sum(p.numel() for p in self.actor.parameters()) / 1e6,
            sum(p.numel() for p in self.latent_policy.parameters()) / 1e6,
            sum(p.numel() for p in self.critic.parameters()) / 1e6,
        )

    def value(self, cond: Dict[str, torch.Tensor]):
        return self.critic(cond).view(-1)

    @torch.no_grad()
    def action_from_latent(
        self,
        cond: Dict[str, torch.Tensor],
        z: torch.Tensor,
        clip_actions: bool = True,
    ):
        return self.actor.predict(cond=cond, z=z, clip_actions=clip_actions)

    @torch.no_grad()
    def get_actions(self, cond: Dict[str, torch.Tensor], deterministic: bool = False):
        latent_out = self.latent_policy.sample(cond, deterministic=deterministic)
        actions = self.actor.predict(
            cond=cond,
            z=latent_out["latents"],
            clip_actions=True,
        )
        value = self.value(cond)
        latent_out["actions"] = actions
        latent_out["value"] = value
        return latent_out

    def evaluate_latents(self, cond: Dict[str, torch.Tensor], z: torch.Tensor):
        latent_out = self.latent_policy.evaluate(cond, z)
        latent_out["value"] = self.value(cond)
        return latent_out

    @torch.no_grad()
    def forward(self, cond: Dict[str, torch.Tensor], deterministic: bool = False):
        return self.get_actions(cond=cond, deterministic=deterministic)["actions"]
