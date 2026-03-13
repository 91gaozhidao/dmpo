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
Latent policy heads for Latent-PPO Drifting.

The latent policy pi_phi(z|s) outputs a diagonal Gaussian in the same shape
as the drifting generator's noise input [horizon_steps, action_dim].

Two variants are provided:
- LatentPolicyHeadLowdim: for low-dimensional (state-only) observations.
- LatentPolicyHeadImage: for image-conditioned observations using a ViT encoder.
"""

import torch
import torch.nn as nn
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import SpatialEmb, RandomShiftsAug


class LatentPolicyHeadLowdim(nn.Module):
    """Diagonal Gaussian z-policy conditioned on low-dim state observations.

    Outputs mu(s) and log_std(s) with shape [horizon_steps, action_dim] each.
    Initialized so that z ~ N(0, I) at the start of training, matching
    drifting pretrain sampling exactly.
    """

    def __init__(
        self,
        cond_dim,
        horizon_steps,
        action_dim,
        mlp_dims=(256, 256, 256),
        activation_type="Mish",
        residual_style=False,
        use_layernorm=False,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.latent_dim = horizon_steps * action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        output_dim = self.latent_dim * 2  # mu and log_std
        dims = [cond_dim] + list(mlp_dims) + [output_dim]
        if residual_style:
            model_cls = ResidualMLP
        else:
            model_cls = MLP
        self.mlp = model_cls(
            dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

        # Initialize output layer to produce mu=0, log_std=0 (std=1)
        self._init_output_layer()

    def _init_output_layer(self):
        """Initialize the last linear layer so mu ≈ 0, log_std ≈ 0."""
        last_layer = None
        for module in reversed(list(self.mlp.modules())):
            if isinstance(module, nn.Linear):
                last_layer = module
                break
        if last_layer is not None:
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(self, cond):
        """
        Args:
            cond: dict with 'state' key, shape (B, To, Do) or (B, D_flat).

        Returns:
            mu: (B, horizon_steps, action_dim)
            log_std: (B, horizon_steps, action_dim), clamped
        """
        if isinstance(cond, dict):
            B = cond["state"].shape[0]
            state = cond["state"].view(B, -1)
        else:
            B = cond.shape[0]
            state = cond.view(B, -1)

        out = self.mlp(state)  # (B, latent_dim * 2)
        mu, log_std = out.split(self.latent_dim, dim=-1)

        mu = mu.view(B, self.horizon_steps, self.action_dim)
        log_std = log_std.view(B, self.horizon_steps, self.action_dim)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)

        return mu, log_std


class LatentPolicyHeadImage(nn.Module):
    """Diagonal Gaussian z-policy conditioned on image + low-dim state.

    Uses a ViT backbone for visual feature extraction, then maps to
    mu(s) and log_std(s) with shape [horizon_steps, action_dim].
    Initialized so that z ~ N(0, I) at the start of training.
    """

    def __init__(
        self,
        backbone,
        cond_dim,
        horizon_steps,
        action_dim,
        img_cond_steps=1,
        mlp_dims=(256, 256, 256),
        activation_type="Mish",
        residual_style=False,
        use_layernorm=False,
        spatial_emb=128,
        visual_feature_dim=128,
        dropout=0.0,
        num_img=1,
        augment=False,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.latent_dim = horizon_steps * action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps

        # Vision encoder
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

        if spatial_emb > 0:
            if num_img > 1:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        input_dim = visual_feature_dim + cond_dim
        output_dim = self.latent_dim * 2
        dims = [input_dim] + list(mlp_dims) + [output_dim]
        if residual_style:
            model_cls = ResidualMLP
        else:
            model_cls = MLP
        self.mlp = model_cls(
            dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

        # Initialize output layer to produce mu=0, log_std=0
        self._init_output_layer()

    def _init_output_layer(self):
        """Initialize the last linear layer so mu ≈ 0, log_std ≈ 0."""
        last_layer = None
        for module in reversed(list(self.mlp.modules())):
            if isinstance(module, nn.Linear):
                last_layer = module
                break
        if last_layer is not None:
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def _encode_visual(self, cond, no_augment=False):
        """Extract visual features from image observations."""
        B, T_rgb, C, H, W = cond["rgb"].shape
        state = cond["state"].view(B, -1)
        rgb = cond["rgb"][:, -self.img_cond_steps:]

        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        rgb = rgb.float()

        if self.num_img > 1:
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:
            if self.augment and not no_augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)
            feat = self.compress.forward(feat, state)

        return torch.cat([feat, state], dim=-1)

    def forward(self, cond, no_augment=False):
        """
        Args:
            cond: dict with 'state' (B, To, Do) and 'rgb' (B, To, C, H, W).

        Returns:
            mu: (B, horizon_steps, action_dim)
            log_std: (B, horizon_steps, action_dim), clamped
        """
        B = cond["state"].shape[0]
        features = self._encode_visual(cond, no_augment=no_augment)
        out = self.mlp(features)  # (B, latent_dim * 2)
        mu, log_std = out.split(self.latent_dim, dim=-1)

        mu = mu.view(B, self.horizon_steps, self.action_dim)
        log_std = log_std.view(B, self.horizon_steps, self.action_dim)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)

        return mu, log_std
