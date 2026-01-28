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
MLP networks for Consistency Model implementation.
Supports both state-based and vision-based policies.
"""

import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Tuple
from torch import Tensor
from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb

log = logging.getLogger(__name__)


class ConsistencyMLP(nn.Module):
    """MLP network for Consistency Model that predicts denoised output."""

    def __init__(
        self,
        horizon_steps,
        action_dim,
        cond_dim,
        time_dim=128,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.act_dim_total = action_dim * horizon_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.mlp_dims = mlp_dims
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style

        # Time embedding for sigma (noise level)
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        model = ResidualMLP if residual_style else MLP

        # Observation encoder
        if cond_mlp_dims:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            self.cond_enc_dim = cond_mlp_dims[-1]
        else:
            self.cond_enc_dim = cond_dim

        # Input: action + time_embedding + condition
        input_dim = time_dim + action_dim * horizon_steps + self.cond_enc_dim

        # Denoising prediction head
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [self.act_dim_total],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )

    def forward(
        self,
        action,
        time,
        cond,
        **kwargs,
    ):
        """
        Forward pass for Consistency Model network.

        Args:
            action: (B, Ta, Da) noisy action trajectories (scaled by c_in)
            time: (B,) or scalar, log-scaled sigma value
            cond: dict with key state; observations
                  state: (B, To, Do)

        Returns:
            Denoised output: (B, Ta, Da)
        """
        B, Ta, Da = action.shape

        # Flatten action chunk
        action = action.view(B, -1)

        # Flatten observation history
        state = cond["state"].view(B, -1)

        # Observation encoder
        cond_emb = self.cond_mlp(state) if hasattr(self, "cond_mlp") else state

        # Time embedding for sigma
        if isinstance(time, (int, float)):
            time = torch.ones((B, 1), device=action.device) * time
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)

        # Combine all features
        feature = torch.cat([action, time_emb, cond_emb], dim=-1)
        output = self.mlp_mean(feature)

        return output.view(B, Ta, Da)


class ConsistencyViT(nn.Module):
    """Vision Transformer version of Consistency Model for image-based policies."""

    def __init__(
        self,
        backbone,  # ViT encoder
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=128,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=128,
        visual_feature_dim=128,
        dropout=0.1,
        num_img=1,
        augment=False,
    ):
        super().__init__()

        # Action and observation dimensions
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = action_dim * horizon_steps
        self.prop_dim = cond_dim
        self.img_cond_steps = img_cond_steps

        # Time embedding
        self.time_dim = time_dim

        # Vision backbone
        self.backbone = backbone
        self.num_img = num_img
        self.augment = augment
        self.dropout = dropout

        # Visual feature processing
        if spatial_emb > 0:
            # Use spatial embeddings if specified
            from model.common.modules import SpatialEmb
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            visual_feature_dim = spatial_emb
        else:
            # Default compression
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        self.cond_enc_dim = visual_feature_dim + self.prop_dim

        # Time embedding (log-scaled sigma)
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # MLP for denoising prediction
        input_dim = time_dim + action_dim * horizon_steps + self.cond_enc_dim
        output_dim = action_dim * horizon_steps

        model = ResidualMLP if residual_style else MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )

    def forward(
        self,
        action,
        time,
        cond: dict,
        **kwargs,
    ):
        """
        Forward pass for vision-based Consistency Model.

        Args:
            action: (B, Ta, Da) noisy action chunk (scaled by c_in)
            time: (B,) log-scaled sigma value
            cond: dict with keys state/rgb
                  state: (B, To, Do)
                  rgb: (B, To, C, H, W)

        Returns:
            Denoised output: (B, Ta, Da)
        """
        B, Ta, Da = action.shape

        # Flatten action
        action = action.view(B, -1)

        # Process proprioceptive state
        state = cond["state"].view(B, -1)

        # Process visual input
        rgb = cond["rgb"][:, -self.img_cond_steps:]
        import einops
        rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        rgb = rgb.float().contiguous()

        # Extract visual features
        if self.augment:
            from model.common.modules import RandomShiftsAug
            aug = RandomShiftsAug(pad=4)
            rgb = aug(rgb)

        try:
            feat = self.backbone.forward(rgb)
        except RuntimeError as err:
            if "Unable to find a valid cuDNN algorithm" in str(err):
                cudnn_enabled = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                try:
                    feat = self.backbone.forward(rgb)
                finally:
                    torch.backends.cudnn.enabled = cudnn_enabled
            else:
                raise

        # Compress visual features
        from model.common.modules import SpatialEmb
        if isinstance(self.compress, SpatialEmb):
            feat = self.compress.forward(feat, state)
        else:
            feat = feat.flatten(1, -1)
            feat = self.compress(feat)

        # Combine visual and proprioceptive features
        cond_encoded = torch.cat([feat, state], dim=-1)

        # Time embedding
        if isinstance(time, (int, float)):
            time = torch.ones((B, 1), device=action.device) * time
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)

        # Combine all features
        emb = torch.cat([action, time_emb, cond_encoded], dim=-1)

        # Predict denoised output
        output = self.mlp_mean(emb)

        return output.view(B, Ta, Da)

    def forward_encoder(self, cond: dict):
        """
        Extract only the condition encoding without full forward pass.
        Used for downstream tasks like PPO fine-tuning.

        Args:
            cond: dict with keys state/rgb

        Returns:
            cond_encoded: (B, cond_enc_dim) encoded condition
        """
        # Process proprioceptive state
        state = cond["state"].view(cond["state"].shape[0], -1)

        # Process visual input
        rgb = cond["rgb"][:, -self.img_cond_steps:]
        import einops
        rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        rgb = rgb.float()

        # Extract visual features
        if self.augment:
            from model.common.modules import RandomShiftsAug
            aug = RandomShiftsAug(pad=4)
            rgb = aug(rgb)

        feat = self.backbone.forward(rgb)

        # Compress visual features
        from model.common.modules import SpatialEmb
        if isinstance(self.compress, SpatialEmb):
            feat = self.compress.forward(feat, state)
        else:
            feat = feat.flatten(1, -1)
            feat = self.compress(feat)

        # Combine visual and proprioceptive features
        cond_encoded = torch.cat([feat, state], dim=-1)

        return cond_encoded
