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
MLP networks for Mean Flow implementation with Average Velocity prediction.
Implements the MeanFlow algorithm from "Mean Flows for One-step Generative Modeling".
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


class MeanFlowMLP(nn.Module):
    """MLP network for MeanFlow that predicts average velocity."""
    
    def __init__(
        self,
        horizon_steps,
        action_dim,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        # MeanFlow specific parameters
        r_embedding_dim=16,  # embedding dimension for r parameter
    ):
        super().__init__()
        self.time_dim = time_dim
        self.r_embedding_dim = r_embedding_dim
        self.act_dim_total = action_dim * horizon_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.mlp_dims = mlp_dims
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style

        # Time embedding for t parameter
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # R embedding for r parameter (dual time embedding)
        self.r_embedding = nn.Sequential(
            SinusoidalPosEmb(r_embedding_dim),
            nn.Linear(r_embedding_dim, r_embedding_dim * 2),
            nn.Mish(),
            nn.Linear(r_embedding_dim * 2, r_embedding_dim),
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
            
        # Input: action + time_embedding + r_embedding + condition
        input_dim = time_dim + r_embedding_dim + action_dim * horizon_steps + self.cond_enc_dim
        
        # Average velocity prediction head
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
        r,  # Additional r parameter for MeanFlow
        cond,
        output_embedding=False,
        **kwargs,
    ):
        """
        Forward pass for MeanFlow network.
        
        Args:
            action: (B, Ta, Da) action trajectories
            time: (B,) or scalar, time parameter t
            r: (B,) or scalar, time parameter r
            cond: dict with key state; observations
                  state: (B, To, Do)
        
        Returns:
            Average velocity u: (B, Ta, Da) when output_embedding==False
            Or (u, time_emb, r_emb, cond_emb) when output_embedding==True
        """
        B, Ta, Da = action.shape

        # Flatten action chunk
        action = action.view(B, -1)

        # Flatten observation history
        state = cond["state"].view(B, -1)

        # Observation encoder
        cond_emb = self.cond_mlp(state) if hasattr(self, "cond_mlp") else state
        
        # Time embedding for t
        if isinstance(time, (int, float)):
            time = torch.ones((B, 1), device=action.device) * time
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        
        # R embedding for r
        if isinstance(r, (int, float)):
            r = torch.ones((B, 1), device=action.device) * r
        r_emb = self.r_embedding(r.view(B, 1)).view(B, self.r_embedding_dim)
        
        # Combine all features
        u_feature = torch.cat([action, time_emb, r_emb, cond_emb], dim=-1)
        u = self.mlp_mean(u_feature)
        
        if output_embedding:
            return u.view(B, Ta, Da), time_emb, r_emb, cond_emb
        return u.view(B, Ta, Da)

    def forward_encoder(self, cond: dict):
        """
        Extract only the condition encoding without full forward pass.
        Used for noise prediction in PPO fine-tuning.

        Args:
            cond: dict with key 'state'; observations
                  state: (B, To, Do)

        Returns:
            cond_encoded: (B, cond_enc_dim) encoded condition
        """
        state = cond["state"].view(cond["state"].shape[0], -1)
        cond_emb = self.cond_mlp(state) if hasattr(self, "cond_mlp") else state
        return cond_emb

    def sample_action(self, cond: dict, inference_steps: int, clip_intermediate_actions: bool,
                      act_range: list, z: Tensor = None, save_chains: bool = False):
        """
        Sample action using MeanFlow integration procedure.

        MeanFlow sampling: x_r = x_t - (t-r) * u(x_t, t, r, s)

        Args:
            cond: dict with key 'state'; observations
            inference_steps: number of denoising steps
            clip_intermediate_actions: whether to clip intermediate actions
            act_range: [min, max] for action clipping
            z: optional initial noise, if None uses random noise
            save_chains: whether to return the denoising trajectory

        Returns:
            x_hat: (B, Ta, Da) sampled action
            x_chain: (B, inference_steps+1, Ta, Da) if save_chains=True
        """
        B = cond['state'].shape[0]
        device = cond['state'].device

        # Initialize from noise
        x_hat: Tensor = z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)

        if save_chains:
            x_chain = torch.zeros((B, inference_steps + 1, self.horizon_steps, self.action_dim), device=device)
            x_chain[:, 0] = x_hat

        # MeanFlow time schedule: from 1.0 to 0.0
        t_vals = torch.linspace(1.0, 0.0, inference_steps + 1, device=device)

        for i in range(inference_steps):
            t_curr = t_vals[i]
            r_next = t_vals[i + 1]

            # Create batch-wise time tensors
            t = torch.full((B,), t_curr, device=device)
            r = torch.full((B,), r_next, device=device)

            # Predict average velocity
            ut = self.forward(x_hat, t, r, cond)

            # MeanFlow update: x_r = x_t - (t-r) * u(x_t, t, r, s)
            time_diff = t_curr - r_next
            x_hat = x_hat - time_diff * ut

            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*act_range)

            if save_chains:
                x_chain[:, i + 1] = x_hat

        if save_chains:
            return x_hat, x_chain
        return x_hat


class MeanFlowViT(nn.Module):
    """Vision Transformer version of MeanFlow for image-based policies."""
    
    def __init__(
        self,
        backbone,  # ViT encoder
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=16,
        r_embedding_dim=16,
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
        
        # Time and r embeddings
        self.time_dim = time_dim
        self.r_embedding_dim = r_embedding_dim
        
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
        
        # Time and r embeddings
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        self.r_embedding = nn.Sequential(
            SinusoidalPosEmb(r_embedding_dim),
            nn.Linear(r_embedding_dim, r_embedding_dim * 2),
            nn.Mish(),
            nn.Linear(r_embedding_dim * 2, r_embedding_dim),
        )
        
        # MLP for average velocity prediction
        input_dim = time_dim + r_embedding_dim + action_dim * horizon_steps + self.cond_enc_dim
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
        r,
        cond: dict,
        output_embedding=False,
        **kwargs,
    ):
        """
        Forward pass for vision-based MeanFlow.
        
        Args:
            action: (B, Ta, Da) action chunk
            time: (B,) time parameter t
            r: (B,) time parameter r  
            cond: dict with keys state/rgb
                  state: (B, To, Do)
                  rgb: (B, To, C, H, W)
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
            # SpatialEmb case - needs both feat and state
            feat = self.compress.forward(feat, state)
        else:
            # Sequential case - only needs feat
            feat = feat.flatten(1, -1)
            feat = self.compress(feat)
            
        # Combine visual and proprioceptive features
        cond_encoded = torch.cat([feat, state], dim=-1)
        
        # Time and r embeddings
        if isinstance(time, (int, float)):
            time = torch.ones((B, 1), device=action.device) * time
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        
        if isinstance(r, (int, float)):
            r = torch.ones((B, 1), device=action.device) * r
        r_emb = self.r_embedding(r.view(B, 1)).view(B, self.r_embedding_dim)
        
        # Combine all features
        emb = torch.cat([action, time_emb, r_emb, cond_encoded], dim=-1)
        
        # Predict average velocity
        u = self.mlp_mean(emb)
        
        if output_embedding:
            return u.view(B, Ta, Da), time_emb, r_emb, cond_encoded
        return u.view(B, Ta, Da)
    
    def forward_encoder(self, cond: dict):
        """
        Extract only the condition encoding without full forward pass.
        Used for noise prediction in PPO fine-tuning.
        
        Args:
            cond: dict with keys state/rgb
                  state: (B, To, Do)
                  rgb: (B, To, C, H, W)
        
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
            # SpatialEmb case - needs both feat and state
            feat = self.compress.forward(feat, state)
        else:
            # Sequential case - only needs feat
            feat = feat.flatten(1, -1)
            feat = self.compress(feat)
            
        # Combine visual and proprioceptive features
        cond_encoded = torch.cat([feat, state], dim=-1)

        return cond_encoded

    def sample_action(self, cond: dict, inference_steps: int, clip_intermediate_actions: bool,
                      act_range: list, z: Tensor = None, save_chains: bool = False,
                      use_checkpoint: bool = True):
        """
        Sample action using MeanFlow integration procedure for vision-based policy.

        MeanFlow sampling: x_r = x_t - (t-r) * u(x_t, t, r, s)

        Args:
            cond: dict with keys 'state' and 'rgb'; observations
            inference_steps: number of denoising steps
            clip_intermediate_actions: whether to clip intermediate actions
            act_range: [min, max] for action clipping
            z: optional initial noise, if None uses random noise
            save_chains: whether to return the denoising trajectory
            use_checkpoint: whether to use gradient checkpointing to save memory

        Returns:
            x_hat: (B, Ta, Da) sampled action
            x_chain: (B, inference_steps+1, Ta, Da) if save_chains=True
        """
        from torch.utils.checkpoint import checkpoint

        B = cond['state'].shape[0]
        device = cond['state'].device

        # Initialize from noise
        x_hat: Tensor = z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)

        if save_chains:
            x_chain = torch.zeros((B, inference_steps + 1, self.horizon_steps, self.action_dim), device=device)
            x_chain[:, 0] = x_hat

        # MeanFlow time schedule: from 1.0 to 0.0
        t_vals = torch.linspace(1.0, 0.0, inference_steps + 1, device=device)

        for i in range(inference_steps):
            t_curr = t_vals[i]
            r_next = t_vals[i + 1]

            # Create batch-wise time tensors
            t = torch.full((B,), t_curr, device=device)
            r = torch.full((B,), r_next, device=device)

            # Predict average velocity with optional gradient checkpointing
            if use_checkpoint and self.training and torch.is_grad_enabled():
                # Use gradient checkpointing to save memory (trades compute for memory)
                ut = checkpoint(self.forward, x_hat, t, r, cond, use_reentrant=False)
            else:
                ut = self.forward(x_hat, t, r, cond)

            # MeanFlow update: x_r = x_t - (t-r) * u(x_t, t, r, s)
            time_diff = t_curr - r_next
            x_hat = x_hat - time_diff * ut

            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*act_range)

            if save_chains:
                x_chain[:, i + 1] = x_hat

        if save_chains:
            return x_hat, x_chain
        return x_hat
