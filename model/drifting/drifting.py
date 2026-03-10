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
Drifting Policy Implementation

Implements the Drifting Policy algorithm which achieves single-step (1 NFE)
inference by front-loading the iterative generation process into the training
phase via a drifting field. This is particularly advantageous for high-frequency
real-time robotic control.
"""

import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from model.flow.mlp_meanflow import MeanFlowMLP

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class DriftingPolicy(nn.Module):
    """
    Drifting Policy implementing drifting-field-based single-step generation.
    
    The core idea is to compute a drifting field V from positive (expert) 
    and optional negative samples, then train the network so that its output
    converges toward the drifted target. At inference time, only a single 
    forward pass (1 NFE) is needed.
    """

    def __init__(
        self,
        network: MeanFlowMLP,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        # Drifting-specific parameters
        drift_coef: float = 1.0,
        neg_drift_coef: float = 0.5,
        mask_self: bool = True,
        bandwidth: float = 1.0,
    ):
        """
        Initialize DriftingPolicy.

        Args:
            network: MeanFlowMLP network for action prediction
            device: Device to run the model on
            horizon_steps: Number of steps in trajectory horizon
            action_dim: Dimension of action space
            act_min: Minimum action value for clipping
            act_max: Maximum action value for clipping
            obs_dim: Dimension of observation space
            max_denoising_steps: Maximum denoising steps (for compatibility)
            seed: Random seed for reproducibility
            drift_coef: Coefficient for positive drift field
            neg_drift_coef: Coefficient for negative drift field
            mask_self: Whether to mask self-interaction in drift field
            bandwidth: Bandwidth parameter for drift field kernel
        """
        super().__init__()

        if int(max_denoising_steps) <= 0:
            raise ValueError('max_denoising_steps must be a positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.horizon_steps, self.action_dim)
        self.act_range = (act_min, act_max)
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)

        # Drifting-specific parameters
        self.drift_coef = drift_coef
        self.neg_drift_coef = neg_drift_coef
        self.mask_self = mask_self
        self.bandwidth = bandwidth

    def compute_V(
        self, x: Tensor, y_pos: Tensor, y_neg: Tensor = None, mask_self: bool = True
    ) -> Tensor:
        """
        Compute the total drifting field V_total from positive and optional negative targets.
        
        The positive field V_pos attracts generated samples toward expert actions,
        while the optional negative field V_neg repels them from undesired actions.
        Uses an RBF-like kernel to compute pairwise interactions.

        Args:
            x: (B, Ta, Da) generated action predictions
            y_pos: (B, Ta, Da) positive (expert) action targets
            y_neg: (B, Ta, Da) or None, negative action targets

        Returns:
            V_total: (B, Ta, Da) total drifting field
        """
        B = x.shape[0]
        x_flat = x.reshape(B, -1)  # (B, Ta*Da)
        y_pos_flat = y_pos.reshape(B, -1)  # (B, Ta*Da)

        # Compute pairwise squared distances: ||x_i - y_pos_j||^2
        diff_pos = x_flat.unsqueeze(1) - y_pos_flat.unsqueeze(0)  # (B, B, Ta*Da)
        dist_sq_pos = (diff_pos ** 2).sum(-1)  # (B, B)

        # RBF kernel weights
        weights_pos = torch.exp(-dist_sq_pos / (2.0 * self.bandwidth ** 2))  # (B, B)

        if mask_self:
            mask = 1.0 - torch.eye(B, device=x.device)
            weights_pos = weights_pos * mask

        # Normalize weights
        weights_pos_sum = weights_pos.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        weights_pos_norm = weights_pos / weights_pos_sum  # (B, B)

        # Positive drift field: weighted average direction toward y_pos
        # V_pos_i = sum_j w_ij * (y_pos_j - x_i)
        V_pos = torch.bmm(
            weights_pos_norm.unsqueeze(1),  # (B, 1, B)
            -diff_pos  # (B, B, Ta*Da) => direction: y_pos_j - x_i
        ).squeeze(1)  # (B, Ta*Da)

        V_total = self.drift_coef * V_pos

        # Negative drift field (repulsion from y_neg)
        if y_neg is not None:
            y_neg_flat = y_neg.reshape(B, -1)
            diff_neg = x_flat.unsqueeze(1) - y_neg_flat.unsqueeze(0)  # (B, B, Ta*Da)
            dist_sq_neg = (diff_neg ** 2).sum(-1)  # (B, B)
            weights_neg = torch.exp(-dist_sq_neg / (2.0 * self.bandwidth ** 2))

            if mask_self:
                weights_neg = weights_neg * mask

            weights_neg_sum = weights_neg.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights_neg_norm = weights_neg / weights_neg_sum

            # V_neg_i = sum_j w_ij * (x_i - y_neg_j), pointing away from negatives
            V_neg = torch.bmm(
                weights_neg_norm.unsqueeze(1),
                diff_neg
            ).squeeze(1)

            V_total = V_total + self.neg_drift_coef * V_neg

        return V_total.reshape(B, self.horizon_steps, self.action_dim)

    def loss(self, x1: Tensor, cond: dict, y_neg: Tensor = None) -> Tensor:
        """
        Compute Drifting Policy loss for offline pretraining.

        In pure behavior cloning (BC) mode, y_pos is the expert action (x1), 
        and y_neg can be omitted or used as regularization.
        
        The loss drives the network output toward the drifted version of itself:
            target = network(z, cond) + V_total
            loss = MSE(network(z, cond), target.detach())

        Args:
            x1: (B, Ta, Da) expert action trajectories (used as y_pos)
            cond: dict with 'state' key for observations
            y_neg: (B, Ta, Da) or None, negative action samples for regularization

        Returns:
            Scalar loss tensor
        """
        B = x1.shape[0]
        y_pos = torch.clamp(x1, *self.act_range)

        # Generate initial noise as input
        x_gen = torch.randn((B,) + self.data_shape, device=self.device)

        # Use t=1.0, r=0.0 for single-step prediction
        t = torch.ones(B, device=self.device)
        r = torch.zeros(B, device=self.device)

        # Network prediction (the current generated action)
        x = self.network(x_gen, t, r, cond)

        # Compute drifting field
        V_total = self.compute_V(x, y_pos, y_neg, mask_self=self.mask_self)

        # Target: drifted version of the prediction
        target = (x + V_total).detach()

        # Loss: drive network output toward the drifted target
        loss = F.mse_loss(x, target)
        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int = 1,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None,
    ) -> Sample:
        """
        1 NFE sampling: only a single forward pass is needed.

        Args:
            cond: Condition dictionary with 'state' key
            inference_steps: Number of inference steps (forced to 1 for Drifting Policy)
            record_intermediate: Whether to record intermediate steps
            clip_intermediate_actions: Whether to clip actions to valid range
            z: Initial noise (if None, sample from Gaussian)

        Returns:
            Sample namedtuple with trajectories and optional chains
        """
        B = cond['state'].shape[0]

        if z is None:
            z = torch.randn((B,) + self.data_shape, device=self.device)

        # Single-step forward pass: t=1.0, r=0.0
        t = torch.ones(B, device=self.device)
        r = torch.zeros(B, device=self.device)

        action = self.network(z, t, r, cond)
        action = action.clamp(*self.act_range)

        chains = None
        if record_intermediate:
            chains = action.unsqueeze(0)

        return Sample(trajectories=action, chains=chains)
