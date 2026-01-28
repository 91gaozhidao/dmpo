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
Improved MeanFlow (iMF) Policy Implementation

Implements the improved MeanFlow algorithm from:
"Improved Mean Flows: On the Challenges of Fastforward Generative Models"
(Geng et al., 2025, arXiv:2512.02012)

Key improvements over original MeanFlow:
1. v-loss formulation: Regression target is network-independent
2. Uses predicted velocity v_θ instead of ground truth v for JVP tangent
3. More stable training with lower variance gradients

Reference: https://arxiv.org/abs/2512.02012
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


def stopgrad(x):
    """Stop gradient computation"""
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss as defined in MeanFlow paper.

    Args:
        error: Tensor of prediction errors (B, Ta, Da) for actions
        gamma: Power parameter for adaptive weighting (default: 0.5)
        c: Small constant for numerical stability (default: 1e-3)

    Returns:
        Scalar loss value
    """
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


class ImprovedMeanFlow(nn.Module):
    """
    Improved MeanFlow (iMF) Policy implementing v-loss based training.

    This implementation follows the iMF paper which reformulates the training
    objective as a v-loss that regresses to network-independent targets,
    resulting in more stable training.

    Key differences from original MeanFlow:
    1. JVP tangent uses predicted velocity v_θ instead of ground truth v
    2. Loss is computed as ||V_θ - v||² instead of ||u - sg(u_target)||²
    3. V_θ = u_θ + (t-r) * sg(∂u/∂t) is the composite function
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
        # MeanFlow specific parameters
        flow_ratio: float = 0.5,
        gamma: float = 0.5,
        c: float = 1e-3,
        sample_t_type: str = 'uniform',
        use_adaptive_loss: bool = False,
        # iMF specific parameters
        use_auxiliary_head: bool = False,  # Whether to use auxiliary v head (scheme B)
    ):
        """
        Initialize Improved MeanFlow model.

        Args:
            network: MeanFlowMLP network for average velocity prediction
            device: Device to run the model on
            horizon_steps: Number of steps in trajectory horizon
            action_dim: Dimension of action space
            act_min: Minimum action value for clipping
            act_max: Maximum action value for clipping
            obs_dim: Dimension of observation space
            max_denoising_steps: Maximum denoising steps
            seed: Random seed for reproducibility
            flow_ratio: Ratio of samples using flow consistency constraint
            gamma: Adaptive loss parameter
            c: Stability constant for adaptive loss
            sample_t_type: Time sampling type
            use_adaptive_loss: Whether to use adaptive L2 loss
            use_auxiliary_head: Whether to use auxiliary head for v prediction (scheme B)
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

        # MeanFlow parameters
        self.flow_ratio = flow_ratio
        self.gamma = gamma
        self.c = c
        self.sample_t_type = sample_t_type
        self.use_adaptive_loss = use_adaptive_loss

        # iMF specific
        self.use_auxiliary_head = use_auxiliary_head

    def get_predicted_velocity(self, xt: Tensor, t: Tensor, cond: dict) -> Tensor:
        """
        Get predicted marginal velocity v_θ using boundary condition.

        Following iMF paper Scheme A: v_θ(z_t, t) ≡ u_θ(z_t, t, t)
        This exploits the boundary condition where average velocity equals
        instantaneous velocity when r = t.

        Args:
            xt: Current trajectory state (B, Ta, Da)
            t: Current time (B,)
            cond: Condition dictionary

        Returns:
            Predicted velocity v_θ (B, Ta, Da)
        """
        # Boundary condition: v_θ(z_t, t) = u_θ(z_t, t, t)
        return self.network(xt, t, t, cond)

    def loss(self, x1: Tensor, cond: dict) -> Tensor:
        """
        Compute Improved MeanFlow v-loss following iMF paper.

        The key innovation is using predicted velocity v_θ for JVP tangent
        instead of ground truth velocity, making the regression target
        network-independent.

        iMF Loss: ||V_θ(z_t) - v||²
        where V_θ = u + (t-r) * sg(∂u/∂t)
        and ∂u/∂t is computed via JVP with tangent v_θ (not v)

        Args:
            x1: Real action trajectories (B, Ta, Da)
            cond: Condition dictionary with 'state' key

        Returns:
            Scalar loss tensor
        """
        batch_size = x1.shape[0]

        # Normalize action data
        x1 = torch.clamp(x1, *self.act_range)

        # Sample time pairs (t, r) using lognormal distribution
        mu, sigma = -0.4, 1.0
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))  # sigmoid

        # Ensure r <= t
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # Flow consistency for some samples (r = t)
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=self.device)
        r = torch.tensor(r_np, device=self.device)

        t_ = t.view(-1, 1, 1)
        r_ = r.view(-1, 1, 1)

        # Sample noise and generate trajectory
        x0 = torch.randn_like(x1)

        # Generate trajectory: xt = (1-t)*x1 + t*x0
        xt = (1 - t_) * x1 + t_ * x0

        # Ground truth instantaneous velocity (target for v-loss)
        v_gt = x0 - x1

        # === iMF Key Innovation ===
        # Step 1: Get predicted velocity v_θ using boundary condition
        # v_θ(z_t, t) = u_θ(z_t, t, t)
        with torch.no_grad():
            v_pred = self.get_predicted_velocity(xt, t, cond)

        # Step 2: Compute JVP with predicted velocity as tangent (not ground truth!)
        def network_fn(z, t_val, r_val):
            return self.network(z, t_val, r_val, cond)

        # JVP tangents: (v_pred, 1, 0) - using predicted velocity!
        u, dudt = torch.autograd.functional.jvp(
            network_fn,
            (xt, t, r),
            (v_pred, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True
        )

        # Step 3: Compute composite function V_θ = u + (t-r) * sg(dudt)
        # Note: stopgrad on dudt as per iMF paper
        V_theta = u + (t_ - r_) * stopgrad(dudt)

        # Step 4: v-loss - regress to network-independent target v_gt
        if self.use_adaptive_loss:
            error = V_theta - v_gt
            return adaptive_l2_loss(error, gamma=self.gamma, c=self.c)
        else:
            return F.mse_loss(V_theta, v_gt)

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int = 5,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None
    ) -> Sample:
        """
        Sample trajectories using MeanFlow multi-step sampling.

        Sampling is identical to original MeanFlow since we use the same
        u network for inference.

        Args:
            cond: Condition dictionary with 'state' key
            inference_steps: Number of inference steps
            record_intermediate: Whether to record intermediate steps
            clip_intermediate_actions: Whether to clip actions
            z: Initial noise (if None, sample from Gaussian)

        Returns:
            Sample namedtuple with trajectories and optional chains
        """
        B = cond['state'].shape[0]

        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)

        # Initial noise
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)

        # Time schedule: from 1.0 (noise) to 0.0 (data)
        t_vals = torch.linspace(1.0, 0.0, inference_steps + 1, device=self.device)

        for i in range(inference_steps):
            t_curr = t_vals[i]
            r_next = t_vals[i + 1]

            t = torch.full((B,), t_curr, device=self.device)
            r = torch.full((B,), r_next, device=self.device)

            # Predict average velocity u(z_t, r, t)
            u = self.network(x_hat, t, r, cond)

            # Apply MeanFlow sampling: z_r = z_t - (t-r) * u
            time_diff = (t_curr - r_next)
            x_hat = x_hat - time_diff * u

            if clip_intermediate_actions:
                x_hat = x_hat.clamp(*self.act_range)

            if record_intermediate:
                x_hat_list[i] = x_hat

        x_hat = x_hat.clamp(*self.act_range)

        return Sample(
            trajectories=x_hat,
            chains=x_hat_list if record_intermediate else None
        )


class ImprovedMeanFlowDispersive(nn.Module):
    """
    Improved MeanFlow (iMF) with Dispersive Loss regularization.

    Combines the v-loss formulation from iMF with dispersive loss regularization
    to encourage representation diversity in hidden spaces.

    Key features:
    1. iMF v-loss: Network-independent regression target for stable training
    2. Dispersive loss: Encourages diverse internal representations
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
        # MeanFlow specific parameters
        flow_ratio: float = 0.5,
        gamma: float = 0.5,
        c: float = 1e-3,
        sample_t_type: str = 'uniform',
        use_adaptive_loss: bool = False,
        # iMF specific parameters
        use_auxiliary_head: bool = False,
        # Dispersive loss parameters
        dispersive_loss_weight: float = 0.1,
        dispersive_loss_type: str = "infonce_l2",
        dispersive_temperature: float = 0.5,
        dispersive_margin: float = 1.0,
        apply_dispersive_to_embeddings: bool = True,
        **kwargs
    ):
        """
        Initialize Improved MeanFlow with Dispersive Loss.

        Args:
            network: MeanFlowMLP network for average velocity prediction
            device: Device to run the model on
            horizon_steps: Number of steps in trajectory horizon
            action_dim: Dimension of action space
            act_min: Minimum action value for clipping
            act_max: Maximum action value for clipping
            obs_dim: Dimension of observation space
            max_denoising_steps: Maximum denoising steps
            seed: Random seed for reproducibility
            flow_ratio: Ratio of samples using flow consistency constraint
            gamma: Adaptive loss parameter
            c: Stability constant for adaptive loss
            sample_t_type: Time sampling type
            use_adaptive_loss: Whether to use adaptive L2 loss
            use_auxiliary_head: Whether to use auxiliary head for v prediction
            dispersive_loss_weight: Weight coefficient for dispersive loss
            dispersive_loss_type: Type of dispersive loss
            dispersive_temperature: Temperature for InfoNCE variants
            dispersive_margin: Margin for hinge loss
            apply_dispersive_to_embeddings: Apply dispersive loss to embeddings
        """
        super().__init__()

        # Import dispersive loss
        from model.loss.dispersive_loss import DispersiveLoss

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

        # MeanFlow parameters
        self.flow_ratio = flow_ratio
        self.gamma = gamma
        self.c = c
        self.sample_t_type = sample_t_type
        self.use_adaptive_loss = use_adaptive_loss

        # iMF specific
        self.use_auxiliary_head = use_auxiliary_head

        # Dispersive loss configuration
        self.dispersive_loss_weight = dispersive_loss_weight
        self.apply_dispersive_to_embeddings = apply_dispersive_to_embeddings

        # Initialize dispersive loss
        self.dispersive_loss = DispersiveLoss(
            loss_type=dispersive_loss_type,
            temperature=dispersive_temperature,
            margin=dispersive_margin,
            weight=dispersive_loss_weight,
        )

        log.info(f"Initialized Improved MeanFlow with Dispersive Loss: "
                f"weight={dispersive_loss_weight}, type={dispersive_loss_type}")

    def get_predicted_velocity(self, xt: Tensor, t: Tensor, cond: dict) -> Tensor:
        """
        Get predicted marginal velocity v_θ using boundary condition.
        v_θ(z_t, t) = u_θ(z_t, t, t)
        """
        return self.network(xt, t, t, cond)

    def loss(self, x1: Tensor, cond: dict, **kwargs) -> Tensor:
        """
        Compute iMF v-loss with dispersive regularization.

        Args:
            x1: Real action trajectories (B, Ta, Da)
            cond: Condition dictionary with 'state' key

        Returns:
            Total loss (iMF v-loss + dispersive loss)
        """
        batch_size = x1.shape[0]

        # Normalize action data
        x1 = torch.clamp(x1, *self.act_range)

        # Sample time pairs (t, r) using lognormal distribution
        mu, sigma = -0.4, 1.0
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))

        # Ensure r <= t
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # Flow consistency for some samples (r = t)
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=self.device)
        r = torch.tensor(r_np, device=self.device)

        t_ = t.view(-1, 1, 1)
        r_ = r.view(-1, 1, 1)

        # Sample noise and generate trajectory
        x0 = torch.randn_like(x1)
        xt = (1 - t_) * x1 + t_ * x0

        # Ground truth instantaneous velocity
        v_gt = x0 - x1

        # === iMF v-loss computation ===
        # Step 1: Get predicted velocity v_θ
        with torch.no_grad():
            v_pred = self.get_predicted_velocity(xt, t, cond)

        # Step 2: Compute JVP with predicted velocity as tangent
        def network_fn(z, t_val, r_val):
            return self.network(z, t_val, r_val, cond)

        u, dudt = torch.autograd.functional.jvp(
            network_fn,
            (xt, t, r),
            (v_pred, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True
        )

        # Step 3: Compute composite function V_θ
        V_theta = u + (t_ - r_) * stopgrad(dudt)

        # Step 4: Compute base v-loss
        if self.use_adaptive_loss:
            error = V_theta - v_gt
            base_loss = adaptive_l2_loss(error, gamma=self.gamma, c=self.c)
        else:
            base_loss = F.mse_loss(V_theta, v_gt)

        # === Dispersive loss computation ===
        if self.dispersive_loss_weight <= 0 or batch_size < 2:
            return base_loss

        dispersive_losses = []

        try:
            # Check if network supports output_embedding
            if hasattr(self.network, 'forward'):
                forward_code = self.network.forward.__code__
                if 'output_embedding' in forward_code.co_varnames:
                    try:
                        result = self.network.forward(
                            x1, t, r, cond, output_embedding=True
                        )

                        if isinstance(result, tuple) and len(result) > 1:
                            # Extract embeddings from network output
                            if self.apply_dispersive_to_embeddings:
                                # result format: (output, time_emb, r_emb, cond_emb)
                                if len(result) >= 4:
                                    cond_emb = result[3]
                                    if cond_emb is not None:
                                        cond_emb_flat = cond_emb.view(batch_size, -1)
                                        cond_dispersive = self.dispersive_loss(cond_emb_flat)
                                        dispersive_losses.append(cond_dispersive)

                    except TypeError:
                        pass

        except Exception as e:
            log.warning(f"Failed to compute dispersive loss for iMF: {e}")

        total_dispersive_loss = sum(dispersive_losses) if dispersive_losses else torch.tensor(0.0, device=self.device)

        return base_loss + total_dispersive_loss

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int = 5,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None
    ) -> Sample:
        """
        Sample trajectories using MeanFlow multi-step sampling.
        Sampling is identical to base ImprovedMeanFlow.
        """
        B = cond['state'].shape[0]

        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)

        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)

        t_vals = torch.linspace(1.0, 0.0, inference_steps + 1, device=self.device)

        for i in range(inference_steps):
            t_curr = t_vals[i]
            r_next = t_vals[i + 1]

            t = torch.full((B,), t_curr, device=self.device)
            r = torch.full((B,), r_next, device=self.device)

            u = self.network(x_hat, t, r, cond)

            time_diff = (t_curr - r_next)
            x_hat = x_hat - time_diff * u

            if clip_intermediate_actions:
                x_hat = x_hat.clamp(*self.act_range)

            if record_intermediate:
                x_hat_list[i] = x_hat

        x_hat = x_hat.clamp(*self.act_range)

        return Sample(
            trajectories=x_hat,
            chains=x_hat_list if record_intermediate else None
        )

    def get_dispersive_loss_info(self):
        """Get information about dispersive loss configuration."""
        return {
            "dispersive_loss_weight": self.dispersive_loss_weight,
            "dispersive_loss_type": self.dispersive_loss.loss_type,
            "dispersive_temperature": self.dispersive_loss.temperature,
            "dispersive_margin": self.dispersive_loss.margin,
            "apply_to_embeddings": self.apply_dispersive_to_embeddings,
        }
