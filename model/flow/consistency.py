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
Consistency Model Policy Implementation for ReinFlow

Implements Consistency Distillation (CD) from a pre-trained Reflow teacher.
Based on:
- "Consistency Models" (Song et al., 2023)
- "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation" (RSS 2024)

The key idea is to distill a multi-step Reflow/Diffusion model into a single-step
consistency model that can generate high-quality samples in one forward pass.
"""

import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from copy import deepcopy

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class ConsistencyModel(nn.Module):
    """
    Consistency Model for policy learning via distillation from Reflow teacher.

    The consistency model learns to map any point on the ODE trajectory directly
    to the clean data (t=1), enabling single-step generation.

    Training uses Consistency Distillation (CD):
    1. Sample time t uniformly from [0, 1)
    2. Get x_t by interpolating between noise x_0 and data x_1
    3. Use teacher (Reflow) to do one ODE step from t to t+dt, getting x_{t+dt}
    4. Train student to satisfy: f_θ(x_t, t) ≈ f_θ(x_{t+dt}, t+dt)
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        # Consistency model parameters
        num_training_steps: int = 40,  # Number of discretization steps for training
        sigma_min: float = 0.002,  # Minimum noise level (for boundary condition)
        sigma_max: float = 1.0,  # Maximum noise level
    ):
        """
        Initialize Consistency Model.

        Args:
            network: Neural network for consistency mapping
            device: Device to run the model on
            horizon_steps: Number of steps in trajectory horizon
            action_dim: Dimension of action space
            act_min: Minimum action value
            act_max: Maximum action value
            obs_dim: Dimension of observation space
            max_denoising_steps: Number of sampling steps at inference
            seed: Random seed
            num_training_steps: Number of discretization steps N for training
            sigma_min: Minimum sigma for boundary condition
            sigma_max: Maximum sigma
        """
        super().__init__()

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

        # Consistency model parameters
        self.num_training_steps = num_training_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Teacher model (will be loaded separately)
        self.teacher = None

    def set_teacher(self, teacher_model: nn.Module):
        """
        Set the teacher model (pre-trained Reflow) for distillation.
        The teacher is frozen and used only for inference.
        """
        self.teacher = teacher_model.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        log.info("Teacher model loaded and frozen for consistency distillation")

    def get_time_schedule(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Get time discretization schedule from 0 to 1.
        Uses uniform spacing for simplicity.

        Args:
            n: Number of discretization steps
            device: Device

        Returns:
            Tensor of shape (n+1,) with times from 0 to 1
        """
        return torch.linspace(0, 1, n + 1, device=device)

    def teacher_ode_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond: dict
    ) -> torch.Tensor:
        """
        Perform one ODE step using the teacher model.

        Supports both Reflow and ShortCutFlow teachers:
        - Reflow: v_t = network(x_t, t, cond)
        - ShortCutFlow: v_t = network(x_t, t, d, cond) where d is step size

        Euler step: x_{t+dt} = x_t + v_θ(x_t, t) * dt

        Args:
            x_t: Current state at time t
            t: Current time (B,)
            t_next: Next time (B,)
            cond: Condition dictionary

        Returns:
            x_{t+dt}: State at next time
        """
        dt = t_next - t
        dt_expanded = dt.view(-1, 1, 1)

        # Check if teacher is ShortCutFlow (has 'd' parameter for step size)
        # ShortCutFlow network expects: forward(x, t, d, cond)
        # Reflow network expects: forward(x, t, cond)
        if hasattr(self.teacher, 'self_consistency_k'):
            # ShortCutFlow teacher - pass d parameter
            # For distillation, we use d=dt (the actual step size)
            d = dt  # (B,) tensor
            v_t = self.teacher.network.forward(x_t, t, d, cond)
        else:
            # Reflow teacher
            v_t = self.teacher.network(x_t, t, cond)

        # Euler step
        x_next = x_t + v_t * dt_expanded

        return x_next

    def consistency_function(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: dict
    ) -> torch.Tensor:
        """
        Apply consistency function f_θ(x_t, t) that maps to the trajectory endpoint.

        Uses skip connection parameterization for boundary condition:
        f_θ(x_t, t) = c_skip(t) * x_t + c_out(t) * F_θ(x_t, t)

        At t=1 (data): c_skip(1) = 1, c_out(1) = 0 → f(x_1, 1) = x_1
        At t=0 (noise): c_skip(0) ≈ 0, c_out(0) ≈ 1 → f(x_0, 0) = F_θ(x_0, 0)

        Args:
            x_t: Input at time t
            t: Time (B,)
            cond: Condition dictionary

        Returns:
            Predicted clean data x_1
        """
        # Simple linear skip connection that satisfies boundary condition
        # c_skip = t, c_out = 1 - t
        # At t=1: f = x_1 (identity)
        # At t=0: f = F_θ(x_0, 0)

        t_expanded = t.view(-1, 1, 1)

        # Network output
        F_out = self.network(x_t, t, cond)

        # Skip connection: f = t * x_t + (1-t) * F_θ
        # This ensures f(x_1, 1) = x_1 (boundary condition)
        output = t_expanded * x_t + (1 - t_expanded) * F_out

        return output

    def loss(self, x1: Tensor, cond: dict) -> Tensor:
        """
        Compute Consistency Distillation loss using Reflow teacher.

        The loss enforces that adjacent points on the ODE trajectory
        map to the same output through the consistency function.

        Args:
            x1: Clean action trajectories (B, Ta, Da) - the "data" at t=1
            cond: Condition dictionary with 'state' key

        Returns:
            Scalar loss tensor
        """
        if self.teacher is None:
            raise RuntimeError("Teacher model not set! Call set_teacher() first.")

        batch_size = x1.shape[0]
        device = x1.device

        # Clamp to valid range
        x1 = torch.clamp(x1, *self.act_range)

        # Sample random time indices n ∈ {0, 1, ..., N-1}
        n = torch.randint(0, self.num_training_steps, (batch_size,), device=device)

        # Get time values: t_n and t_{n+1}
        times = self.get_time_schedule(self.num_training_steps, device)
        t_n = times[n]  # Current time
        t_n1 = times[n + 1]  # Next time (closer to data)

        # Sample noise x_0
        x0 = torch.randn_like(x1)

        # Interpolate to get x at time t_n
        # Using Reflow convention: x_t = t * x_1 + (1-t) * x_0
        t_n_expanded = t_n.view(-1, 1, 1)
        x_t_n = t_n_expanded * x1 + (1 - t_n_expanded) * x0

        # Use teacher to do one ODE step from t_n to t_{n+1}
        with torch.no_grad():
            x_t_n1 = self.teacher_ode_step(x_t_n, t_n, t_n1, cond)

        # Consistency model outputs
        # Student prediction at t_n
        pred_n = self.consistency_function(x_t_n, t_n, cond)

        # Target: consistency model at t_{n+1} (with stop gradient)
        with torch.no_grad():
            target = self.consistency_function(x_t_n1, t_n1, cond)

        # MSE loss between predictions
        loss = F.mse_loss(pred_n, target)

        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int = 1,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None
    ) -> Sample:
        """
        Sample trajectories using the consistency model.

        Single-step sampling: directly map noise to data.
        Multi-step sampling: iteratively refine (can improve quality).

        Args:
            cond: Condition dictionary with 'state' key
            inference_steps: Number of sampling steps (1 for single-step)
            record_intermediate: Whether to record intermediate steps
            clip_intermediate_actions: Whether to clip actions
            z: Initial noise (if None, sample from Gaussian)

        Returns:
            Sample namedtuple with trajectories and optional chains
        """
        B = cond['state'].shape[0]

        if record_intermediate:
            x_hat_list = []

        # Start from noise at t=0
        if z is not None:
            x_hat = z
        else:
            x_hat = torch.randn((B,) + self.data_shape, device=self.device)

        if inference_steps == 1:
            # Single-step generation: directly map noise to data
            t = torch.zeros(B, device=self.device)
            x_hat = self.consistency_function(x_hat, t, cond)

            if record_intermediate:
                x_hat_list.append(x_hat.clone())
        else:
            # Multi-step generation for improved quality
            times = self.get_time_schedule(inference_steps, self.device)

            for i in range(inference_steps):
                t_curr = times[i]
                t_batch = torch.full((B,), t_curr.item(), device=self.device)

                # Apply consistency function
                x_hat = self.consistency_function(x_hat, t_batch, cond)

                if i < inference_steps - 1:
                    # Add noise for next step (stochastic chaining)
                    t_next = times[i + 1]
                    # Re-noise to t_next level
                    noise = torch.randn_like(x_hat)
                    t_next_expanded = t_next.view(1, 1, 1)
                    x_hat = t_next_expanded * x_hat + (1 - t_next_expanded) * noise

                if clip_intermediate_actions:
                    x_hat = x_hat.clamp(*self.act_range)

                if record_intermediate:
                    x_hat_list.append(x_hat.clone())

        # Final clipping
        x_hat = x_hat.clamp(*self.act_range)

        return Sample(
            trajectories=x_hat,
            chains=torch.stack(x_hat_list) if record_intermediate else None
        )
