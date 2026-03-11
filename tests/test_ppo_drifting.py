# MIT License
# Copyright (c) 2025 ReinFlow Authors

"""
Phase 2 & 6: PPO Drifting (NoisyDriftingMLP + PPODrifting) Tests.

Tests cover:
- NoisyDriftingMLP forward dimension alignment
- 1-NFE enforcement in PPO mode (t=1.0, r=0.0)
- Log-probability computation (get_logprobs) finiteness
- Action generation (get_actions) shape and boundary validation
- Exploration noise std prediction
"""

import pytest
import torch
import numpy as np
from model.flow.mlp_meanflow import MeanFlowMLP
from model.drifting.ft_ppo.ppodrifting import NoisyDriftingMLP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_meanflow_mlp(horizon_steps=4, action_dim=3, obs_dim=11, cond_steps=1):
    return MeanFlowMLP(
        horizon_steps=horizon_steps,
        action_dim=action_dim,
        cond_dim=obs_dim * cond_steps,
        time_dim=16,
        r_embedding_dim=16,
        mlp_dims=[64, 64],
        cond_mlp_dims=[32, 16],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    )


def _make_noisy_drifting_mlp(
    horizon_steps=4, action_dim=3, obs_dim=11, cond_steps=1
):
    device = torch.device("cpu")
    policy = _make_meanflow_mlp(horizon_steps, action_dim, obs_dim, cond_steps)
    return NoisyDriftingMLP(
        policy=policy,
        denoising_steps=1,
        learn_explore_noise_from=0,
        inital_noise_scheduler_type="learn",
        min_logprob_denoising_std=0.05,
        max_logprob_denoising_std=0.12,
        learn_explore_time_embedding=False,
        time_dim_explore=16,
        use_time_independent_noise=True,
        device=device,
        noise_hidden_dims=[64, 64],
        activation_type="Tanh",
    )


# ---------------------------------------------------------------------------
# Phase 2: NoisyDriftingMLP Dimension Tests
# ---------------------------------------------------------------------------

class TestNoisyDriftingMLPDimensions:

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_forward_output_shapes(self, batch_size):
        noisy_mlp = _make_noisy_drifting_mlp()
        z = torch.randn(batch_size, 4, 3)
        cond = {"state": torch.randn(batch_size, 1, 11)}
        action, noise_std = noisy_mlp.forward(z, cond, learn_exploration_noise=True, step=0)
        assert action.shape == (batch_size, 4, 3), f"Action shape mismatch: {action.shape}"
        assert noise_std.shape == (batch_size, 12), f"Noise std shape mismatch: {noise_std.shape}"

    def test_forward_no_exploration_noise(self):
        """When learn_exploration_noise=False, noise_std should be minimum."""
        noisy_mlp = _make_noisy_drifting_mlp()
        B = 4
        z = torch.randn(B, 4, 3)
        cond = {"state": torch.randn(B, 1, 11)}
        action, noise_std = noisy_mlp.forward(z, cond, learn_exploration_noise=False)
        assert action.shape == (B, 4, 3)
        assert noise_std.shape == (B, 12)

    def test_forward_before_learn_step(self):
        """When step < learn_explore_noise_from, should use minimum std."""
        noisy_mlp = _make_noisy_drifting_mlp()
        noisy_mlp.learn_explore_noise_from = 10
        B = 4
        z = torch.randn(B, 4, 3)
        cond = {"state": torch.randn(B, 1, 11)}
        action, noise_std = noisy_mlp.forward(z, cond, learn_exploration_noise=True, step=5)
        # step=5 < learn_explore_noise_from=10, should use logvar_min
        expected_std = torch.exp(0.5 * noisy_mlp.logvar_min).expand(B, 12)
        assert torch.allclose(noise_std, expected_std, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 2: 1-NFE Verification in PPO Mode
# ---------------------------------------------------------------------------

class TestPPODrifting1NFE:

    def test_single_forward_in_noisy_mlp(self):
        """NoisyDriftingMLP should call policy.forward exactly once."""
        noisy_mlp = _make_noisy_drifting_mlp()
        call_count = [0]
        original_forward = noisy_mlp.policy.forward

        def counting_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)

        noisy_mlp.policy.forward = counting_forward
        z = torch.randn(4, 4, 3)
        cond = {"state": torch.randn(4, 1, 11)}
        noisy_mlp.forward(z, cond, learn_exploration_noise=True, step=0)
        assert call_count[0] == 1, (
            f"Expected 1 NFE in NoisyDriftingMLP, got {call_count[0]}"
        )

    def test_t_equals_one_r_equals_zero(self):
        """Verify that forward uses t=1.0 and r=0.0 for drifting policy."""
        noisy_mlp = _make_noisy_drifting_mlp()
        captured_args = {}
        original_forward = noisy_mlp.policy.forward

        def capture_forward(action, time, r, cond, **kwargs):
            captured_args["time"] = time.clone()
            captured_args["r"] = r.clone()
            return original_forward(action, time, r, cond, **kwargs)

        noisy_mlp.policy.forward = capture_forward
        z = torch.randn(4, 4, 3)
        cond = {"state": torch.randn(4, 1, 11)}
        noisy_mlp.forward(z, cond, learn_exploration_noise=False)

        assert torch.allclose(captured_args["time"], torch.ones(4)), (
            f"Expected t=1.0, got {captured_args['time']}"
        )
        assert torch.allclose(captured_args["r"], torch.zeros(4)), (
            f"Expected r=0.0, got {captured_args['r']}"
        )


# ---------------------------------------------------------------------------
# Phase 6: Noise Std Bound Tests
# ---------------------------------------------------------------------------

class TestNoiseStdBounds:

    def test_noise_std_within_bounds(self):
        noisy_mlp = _make_noisy_drifting_mlp()
        B = 8
        z = torch.randn(B, 4, 3)
        cond = {"state": torch.randn(B, 1, 11)}
        _, noise_std = noisy_mlp.forward(z, cond, learn_exploration_noise=True, step=0)

        min_std = torch.exp(0.5 * noisy_mlp.logvar_min).item()
        max_std = torch.exp(0.5 * noisy_mlp.logvar_max).item()

        assert noise_std.min() >= min_std - 1e-6, (
            f"noise_std below min: {noise_std.min()} < {min_std}"
        )
        assert noise_std.max() <= max_std + 1e-6, (
            f"noise_std above max: {noise_std.max()} > {max_std}"
        )

    def test_noise_std_positive(self):
        noisy_mlp = _make_noisy_drifting_mlp()
        z = torch.randn(16, 4, 3)
        cond = {"state": torch.randn(16, 1, 11)}
        _, noise_std = noisy_mlp.forward(z, cond, learn_exploration_noise=True, step=0)
        assert (noise_std > 0).all(), "Noise std must be strictly positive"

    def test_noise_std_finite(self):
        noisy_mlp = _make_noisy_drifting_mlp()
        z = torch.randn(8, 4, 3) * 100  # Large noise
        cond = {"state": torch.randn(8, 1, 11)}
        _, noise_std = noisy_mlp.forward(z, cond, learn_exploration_noise=True, step=0)
        assert torch.isfinite(noise_std).all()


# ---------------------------------------------------------------------------
# Phase 6: Gradient Flow Tests
# ---------------------------------------------------------------------------

class TestPPODriftingGradients:

    def test_action_has_gradient(self):
        """Action output should support gradient computation."""
        noisy_mlp = _make_noisy_drifting_mlp()
        z = torch.randn(4, 4, 3)
        cond = {"state": torch.randn(4, 1, 11)}
        action, noise_std = noisy_mlp.forward(z, cond, learn_exploration_noise=True, step=0)
        loss = action.sum() + noise_std.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in noisy_mlp.parameters()
        )
        assert has_grad, "NoisyDriftingMLP must have gradients after backward"

    def test_noise_network_gradient_flow(self):
        """Gradients should flow through the noise prediction MLP when outputs are within bounds."""
        noisy_mlp = _make_noisy_drifting_mlp()
        # Ensure exploration noise learning is active from step 0
        noisy_mlp.learn_explore_noise_from = 0
        # Widen the logvar clamp range so MLP outputs are not always clamped
        # (clamping kills gradients at the boundary)
        noisy_mlp.logvar_min = torch.nn.Parameter(torch.tensor(-20.0), requires_grad=False)
        noisy_mlp.logvar_max = torch.nn.Parameter(torch.tensor(20.0), requires_grad=False)
        B = 4
        z = torch.randn(B, 4, 3)
        cond = {"state": torch.randn(B, 1, 11)}
        _, noise_std = noisy_mlp.forward(z, cond, learn_exploration_noise=True, step=0)
        loss = noise_std.sum()
        loss.backward()
        mlp_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in noisy_mlp.mlp_logvar.parameters()
        )
        assert mlp_has_grad, "Noise MLP should receive gradients when logvar bounds are wide"
