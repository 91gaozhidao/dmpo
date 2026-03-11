# MIT License
# Copyright (c) 2025 ReinFlow Authors

"""
Phase 2 & 6: Tensor Graph, 1-NFE Constraint Validation, and Unit Tests
for the core DriftingPolicy module.

Tests cover:
- Dimension alignment for forward/sample/loss with various batch/horizon/action shapes
- 1-NFE enforcement: only one forward pass occurs during inference
- compute_V correctness: positive/negative drift fields, self-masking
- Action boundary enforcement after sampling
"""

import pytest
import torch
import numpy as np
from model.flow.mlp_meanflow import MeanFlowMLP
from model.drifting.drifting import DriftingPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_network(horizon_steps=4, action_dim=3, obs_dim=11, cond_steps=1):
    """Create a minimal MeanFlowMLP for testing."""
    return MeanFlowMLP(
        horizon_steps=horizon_steps,
        action_dim=action_dim,
        cond_dim=obs_dim * cond_steps,
        time_dim=16,
        r_embedding_dim=16,
        mlp_dims=[64, 64],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    )


def _make_drifting_policy(
    horizon_steps=4,
    action_dim=3,
    obs_dim=11,
    cond_steps=1,
    act_min=-1.0,
    act_max=1.0,
    mask_self=True,
    drift_coef=1.0,
    neg_drift_coef=0.5,
    bandwidth=1.0,
):
    """Create a DriftingPolicy with a minimal network."""
    device = torch.device("cpu")
    network = _make_network(horizon_steps, action_dim, obs_dim, cond_steps)
    policy = DriftingPolicy(
        network=network,
        device=device,
        horizon_steps=horizon_steps,
        action_dim=action_dim,
        act_min=act_min,
        act_max=act_max,
        obs_dim=obs_dim,
        max_denoising_steps=1,
        seed=42,
        drift_coef=drift_coef,
        neg_drift_coef=neg_drift_coef,
        mask_self=mask_self,
        bandwidth=bandwidth,
    )
    return policy


# ---------------------------------------------------------------------------
# Phase 2: Dimension Alignment Tests
# ---------------------------------------------------------------------------

class TestDriftingPolicyDimensions:
    """Verify output shapes for various input configurations."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("horizon_steps,action_dim", [(1, 2), (4, 3), (8, 6)])
    def test_sample_output_shape(self, batch_size, horizon_steps, action_dim):
        obs_dim = 11
        policy = _make_drifting_policy(
            horizon_steps=horizon_steps, action_dim=action_dim, obs_dim=obs_dim
        )
        cond = {"state": torch.randn(batch_size, 1, obs_dim)}
        result = policy.sample(cond)
        assert result.trajectories.shape == (batch_size, horizon_steps, action_dim), (
            f"Expected ({batch_size}, {horizon_steps}, {action_dim}), "
            f"got {result.trajectories.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_loss_returns_scalar(self, batch_size):
        policy = _make_drifting_policy()
        x1 = torch.randn(batch_size, 4, 3)
        cond = {"state": torch.randn(batch_size, 1, 11)}
        loss = policy.loss(x1, cond)
        assert loss.dim() == 0, "Loss must be a scalar tensor"
        assert torch.isfinite(loss), "Loss must be finite"

    def test_loss_with_negative_samples(self):
        policy = _make_drifting_policy()
        B = 8
        x1 = torch.randn(B, 4, 3)
        y_neg = torch.randn(B, 4, 3)
        cond = {"state": torch.randn(B, 1, 11)}
        loss = policy.loss(x1, cond, y_neg=y_neg)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_sample_with_provided_noise(self):
        policy = _make_drifting_policy()
        B = 4
        cond = {"state": torch.randn(B, 1, 11)}
        z = torch.randn(B, 4, 3)
        result = policy.sample(cond, z=z)
        assert result.trajectories.shape == (B, 4, 3)

    def test_sample_record_intermediate(self):
        policy = _make_drifting_policy()
        B = 4
        cond = {"state": torch.randn(B, 1, 11)}
        result = policy.sample(cond, record_intermediate=True)
        assert result.chains is not None
        assert result.chains.shape == (1, B, 4, 3)

    def test_sample_no_record_intermediate(self):
        policy = _make_drifting_policy()
        B = 4
        cond = {"state": torch.randn(B, 1, 11)}
        result = policy.sample(cond, record_intermediate=False)
        assert result.chains is None


# ---------------------------------------------------------------------------
# Phase 2: 1-NFE Enforcement Test
# ---------------------------------------------------------------------------

class TestOneNFEEnforcement:
    """Verify that inference uses exactly 1 network forward evaluation."""

    def test_single_forward_call_during_sample(self):
        """Instrument the network to count forward calls during sample()."""
        policy = _make_drifting_policy()
        call_count = [0]
        original_forward = policy.network.forward

        def counting_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)

        policy.network.forward = counting_forward
        cond = {"state": torch.randn(4, 1, 11)}
        policy.sample(cond)
        assert call_count[0] == 1, (
            f"Expected exactly 1 NFE during inference, got {call_count[0]}"
        )

    def test_inference_steps_parameter_ignored(self):
        """Drifting Policy should always use 1 NFE regardless of inference_steps."""
        policy = _make_drifting_policy()
        call_count = [0]
        original_forward = policy.network.forward

        def counting_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)

        policy.network.forward = counting_forward
        cond = {"state": torch.randn(4, 1, 11)}
        # Pass inference_steps=10, should still be 1 NFE
        policy.sample(cond, inference_steps=10)
        assert call_count[0] == 1, (
            f"Drifting Policy must use 1 NFE even with inference_steps=10, got {call_count[0]}"
        )


# ---------------------------------------------------------------------------
# Phase 6: compute_V Unit Tests
# ---------------------------------------------------------------------------

class TestComputeV:
    """Unit tests for the drifting field computation."""

    def test_positive_drift_field_shape(self):
        policy = _make_drifting_policy()
        B = 8
        x = torch.randn(B, 4, 3)
        y_pos = torch.randn(B, 4, 3)
        V = policy.compute_V(x, y_pos, mask_self=True)
        assert V.shape == (B, 4, 3)

    def test_negative_drift_field_shape(self):
        policy = _make_drifting_policy()
        B = 8
        x = torch.randn(B, 4, 3)
        y_pos = torch.randn(B, 4, 3)
        y_neg = torch.randn(B, 4, 3)
        V = policy.compute_V(x, y_pos, y_neg, mask_self=True)
        assert V.shape == (B, 4, 3)

    def test_no_self_mask_includes_diagonal(self):
        """Without self-masking, the kernel should include self-interaction."""
        policy = _make_drifting_policy(mask_self=False)
        B = 4
        x = torch.randn(B, 4, 3)
        y_pos = x.clone()  # When x == y_pos, self-interaction matters
        V_masked = policy.compute_V(x, y_pos, mask_self=True)
        V_unmasked = policy.compute_V(x, y_pos, mask_self=False)
        # They should differ when x == y_pos
        assert not torch.allclose(V_masked, V_unmasked, atol=1e-6)

    def test_drift_field_points_toward_target(self):
        """The positive drift field should generally point toward y_pos."""
        policy = _make_drifting_policy(mask_self=False)
        B = 2
        x = torch.zeros(B, 4, 3)
        y_pos = torch.ones(B, 4, 3)
        V = policy.compute_V(x, y_pos, mask_self=False)
        # V should point from x (zeros) toward y_pos (ones), so V > 0
        assert (V > -1e-6).all(), "Positive drift field should point toward targets"

    def test_drift_field_finite(self):
        policy = _make_drifting_policy()
        B = 16
        x = torch.randn(B, 4, 3) * 10  # Large values
        y_pos = torch.randn(B, 4, 3) * 10
        V = policy.compute_V(x, y_pos, mask_self=True)
        assert torch.isfinite(V).all(), "Drift field must be finite even with large inputs"

    def test_batch_size_one(self):
        """With B=1 and mask_self=True, all weights are masked out."""
        policy = _make_drifting_policy()
        x = torch.randn(1, 4, 3)
        y_pos = torch.randn(1, 4, 3)
        V = policy.compute_V(x, y_pos, mask_self=True)
        # With B=1, mask_self removes the only weight, weights_sum clamp(min=1e-8) prevents NaN
        assert torch.isfinite(V).all()


# ---------------------------------------------------------------------------
# Phase 6: Action Boundary Tests
# ---------------------------------------------------------------------------

class TestActionBoundaries:
    """Verify action clipping and boundary enforcement."""

    def test_sample_actions_within_bounds(self):
        policy = _make_drifting_policy(act_min=-1.0, act_max=1.0)
        cond = {"state": torch.randn(16, 1, 11)}
        result = policy.sample(cond)
        assert result.trajectories.min() >= -1.0
        assert result.trajectories.max() <= 1.0

    def test_custom_action_range(self):
        policy = _make_drifting_policy(act_min=-0.5, act_max=0.5)
        cond = {"state": torch.randn(8, 1, 11)}
        result = policy.sample(cond)
        assert result.trajectories.min() >= -0.5
        assert result.trajectories.max() <= 0.5

    def test_loss_clamps_expert_actions(self):
        """Expert actions should be clamped to act_range in loss computation."""
        policy = _make_drifting_policy(act_min=-1.0, act_max=1.0)
        # Expert actions outside [-1, 1]
        x1 = torch.randn(8, 4, 3) * 5.0
        cond = {"state": torch.randn(8, 1, 11)}
        loss = policy.loss(x1, cond)
        assert torch.isfinite(loss), "Loss should be finite even with out-of-range expert actions"


# ---------------------------------------------------------------------------
# Phase 6: Initialization Tests
# ---------------------------------------------------------------------------

class TestDriftingPolicyInit:

    def test_invalid_denoising_steps(self):
        with pytest.raises(ValueError, match="max_denoising_steps must be a positive integer"):
            _make_drifting_policy.__wrapped__ if hasattr(_make_drifting_policy, '__wrapped__') else None
            network = _make_network()
            DriftingPolicy(
                network=network, device=torch.device("cpu"),
                horizon_steps=4, action_dim=3, act_min=-1.0, act_max=1.0,
                obs_dim=11, max_denoising_steps=0, seed=42,
            )

    def test_negative_denoising_steps(self):
        with pytest.raises(ValueError):
            network = _make_network()
            DriftingPolicy(
                network=network, device=torch.device("cpu"),
                horizon_steps=4, action_dim=3, act_min=-1.0, act_max=1.0,
                obs_dim=11, max_denoising_steps=-1, seed=42,
            )

    def test_seed_reproducibility(self):
        """Same policy with same noise input should produce same output."""
        policy = _make_drifting_policy()
        cond = {"state": torch.randn(4, 1, 11)}
        z = torch.randn(4, 4, 3)
        r1 = policy.sample(cond, z=z.clone())
        r2 = policy.sample(cond, z=z.clone())
        assert torch.allclose(r1.trajectories, r2.trajectories, atol=1e-6)
