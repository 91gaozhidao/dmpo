# MIT License
# Copyright (c) 2025 ReinFlow Authors

"""
Phase 3 & 6: GRPO Buffer Tests.

Tests cover:
- Group advantage computation with Z-score normalization
- Zero-variance protection (all-zero returns)
- Population std (unbiased=False) correctness
- Remainder group handling
- make_dataset flattening and device placement
- Trajectory add/clear lifecycle
"""

import pytest
import torch
import numpy as np
from agent.finetune.grpo.buffer import GRPOBuffer, ADVANTAGE_STD_THRESHOLD


# ---------------------------------------------------------------------------
# Phase 3: Zero-Variance Advantage Protection
# ---------------------------------------------------------------------------

class TestGRPOBufferZeroVariance:
    """Stress test for zero-variance protection in compute_group_advantages."""

    def test_all_zero_returns_produce_zero_advantages(self):
        """Phase 3: Inject all-zero episodic returns."""
        buf = GRPOBuffer(group_size=4)
        # Add 4 trajectories with zero return (one full group)
        for _ in range(4):
            buf.add_trajectory(
                obs_seq=torch.randn(10, 1, 11),
                act_seq=torch.randn(10, 4, 3),
                log_prob_seq=torch.randn(10),
                episodic_return=0.0,
            )
        advantages = buf.compute_group_advantages()
        assert torch.allclose(advantages, torch.zeros(4)), (
            f"All-zero returns should produce all-zero advantages, got {advantages}"
        )

    def test_identical_nonzero_returns_produce_zero_advantages(self):
        """All-identical returns (e.g., all 5.0) have zero variance."""
        buf = GRPOBuffer(group_size=4)
        for _ in range(4):
            buf.add_trajectory(
                obs_seq=torch.randn(5, 1, 11),
                act_seq=torch.randn(5, 4, 3),
                log_prob_seq=torch.randn(5),
                episodic_return=5.0,
            )
        advantages = buf.compute_group_advantages()
        assert torch.allclose(advantages, torch.zeros(4)), (
            f"Identical returns should produce zero advantages, got {advantages}"
        )

    def test_near_zero_std_triggers_protection(self):
        """Returns with std < ADVANTAGE_STD_THRESHOLD should trigger protection."""
        buf = GRPOBuffer(group_size=4)
        tiny_diff = ADVANTAGE_STD_THRESHOLD * 0.1  # Below threshold
        for i in range(4):
            buf.add_trajectory(
                obs_seq=torch.randn(5, 1, 11),
                act_seq=torch.randn(5, 4, 3),
                log_prob_seq=torch.randn(5),
                episodic_return=1.0 + i * tiny_diff,
            )
        advantages = buf.compute_group_advantages()
        # Population std of [1.0, 1.0+e, 1.0+2e, 1.0+3e] with tiny e
        # should be below ADVANTAGE_STD_THRESHOLD
        assert torch.allclose(advantages, torch.zeros(4), atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 6: Normal Group Advantage Tests
# ---------------------------------------------------------------------------

class TestGRPOBufferAdvantages:

    def test_basic_advantage_normalization(self):
        """Verify Z-score normalization for a simple group."""
        buf = GRPOBuffer(group_size=4)
        returns = [1.0, 2.0, 3.0, 4.0]
        for r in returns:
            buf.add_trajectory(
                obs_seq=torch.randn(5, 1, 11),
                act_seq=torch.randn(5, 4, 3),
                log_prob_seq=torch.randn(5),
                episodic_return=r,
            )
        advantages = buf.compute_group_advantages()

        # Verify zero-mean
        assert abs(advantages.mean().item()) < 1e-4, (
            f"Advantages should be zero-mean, got mean={advantages.mean()}"
        )
        # Verify correct ordering
        assert advantages[0] < advantages[1] < advantages[2] < advantages[3]

    def test_population_std_used(self):
        """Verify population std (unbiased=False) is used, not sample std."""
        buf = GRPOBuffer(group_size=2)
        buf.add_trajectory(
            obs_seq=torch.randn(3, 1, 11), act_seq=torch.randn(3, 4, 3),
            log_prob_seq=torch.randn(3), episodic_return=0.0,
        )
        buf.add_trajectory(
            obs_seq=torch.randn(3, 1, 11), act_seq=torch.randn(3, 4, 3),
            log_prob_seq=torch.randn(3), episodic_return=2.0,
        )
        advantages = buf.compute_group_advantages()

        # Population std of [0.0, 2.0] = 1.0
        # Z-scores: (0 - 1) / (1 + 1e-6) = -1, (2 - 1) / (1 + 1e-6) = 1
        expected = torch.tensor([-1.0, 1.0]) / (1.0 + ADVANTAGE_STD_THRESHOLD)
        assert torch.allclose(advantages, expected, atol=1e-5)

    def test_multiple_complete_groups(self):
        """Two groups with different returns should be normalized independently."""
        buf = GRPOBuffer(group_size=2)
        # Group 1: returns [0, 10]
        buf.add_trajectory(
            obs_seq=torch.randn(3, 1, 11), act_seq=torch.randn(3, 4, 3),
            log_prob_seq=torch.randn(3), episodic_return=0.0,
        )
        buf.add_trajectory(
            obs_seq=torch.randn(3, 1, 11), act_seq=torch.randn(3, 4, 3),
            log_prob_seq=torch.randn(3), episodic_return=10.0,
        )
        # Group 2: returns [100, 200]
        buf.add_trajectory(
            obs_seq=torch.randn(3, 1, 11), act_seq=torch.randn(3, 4, 3),
            log_prob_seq=torch.randn(3), episodic_return=100.0,
        )
        buf.add_trajectory(
            obs_seq=torch.randn(3, 1, 11), act_seq=torch.randn(3, 4, 3),
            log_prob_seq=torch.randn(3), episodic_return=200.0,
        )
        advantages = buf.compute_group_advantages()

        # Each group's advantages should be normalized independently
        assert advantages[0] < 0  # Below group 1 mean
        assert advantages[1] > 0  # Above group 1 mean
        assert advantages[2] < 0  # Below group 2 mean
        assert advantages[3] > 0  # Above group 2 mean

    def test_remainder_group_handling(self):
        """Trajectories that don't fill a complete group should be handled."""
        buf = GRPOBuffer(group_size=3)
        # 5 trajectories: 1 complete group of 3, 1 remainder group of 2
        for i in range(5):
            buf.add_trajectory(
                obs_seq=torch.randn(3, 1, 11), act_seq=torch.randn(3, 4, 3),
                log_prob_seq=torch.randn(3), episodic_return=float(i),
            )
        advantages = buf.compute_group_advantages()
        assert advantages.shape == (5,)
        assert torch.isfinite(advantages).all()


# ---------------------------------------------------------------------------
# Phase 6: Buffer Lifecycle Tests
# ---------------------------------------------------------------------------

class TestGRPOBufferLifecycle:

    def test_clear_resets_storage(self):
        buf = GRPOBuffer(group_size=4)
        buf.add_trajectory(
            obs_seq=torch.randn(5, 1, 11), act_seq=torch.randn(5, 4, 3),
            log_prob_seq=torch.randn(5), episodic_return=1.0,
        )
        assert len(buf.obs) == 1
        buf.clear()
        assert len(buf.obs) == 0
        assert len(buf.actions) == 0
        assert len(buf.old_log_probs) == 0
        assert len(buf.group_returns) == 0

    def test_make_dataset_shapes(self):
        buf = GRPOBuffer(group_size=2)
        T = 5
        for r in [1.0, 3.0]:
            buf.add_trajectory(
                obs_seq=torch.randn(T, 1, 11),
                act_seq=torch.randn(T, 4, 3),
                log_prob_seq=torch.randn(T),
                episodic_return=r,
            )
        obs, actions, old_lps, advs = buf.make_dataset()
        total_steps = T * 2
        assert obs["state"].shape == (total_steps, 1, 11)
        assert actions.shape == (total_steps, 4, 3)
        assert old_lps.shape == (total_steps,)
        assert advs.shape == (total_steps,)

    def test_make_dataset_advantage_broadcast(self):
        """Trajectory-level advantage should be broadcast to all steps."""
        buf = GRPOBuffer(group_size=2)
        T1, T2 = 3, 5
        buf.add_trajectory(
            obs_seq=torch.randn(T1, 1, 11), act_seq=torch.randn(T1, 4, 3),
            log_prob_seq=torch.randn(T1), episodic_return=0.0,
        )
        buf.add_trajectory(
            obs_seq=torch.randn(T2, 1, 11), act_seq=torch.randn(T2, 4, 3),
            log_prob_seq=torch.randn(T2), episodic_return=2.0,
        )
        _, _, _, advs = buf.make_dataset()
        # First T1 steps should have same advantage
        assert torch.allclose(advs[:T1], advs[0].expand(T1))
        # Last T2 steps should have same advantage
        assert torch.allclose(advs[T1:], advs[T1].expand(T2))

    def test_make_dataset_device_placement(self):
        buf = GRPOBuffer(group_size=2, device=torch.device("cpu"))
        for r in [1.0, 2.0]:
            buf.add_trajectory(
                obs_seq=torch.randn(3, 1, 11),
                act_seq=torch.randn(3, 4, 3),
                log_prob_seq=torch.randn(3),
                episodic_return=r,
            )
        obs, actions, old_lps, advs = buf.make_dataset(device=torch.device("cpu"))
        assert obs["state"].device.type == "cpu"
        assert actions.device.type == "cpu"

    def test_summarize_episode_reward(self):
        buf = GRPOBuffer(group_size=2)
        buf.add_trajectory(
            obs_seq=torch.randn(3, 1, 11), act_seq=torch.randn(3, 4, 3),
            log_prob_seq=torch.randn(3), episodic_return=10.0,
        )
        buf.add_trajectory(
            obs_seq=torch.randn(3, 1, 11), act_seq=torch.randn(3, 4, 3),
            log_prob_seq=torch.randn(3), episodic_return=0.0,
        )
        buf.summarize_episode_reward()
        assert buf.avg_episode_reward == 5.0
        assert buf.success_rate == 0.5  # One >0, one ==0

    def test_empty_buffer_compute_advantages(self):
        """Empty buffer should return empty advantage tensor."""
        buf = GRPOBuffer(group_size=4)
        advantages = buf.compute_group_advantages()
        assert advantages.shape == (0,)
