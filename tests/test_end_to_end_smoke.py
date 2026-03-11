# MIT License
# Copyright (c) 2025 ReinFlow Authors

"""
Phase 5: Single-step End-to-End Smoke Tests.

Tests cover:
- GRPODrifting full forward-backward cycle with mock environment data
- NoisyDriftingPolicy action collection → advantage computation → policy update
- Gradient validity (non-None, non-NaN) after update
- Reference policy gradient isolation after update
- GRPO buffer integration: collect → normalize → make_dataset → update cycle
"""

import pytest
import torch
import numpy as np
from model.flow.mlp_meanflow import MeanFlowMLP
from model.drifting.drifting import DriftingPolicy
from model.drifting.ft_grpo.grpodrifting import GRPODrifting, NoisyDriftingPolicy
from agent.finetune.grpo.buffer import GRPOBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_network(horizon_steps=4, action_dim=3, obs_dim=11):
    return MeanFlowMLP(
        horizon_steps=horizon_steps,
        action_dim=action_dim,
        cond_dim=obs_dim,
        time_dim=16,
        r_embedding_dim=16,
        mlp_dims=[64, 64],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    )


def _make_full_grpo_stack(
    horizon_steps=4, action_dim=3, obs_dim=11,
    group_size=4, beta=0.01, epsilon=0.2,
):
    """Create a full GRPO stack: DriftingPolicy → NoisyDriftingPolicy → GRPODrifting."""
    device = torch.device("cpu")
    network = _make_network(horizon_steps, action_dim, obs_dim)
    dp = DriftingPolicy(
        network=network, device=device,
        horizon_steps=horizon_steps, action_dim=action_dim,
        act_min=-1.0, act_max=1.0, obs_dim=obs_dim,
        max_denoising_steps=1, seed=42,
    )
    noisy = NoisyDriftingPolicy(
        drifting_policy=dp, action_dim=action_dim,
        horizon_steps=horizon_steps, init_log_std=-0.5,
    )
    grpo = GRPODrifting(
        actor=noisy, beta=beta, epsilon=epsilon,
        act_min=-1.0, act_max=1.0,
    )
    buffer = GRPOBuffer(group_size=group_size, device=device)
    return grpo, buffer


# ---------------------------------------------------------------------------
# Phase 5: End-to-End Smoke Test
# ---------------------------------------------------------------------------

class TestEndToEndGRPOSmoke:
    """Simulated end-to-end GRPO training with mock environment data."""

    def test_full_grpo_training_cycle(self):
        """
        End-to-end test:
        1. Collect group trajectories (mock)
        2. Compute group advantages
        3. Run policy update
        4. Verify gradients are valid
        5. Verify reference policy has no gradients
        """
        horizon_steps = 4
        action_dim = 3
        obs_dim = 11
        group_size = 4
        num_steps = 10

        grpo, buffer = _make_full_grpo_stack(
            horizon_steps=horizon_steps,
            action_dim=action_dim,
            obs_dim=obs_dim,
            group_size=group_size,
        )

        # Step 1: Collect G trajectories with mock environment data
        for g in range(group_size):
            obs_seq = torch.randn(num_steps, 1, obs_dim)
            cond = {"state": obs_seq}

            # Collect actions and log probs
            actions_list = []
            log_probs_list = []
            for t in range(num_steps):
                step_cond = {"state": obs_seq[t:t+1]}
                with torch.no_grad():
                    action, log_prob = grpo.actor.sample_action(step_cond)
                actions_list.append(action.squeeze(0))
                log_probs_list.append(log_prob.squeeze(0))

            act_seq = torch.stack(actions_list, dim=0)
            lp_seq = torch.stack(log_probs_list, dim=0)

            # Mock reward: random episodic return
            episodic_return = float(np.random.randn() * 10 + 5)

            buffer.add_trajectory(
                obs_seq=obs_seq,
                act_seq=act_seq,
                log_prob_seq=lp_seq,
                episodic_return=episodic_return,
            )

        # Step 2: Make dataset
        all_obs, all_actions, all_old_lps, all_advantages = buffer.make_dataset()

        # Verify dataset shapes
        total_steps = num_steps * group_size
        assert all_obs["state"].shape == (total_steps, 1, obs_dim)
        assert all_actions.shape == (total_steps, horizon_steps, action_dim)
        assert all_old_lps.shape == (total_steps,)
        assert all_advantages.shape == (total_steps,)

        # Step 3: Run one policy update
        optimizer = torch.optim.AdamW(grpo.actor.parameters(), lr=1e-3)

        batch_size = min(16, total_steps)
        idx = torch.randperm(total_steps)[:batch_size]
        batch_obs = {"state": all_obs["state"][idx]}
        batch_actions = all_actions[idx]
        batch_advantages = all_advantages[idx]
        batch_old_lps = all_old_lps[idx]

        loss, metrics = grpo.compute_loss(
            batch_obs, batch_actions, batch_advantages, batch_old_lps
        )

        optimizer.zero_grad()
        loss.backward()

        # Step 4: Verify gradients are valid
        # Note: Not all params may get gradients if they are inside
        # DriftingPolicy.sample() which runs under torch.no_grad().
        # The key learnable param (log_std) must have valid gradients.
        has_any_grad = False
        for name, param in grpo.actor.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"Actor param {name} has NaN gradient"
                )
                if param.grad.abs().sum() > 0:
                    has_any_grad = True
        assert has_any_grad, "At least some actor params must have non-zero gradients"
        # log_std specifically must have gradients (key GRPO learnable param)
        assert grpo.actor.log_std.grad is not None, "log_std must have gradient"
        assert not torch.isnan(grpo.actor.log_std.grad).any(), "log_std grad has NaN"

        # Step 5: Verify reference policy has no gradients
        for name, param in grpo.actor_ref.named_parameters():
            assert param.grad is None or param.grad.abs().sum() == 0, (
                f"Reference policy param {name} should not have gradient"
            )

        # Step 6: Apply update
        torch.nn.utils.clip_grad_norm_(grpo.actor.parameters(), 1.0)
        optimizer.step()

        # Verify metrics are reasonable
        assert torch.isfinite(loss)
        for k, v in metrics.items():
            assert np.isfinite(v), f"Metric {k} is not finite: {v}"

    def test_multiple_update_epochs(self):
        """Test multiple update epochs (as in real GRPO training)."""
        grpo, buffer = _make_full_grpo_stack(group_size=4)
        num_steps = 5

        # Collect trajectories
        for g in range(4):
            obs_seq = torch.randn(num_steps, 1, 11)
            act_seq = torch.randn(num_steps, 4, 3).tanh()
            lp_seq = torch.randn(num_steps)
            buffer.add_trajectory(obs_seq, act_seq, lp_seq, float(g * 2.0))

        optimizer = torch.optim.AdamW(grpo.actor.parameters(), lr=1e-3)

        # Multiple epochs
        for epoch in range(3):
            all_obs, all_actions, all_old_lps, all_advs = buffer.make_dataset()
            N = all_actions.shape[0]
            perm = torch.randperm(N)

            for start in range(0, N, 8):
                end = min(start + 8, N)
                idx = perm[start:end]
                batch_obs = {"state": all_obs["state"][idx]}

                loss, metrics = grpo.compute_loss(
                    batch_obs, all_actions[idx],
                    all_advs[idx], all_old_lps[idx],
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(grpo.actor.parameters(), 1.0)
                optimizer.step()

        # Should complete without errors

    def test_beta_decay(self):
        """Test KL penalty coefficient decay."""
        grpo, _ = _make_full_grpo_stack()
        initial_beta = grpo.beta
        decay_rate = 0.995
        beta_min = 0.001

        for _ in range(100):
            grpo.beta = max(grpo.beta * decay_rate, beta_min)

        assert grpo.beta < initial_beta
        assert grpo.beta >= beta_min


# ---------------------------------------------------------------------------
# Phase 5: Pretrain Smoke Test
# ---------------------------------------------------------------------------

class TestPretrainDriftingSmoke:
    """Smoke test for offline pretraining of DriftingPolicy."""

    def test_pretrain_forward_backward(self):
        """Test a single pretraining step: loss → backward → step."""
        device = torch.device("cpu")
        network = _make_network()
        policy = DriftingPolicy(
            network=network, device=device,
            horizon_steps=4, action_dim=3,
            act_min=-1.0, act_max=1.0, obs_dim=11,
            max_denoising_steps=1, seed=42,
        )

        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        # Mock expert data
        B = 16
        x1 = torch.randn(B, 4, 3).clamp(-1, 1)
        cond = {"state": torch.randn(B, 1, 11)}

        # Forward
        loss = policy.loss(x1, cond)
        assert torch.isfinite(loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        for name, p in policy.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

        # Step
        optimizer.step()

    def test_pretrain_loss_decreases(self):
        """Loss should decrease over multiple training steps."""
        device = torch.device("cpu")
        network = _make_network()
        policy = DriftingPolicy(
            network=network, device=device,
            horizon_steps=4, action_dim=3,
            act_min=-1.0, act_max=1.0, obs_dim=11,
            max_denoising_steps=1, seed=42,
        )
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        # Fixed training data
        torch.manual_seed(0)
        B = 32
        x1 = torch.randn(B, 4, 3).clamp(-1, 1)
        cond = {"state": torch.randn(B, 1, 11)}

        losses = []
        for _ in range(20):
            loss = policy.loss(x1, cond)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (compare first vs last)
        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Phase 6: PPO Drifting Smoke Test
# ---------------------------------------------------------------------------

class TestPPODriftingSmokeTest:
    """Smoke test for PPO fine-tuning of Drifting Policy."""

    def test_ppo_noisy_mlp_forward_backward(self):
        """Test forward-backward through NoisyDriftingMLP."""
        from model.drifting.ft_ppo.ppodrifting import NoisyDriftingMLP

        device = torch.device("cpu")
        policy = _make_network()
        noisy = NoisyDriftingMLP(
            policy=policy, denoising_steps=1,
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

        optimizer = torch.optim.AdamW(noisy.parameters(), lr=1e-3)

        B = 8
        z = torch.randn(B, 4, 3)
        cond = {"state": torch.randn(B, 1, 11)}

        action, noise_std = noisy.forward(z, cond, learn_exploration_noise=True, step=0)
        loss = action.sum() + noise_std.sum()

        optimizer.zero_grad()
        loss.backward()

        for name, p in noisy.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

        optimizer.step()
