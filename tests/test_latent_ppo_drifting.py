"""Unit tests for Latent-PPO Drifting modules.

These tests validate:
1. LatentPolicyHeadLowdim initialization produces z ~ N(0, I).
2. LatentPPODrifting frozen generator + z-policy integration.
3. PPO logprob computation consistency.
4. PPODriftingLatentBuffer GAE computation.
5. LatentPPODrifting loss computation shapes and values.
"""

import os
import sys
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ===================================================================
# Helpers
# ===================================================================

class DummyGenerator(nn.Module):
    """Minimal drifting-generator stub: a = z (identity mapping)."""

    def __init__(self, horizon_steps, action_dim):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        # Need at least one parameter so the module is non-empty
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def predict(self, cond, z, clip_actions=True):
        # Identity mapping: action = z (clipped to [-1, 1])
        if clip_actions:
            return z.clamp(-1, 1)
        return z


# ===================================================================
# 1. LatentPolicyHeadLowdim initialization
# ===================================================================

class TestLatentPolicyHeadLowdimInit:
    """Verify that initial latent policy outputs z ~ N(0, I)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from model.drifting.ft_latent_ppo.latent_policy_head import (
            LatentPolicyHeadLowdim,
        )

        self.obs_dim = 11
        self.cond_steps = 2
        self.horizon_steps = 4
        self.action_dim = 3
        self.head = LatentPolicyHeadLowdim(
            cond_dim=self.obs_dim * self.cond_steps,
            horizon_steps=self.horizon_steps,
            action_dim=self.action_dim,
        )
        self.head.eval()

    def test_initial_mu_is_zero(self):
        B = 8
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        mu, log_std = self.head(cond)
        assert mu.shape == (B, self.horizon_steps, self.action_dim)
        assert torch.allclose(mu, torch.zeros_like(mu), atol=1e-6)

    def test_initial_log_std_is_zero(self):
        B = 8
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        mu, log_std = self.head(cond)
        assert log_std.shape == (B, self.horizon_steps, self.action_dim)
        assert torch.allclose(log_std, torch.zeros_like(log_std), atol=1e-6)

    def test_sampled_z_matches_standard_normal(self):
        """Statistical test: samples from initial policy should ≈ N(0,1)."""
        B = 1000
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        with torch.no_grad():
            mu, log_std = self.head(cond)
        std = log_std.exp()
        z = mu + std * torch.randn_like(std)
        # Check mean ≈ 0 and std ≈ 1
        assert abs(z.mean().item()) < 0.15, f"Mean too far from 0: {z.mean().item()}"
        assert abs(z.std().item() - 1.0) < 0.15, f"Std too far from 1: {z.std().item()}"


# ===================================================================
# 2. LatentPPODrifting integration
# ===================================================================

class TestLatentPPODriftingIntegration:
    """Test the full LatentPPODrifting module with a dummy generator."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from model.drifting.ft_latent_ppo.latent_policy_head import (
            LatentPolicyHeadLowdim,
        )
        from model.drifting.ft_latent_ppo.latent_ppo_drifting import (
            LatentPPODrifting,
        )
        from model.common.critic import CriticObs

        self.obs_dim = 11
        self.cond_steps = 1
        self.horizon_steps = 4
        self.action_dim = 3
        self.act_steps = 2
        self.device = "cpu"

        generator = DummyGenerator(self.horizon_steps, self.action_dim)
        latent_policy = LatentPolicyHeadLowdim(
            cond_dim=self.obs_dim * self.cond_steps,
            horizon_steps=self.horizon_steps,
            action_dim=self.action_dim,
        )
        critic = CriticObs(
            cond_dim=self.obs_dim * self.cond_steps,
            mlp_dims=[64, 64],
        )

        self.model = LatentPPODrifting(
            device=self.device,
            latent_policy=latent_policy,
            critic=critic,
            act_dim=self.action_dim,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            act_min=-1,
            act_max=1,
            obs_dim=self.obs_dim,
            cond_steps=self.cond_steps,
            generator=generator,
        )

    def test_generator_is_frozen(self):
        for p in self.model.generator.parameters():
            assert not p.requires_grad, "Generator params should be frozen"

    def test_latent_policy_is_trainable(self):
        trainable = [p for p in self.model.latent_policy.parameters() if p.requires_grad]
        assert len(trainable) > 0, "Latent policy should have trainable params"

    def test_critic_is_trainable(self):
        trainable = [p for p in self.model.critic.parameters() if p.requires_grad]
        assert len(trainable) > 0, "Critic should have trainable params"

    def test_forward_deterministic(self):
        B = 4
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        action = self.model(cond, deterministic=True)
        assert action.shape == (B, self.horizon_steps, self.action_dim)

    def test_forward_stochastic(self):
        B = 4
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        action = self.model(cond, deterministic=False)
        assert action.shape == (B, self.horizon_steps, self.action_dim)

    def test_sample_with_latent_shapes(self):
        B = 4
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        action, z, logprob, value = self.model.sample_with_latent(cond)
        assert action.shape == (B, self.horizon_steps, self.action_dim)
        assert z.shape == (B, self.horizon_steps, self.action_dim)
        assert logprob.shape == (B,)
        assert value.shape == (B,)

    def test_deterministic_gives_same_action(self):
        """Deterministic mode should give identical outputs."""
        torch.manual_seed(42)
        B = 4
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        a1 = self.model(cond, deterministic=True)
        a2 = self.model(cond, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_frozen_generator_output_consistency(self):
        """Under the same z, generator output must be identical."""
        B = 4
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        z = torch.randn(B, self.horizon_steps, self.action_dim)
        a1 = self.model.generator.predict(cond, z, clip_actions=True)
        a2 = self.model.generator.predict(cond, z, clip_actions=True)
        assert torch.allclose(a1, a2)


# ===================================================================
# 3. Logprob computation consistency
# ===================================================================

class TestLogprobConsistency:
    """Verify get_logprobs_entropy is consistent with stored z logprobs."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from model.drifting.ft_latent_ppo.latent_policy_head import (
            LatentPolicyHeadLowdim,
        )
        from model.drifting.ft_latent_ppo.latent_ppo_drifting import (
            LatentPPODrifting,
        )
        from model.common.critic import CriticObs

        self.obs_dim = 11
        self.cond_steps = 1
        self.horizon_steps = 4
        self.action_dim = 3
        self.device = "cpu"

        generator = DummyGenerator(self.horizon_steps, self.action_dim)
        latent_policy = LatentPolicyHeadLowdim(
            cond_dim=self.obs_dim * self.cond_steps,
            horizon_steps=self.horizon_steps,
            action_dim=self.action_dim,
        )
        critic = CriticObs(
            cond_dim=self.obs_dim * self.cond_steps,
            mlp_dims=[64, 64],
        )

        self.model = LatentPPODrifting(
            device=self.device,
            latent_policy=latent_policy,
            critic=critic,
            act_dim=self.action_dim,
            horizon_steps=self.horizon_steps,
            act_steps=self.horizon_steps,
            act_min=-1,
            act_max=1,
            obs_dim=self.obs_dim,
            cond_steps=self.cond_steps,
            generator=generator,
        )

    def test_logprob_recomputation_matches(self):
        """Logprobs from sample_with_latent match re-computed logprobs."""
        B = 8
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        _, z, logprob_orig, _ = self.model.sample_with_latent(cond)

        # Re-compute
        logprob_recomp, _, _ = self.model.get_logprobs_entropy(cond, z)
        assert torch.allclose(logprob_orig, logprob_recomp, atol=1e-5)

    def test_logprob_is_negative_for_unit_gaussian(self):
        """Log-prob of z ~ N(0,1) under N(0,1) should be negative and finite."""
        B = 8
        cond = {"state": torch.randn(B, self.cond_steps, self.obs_dim)}
        z = torch.randn(B, self.horizon_steps, self.action_dim)
        logprob, _, _ = self.model.get_logprobs_entropy(cond, z)
        assert torch.all(logprob < 0), "Logprobs should be negative"
        assert torch.all(torch.isfinite(logprob)), "Logprobs should be finite"

    def test_logprob_manual_check(self):
        """Cross-validate logprob against PyTorch Normal distribution."""
        B = 4
        cond = {"state": torch.zeros(B, self.cond_steps, self.obs_dim)}
        z = torch.ones(B, self.horizon_steps, self.action_dim) * 0.5

        mu, log_std = self.model.latent_policy(cond)
        std = log_std.exp()

        # Manual using torch.distributions
        dist = torch.distributions.Normal(mu, std)
        expected = dist.log_prob(z).reshape(B, -1).sum(dim=-1)

        # Our implementation
        actual = LatentPPODrifting._compute_logprob(mu, log_std, z)

        assert torch.allclose(actual, expected, atol=1e-5)


# ===================================================================
# 4. PPODriftingLatentBuffer GAE
# ===================================================================

class TestPPODriftingLatentBuffer:
    """Validate GAE computation in the latent PPO buffer."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from agent.finetune.drifting.ppo_drifting_latent_buffer import (
            PPODriftingLatentBuffer,
        )

        self.n_steps = 5
        self.n_envs = 2
        self.cond_steps = 1
        self.obs_dim = 4
        self.horizon_steps = 2
        self.action_dim = 2
        self.gamma = 0.99
        self.gae_lambda = 0.95

        self.buf = PPODriftingLatentBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            cond_steps=self.cond_steps,
            obs_dim=self.obs_dim,
            horizon_steps=self.horizon_steps,
            action_dim=self.action_dim,
            device="cpu",
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def test_add_and_gae_shapes(self):
        for t in range(self.n_steps):
            obs = {"state": np.random.randn(self.n_envs, self.cond_steps, self.obs_dim)}
            z = np.random.randn(self.n_envs, self.horizon_steps, self.action_dim)
            action = np.random.randn(self.n_envs, self.horizon_steps, self.action_dim)
            reward = np.random.randn(self.n_envs)
            done = np.zeros(self.n_envs)
            value = np.random.randn(self.n_envs)
            logprob = np.random.randn(self.n_envs)
            self.buf.add(obs, z, action, reward, done, value, logprob)

        last_values = np.zeros(self.n_envs)
        last_dones = np.zeros(self.n_envs)
        advantages, returns = self.buf.compute_gae(last_values, last_dones)

        assert advantages.shape == (self.n_steps, self.n_envs)
        assert returns.shape == (self.n_steps, self.n_envs)

    def test_gae_returns_match_advantages_plus_values(self):
        for t in range(self.n_steps):
            obs = {"state": np.random.randn(self.n_envs, self.cond_steps, self.obs_dim)}
            z = np.random.randn(self.n_envs, self.horizon_steps, self.action_dim)
            action = z.copy()
            reward = np.ones(self.n_envs) * 1.0
            done = np.zeros(self.n_envs)
            value = np.ones(self.n_envs) * 0.5
            logprob = np.zeros(self.n_envs)
            self.buf.add(obs, z, action, reward, done, value, logprob)

        last_values = np.ones(self.n_envs) * 0.5
        last_dones = np.zeros(self.n_envs)
        advantages, returns = self.buf.compute_gae(last_values, last_dones)

        np.testing.assert_allclose(returns, advantages + self.buf.values, atol=1e-7)

    def test_get_tensors_shapes(self):
        for t in range(self.n_steps):
            obs = {"state": np.random.randn(self.n_envs, self.cond_steps, self.obs_dim)}
            z = np.random.randn(self.n_envs, self.horizon_steps, self.action_dim)
            action = z.copy()
            reward = np.ones(self.n_envs)
            done = np.zeros(self.n_envs)
            value = np.ones(self.n_envs)
            logprob = np.zeros(self.n_envs)
            self.buf.add(obs, z, action, reward, done, value, logprob)

        advantages, returns = self.buf.compute_gae(
            np.zeros(self.n_envs), np.zeros(self.n_envs)
        )
        tensors = self.buf.get_tensors(advantages, returns)
        total = self.n_steps * self.n_envs

        assert tensors["obs_state"].shape == (total, self.cond_steps, self.obs_dim)
        assert tensors["z"].shape == (total, self.horizon_steps, self.action_dim)
        assert tensors["actions"].shape == (total, self.horizon_steps, self.action_dim)
        assert tensors["returns"].shape == (total,)
        assert tensors["values"].shape == (total,)
        assert tensors["advantages"].shape == (total,)
        assert tensors["logprobs"].shape == (total,)


# ===================================================================
# 5. LatentPPODrifting loss computation
# ===================================================================

class TestLatentPPODriftingLoss:
    """Verify PPO loss computation shapes and basic properties."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from model.drifting.ft_latent_ppo.latent_policy_head import (
            LatentPolicyHeadLowdim,
        )
        from model.drifting.ft_latent_ppo.latent_ppo_drifting import (
            LatentPPODrifting,
        )
        from model.common.critic import CriticObs

        self.obs_dim = 11
        self.cond_steps = 1
        self.horizon_steps = 4
        self.action_dim = 3
        self.B = 8

        generator = DummyGenerator(self.horizon_steps, self.action_dim)
        latent_policy = LatentPolicyHeadLowdim(
            cond_dim=self.obs_dim * self.cond_steps,
            horizon_steps=self.horizon_steps,
            action_dim=self.action_dim,
        )
        critic = CriticObs(
            cond_dim=self.obs_dim * self.cond_steps,
            mlp_dims=[64, 64],
        )
        self.model = LatentPPODrifting(
            device="cpu",
            latent_policy=latent_policy,
            critic=critic,
            act_dim=self.action_dim,
            horizon_steps=self.horizon_steps,
            act_steps=self.horizon_steps,
            act_min=-1,
            act_max=1,
            obs_dim=self.obs_dim,
            cond_steps=self.cond_steps,
            generator=generator,
        )

    def test_loss_returns_correct_types(self):
        obs = {"state": torch.randn(self.B, self.cond_steps, self.obs_dim)}
        z = torch.randn(self.B, self.horizon_steps, self.action_dim)
        returns = torch.randn(self.B)
        old_values = torch.randn(self.B)
        advantages = torch.randn(self.B)
        old_logprobs = torch.randn(self.B) - 10  # typical negative values

        pg_loss, ent_loss, v_loss, clipfrac, approx_kl, ratio, std = self.model.loss(
            obs, z, returns, old_values, advantages, old_logprobs,
        )

        assert isinstance(pg_loss, torch.Tensor)
        assert isinstance(ent_loss, torch.Tensor)
        assert isinstance(v_loss, torch.Tensor)
        assert isinstance(clipfrac, float)
        assert isinstance(approx_kl, float)
        assert isinstance(ratio, float)

    def test_loss_backward_updates_latent_policy_only(self):
        """Loss backward should only affect latent_policy and critic, not generator."""
        obs = {"state": torch.randn(self.B, self.cond_steps, self.obs_dim)}
        z = torch.randn(self.B, self.horizon_steps, self.action_dim)
        returns = torch.randn(self.B)
        old_values = torch.randn(self.B)
        advantages = torch.randn(self.B)
        old_logprobs = torch.randn(self.B) - 10

        pg_loss, ent_loss, v_loss, _, _, _, _ = self.model.loss(
            obs, z, returns, old_values, advantages, old_logprobs,
        )
        total_loss = pg_loss + ent_loss * 0.01 + v_loss * 0.5
        total_loss.backward()

        # Latent policy should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.latent_policy.parameters()
            if p.requires_grad
        )
        assert has_grad, "Latent policy should receive gradients"

        # Generator should have no gradients
        for p in self.model.generator.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                "Generator should NOT receive gradients"
            )


# Need to import LatentPPODrifting at module level for test_logprob_manual_check
from model.drifting.ft_latent_ppo.latent_ppo_drifting import LatentPPODrifting
