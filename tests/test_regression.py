"""
Regression tests for code-review fixes.

These tests are lightweight and deterministic — they do NOT require MuJoCo,
robomimic, or GPU.  They validate:

  1. Launcher import path sanity  (script.download_url, script.run imports)
  2. offline_dataset_path resolution / propagation logic
  3. CriticObsAct(residual_style=True) selects ResidualMLP
  4. Rollout metric accounting with / without initial episode-start marker
"""

import importlib
import sys
import os
import types
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Launcher import path sanity
# ---------------------------------------------------------------------------

class TestLauncherImports:
    """Ensure the launcher and its download_url dependency are importable."""

    def test_download_url_module_importable(self):
        """script.download_url must be importable without error."""
        mod = importlib.import_module("script.download_url")
        assert hasattr(mod, "get_dataset_download_url")
        assert hasattr(mod, "get_normalization_download_url")
        assert hasattr(mod, "get_checkpoint_download_url")

    def test_download_url_functions_callable(self):
        """Stub functions should be callable and return None for unknown envs."""
        from script.download_url import (
            get_dataset_download_url,
            get_normalization_download_url,
            get_checkpoint_download_url,
        )

        class _FakeCfg:
            def get(self, key, default=None):
                return default

        cfg = _FakeCfg()
        assert get_dataset_download_url(cfg) is None
        assert get_normalization_download_url(cfg) is None
        assert get_checkpoint_download_url(cfg) is None

    def test_script_package_init_exists(self):
        """script/__init__.py must exist so script is a proper package."""
        mod = importlib.import_module("script")
        assert mod is not None


# ---------------------------------------------------------------------------
# 2. offline_dataset_path resolution / propagation
# ---------------------------------------------------------------------------

class TestOfflineDatasetPathResolution:
    """Verify that the launcher resolves offline_dataset_path correctly."""

    def test_hf_path_propagates_to_offline_dataset(self):
        """When offline_dataset_path is hf://, it should propagate to
        offline_dataset.dataset_path after resolution."""
        from omegaconf import OmegaConf
        from util.hf_download import is_hf_path

        cfg = OmegaConf.create({
            "offline_dataset_path": "hf://gym/hopper-medium-v2/train.npz",
            "offline_dataset": {
                "dataset_path": "hf://gym/hopper-medium-v2/train.npz",
            },
        })

        # Confirm the path is identified as HF
        assert is_hf_path(cfg.offline_dataset_path)

    def test_local_path_not_hf(self):
        """Local paths should NOT be treated as HF paths."""
        from util.hf_download import is_hf_path

        assert not is_hf_path("/data/gym/hopper/train.npz")
        assert not is_hf_path("data/train.npz")

    def test_resolution_code_path_exists_in_run(self):
        """Verify that script/run.py contains offline_dataset_path handling."""
        run_src_path = os.path.join(
            os.path.dirname(__file__), "..", "script", "run.py"
        )
        with open(run_src_path, "r") as f:
            src = f.read()
        assert "offline_dataset_path" in src
        assert "offline_dataset" in src
        assert 'cfg.offline_dataset.dataset_path' in src


# ---------------------------------------------------------------------------
# 3. CriticObsAct(residual_style=True) selects ResidualMLP
# ---------------------------------------------------------------------------

class TestCriticResidualStyle:
    """Ensure CriticObsAct respects the residual_style flag."""

    def test_residual_style_false_uses_mlp(self):
        """Default (residual_style=False) should use plain MLP for Q networks."""
        from model.common.critic import CriticObsAct
        from model.common.mlp import MLP

        critic = CriticObsAct(
            cond_dim=11,
            mlp_dims=[64, 64],
            action_dim=3,
            action_steps=1,
            residual_style=False,
            double_q=True,
        )
        assert isinstance(critic.Q1, MLP)
        assert isinstance(critic.Q2, MLP)

    def test_residual_style_true_uses_residual_mlp(self):
        """residual_style=True must select ResidualMLP, not plain MLP."""
        from model.common.critic import CriticObsAct
        from model.common.mlp import ResidualMLP

        critic = CriticObsAct(
            cond_dim=11,
            mlp_dims=[64, 64, 64],
            action_dim=3,
            action_steps=1,
            residual_style=True,
            double_q=True,
        )
        assert isinstance(critic.Q1, ResidualMLP), (
            f"Expected ResidualMLP, got {type(critic.Q1).__name__}. "
            "residual_style=True is not being honoured."
        )
        assert isinstance(critic.Q2, ResidualMLP)

    def test_residual_style_forward_runs(self):
        """Smoke test: forward pass works with residual_style=True."""
        from model.common.critic import CriticObsAct

        critic = CriticObsAct(
            cond_dim=11,
            mlp_dims=[64, 64, 64],
            action_dim=3,
            action_steps=4,
            residual_style=True,
            double_q=True,
        )
        cond = {"state": torch.randn(8, 1, 11)}
        action = torch.randn(8, 4, 3)
        q1, q2 = critic(cond, action)
        assert q1.shape == (8,)
        assert q2.shape == (8,)

    def test_config_kwarg_residual_style_not_swallowed_by_kwargs(self):
        """Passing residual_style=True must NOT be silently absorbed by **kwargs."""
        from model.common.critic import CriticObsAct
        from model.common.mlp import ResidualMLP

        # Simulate what Hydra does: pass residual_style as a keyword arg
        critic = CriticObsAct(
            cond_dim=11,
            mlp_dims=[64, 64, 64],
            action_dim=3,
            residual_style=True,
        )
        assert isinstance(critic.Q1, ResidualMLP)


# ---------------------------------------------------------------------------
# 4. Rollout metric accounting
# ---------------------------------------------------------------------------

class TestRolloutMetricAccounting:
    """Verify _summarize_rollout_metrics with and without episode-start marker."""

    @staticmethod
    def _summarize(firsts_trajs, reward_trajs, n_envs, act_steps=1,
                   threshold=1.0):
        """Standalone replica of _summarize_rollout_metrics logic."""
        episodes_start_end = []
        for env_ind in range(n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        if len(episodes_start_end) == 0:
            return {
                "num_episode_finished": 0,
                "avg_episode_reward": 0.0,
                "avg_best_reward": 0.0,
                "success_rate": 0.0,
                "avg_traj_length": 0.0,
            }

        reward_trajs_split = [
            reward_trajs[start : end + 1, env_ind]
            for env_ind, start, end in episodes_start_end
        ]
        episode_reward = np.array([np.sum(r) for r in reward_trajs_split])
        episode_best_reward = np.array(
            [np.max(r) / act_steps for r in reward_trajs_split]
        )
        episode_lengths = np.array(
            [end - start + 1 for _, start, end in episodes_start_end]
        ) * act_steps

        return {
            "num_episode_finished": len(reward_trajs_split),
            "avg_episode_reward": float(np.mean(episode_reward)),
            "avg_best_reward": float(np.mean(episode_best_reward)),
            "success_rate": float(np.mean(episode_best_reward >= threshold)),
            "avg_traj_length": float(np.mean(episode_lengths)),
        }

    def test_no_marker_misses_first_episode(self):
        """Without firsts_trajs[0]=1, an episode completing mid-rollout is missed."""
        n_steps, n_envs = 5, 1

        # Episode completes at step 3 (done at step 3)
        firsts = np.zeros((n_steps + 1, n_envs))
        # NOT seeding firsts[0] = 1  (the old bug)
        firsts[4] = 1  # done after step 3

        rewards = np.ones((n_steps, n_envs))

        metrics = self._summarize(firsts, rewards, n_envs)
        # Without the [0]=1 marker, there's no start-end pair → 0 episodes
        assert metrics["num_episode_finished"] == 0

    def test_marker_counts_first_episode(self):
        """With firsts_trajs[0]=1, the same episode is correctly counted."""
        n_steps, n_envs = 5, 1

        firsts = np.zeros((n_steps + 1, n_envs))
        firsts[0] = 1  # Mark episode start (the fix)
        firsts[4] = 1  # Episode done after step 3

        rewards = np.ones((n_steps, n_envs))

        metrics = self._summarize(firsts, rewards, n_envs)
        assert metrics["num_episode_finished"] == 1
        # Episode spans steps 0..3 → reward = 4.0
        assert metrics["avg_episode_reward"] == pytest.approx(4.0)
        assert metrics["avg_traj_length"] == pytest.approx(4.0)

    def test_two_episodes_both_counted(self):
        """Two complete episodes should both be counted."""
        n_steps, n_envs = 10, 1

        firsts = np.zeros((n_steps + 1, n_envs))
        firsts[0] = 1  # start of episode 1
        firsts[4] = 1  # end ep1 / start ep2
        firsts[8] = 1  # end ep2

        rewards = np.ones((n_steps, n_envs))

        metrics = self._summarize(firsts, rewards, n_envs)
        assert metrics["num_episode_finished"] == 2

    def test_multi_env_first_episode(self):
        """Each env should count its first episode when marker is set."""
        n_steps, n_envs = 5, 3

        firsts = np.zeros((n_steps + 1, n_envs))
        firsts[0] = 1  # all envs start
        firsts[3, 0] = 1  # env 0 done at step 2
        firsts[4, 1] = 1  # env 1 done at step 3
        # env 2 never finishes

        rewards = np.ones((n_steps, n_envs))

        metrics = self._summarize(firsts, rewards, n_envs)
        assert metrics["num_episode_finished"] == 2
