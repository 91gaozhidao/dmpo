"""Regression tests for Q-Guided Drifting fixes.

These tests are lightweight and deterministic – they do **not** require
MuJoCo, robomimic, or GPU access.  They validate:

1. ``script/download_url.py`` compatibility with Q-guided fine-tune configs.
2. ``script/run.py`` offline_dataset_path launcher resolution logic.
3. ``CriticObsAct(residual_style=True)`` selecting the residual MLP path.
4. Rollout metric accounting with proper initial episode markers.
"""

import importlib
import os
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so bare imports work (e.g. ``model.…``)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ===================================================================
# 1. download_url.py – env-name / dataset-path resolution
# ===================================================================

class _FakeCfg:
    """Minimal object that quacks like an OmegaConf DictConfig."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)


class TestDownloadUrlCompat:
    """get_dataset_download_url must work with both old-style (pretrain)
    and new-style (Q-guided finetune) config layouts."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from script.download_url import get_dataset_download_url
        self.get_url = get_dataset_download_url

    def test_old_style_env_string(self):
        """Old pretrain configs expose ``cfg.env`` as a plain string."""
        cfg = _FakeCfg(env="hopper-medium-v2")
        url = self.get_url(cfg)
        assert "google.com" in url

    def test_new_style_env_name(self):
        """Q-guided configs expose ``cfg.env_name``."""
        cfg = _FakeCfg(env_name="hopper-medium-v2")
        url = self.get_url(cfg)
        assert "google.com" in url

    def test_offline_dataset_path_for_robomimic_img(self):
        """Q-guided robomimic configs use offline_dataset_path with ``-img``."""
        cfg = _FakeCfg(
            env_name="lift",
            offline_dataset_path="/data/robomimic/lift-img/train.npz",
        )
        url = self.get_url(cfg)
        assert "google.com" in url
        # Should match the img branch
        assert "1H-UncdzHx6wd5NWVzrQyftfGls7KGz1O" in url

    def test_train_dataset_path_still_works(self):
        """Pretrain configs still pass ``train_dataset_path``."""
        cfg = _FakeCfg(
            env="can",
            train_dataset_path="/data/robomimic/can-ph/train.npz",
        )
        url = self.get_url(cfg)
        assert "google.com" in url

    def test_env_dict_with_name(self):
        """Some configs have ``cfg.env`` as a dict with a ``name`` key."""
        env_obj = types.SimpleNamespace(name="kitchen-mixed-v0")
        cfg = _FakeCfg(env=env_obj)
        url = self.get_url(cfg)
        assert "google.com" in url


# ===================================================================
# 2. run.py – offline_dataset_path launcher logic (unit-level)
# ===================================================================

class TestRunLauncherOfflineDatasetPath:
    """Verify that the launcher code in run.py handles
    ``offline_dataset_path`` for both hf:// and local paths."""

    @pytest.fixture(autouse=True)
    def _read_source(self):
        """Parse run.py source to confirm the expected code-paths exist."""
        run_py = os.path.join(REPO_ROOT, "script", "run.py")
        with open(run_py, "r") as f:
            self.source = f.read()

    def test_offline_dataset_path_block_exists(self):
        assert "offline_dataset_path" in self.source, (
            "run.py must contain an offline_dataset_path resolution block"
        )

    def test_hf_branch_for_offline_dataset(self):
        assert 'is_hf_path(cfg.offline_dataset_path)' in self.source

    def test_fallback_download_uses_official_helper(self):
        assert 'get_dataset_download_url(cfg)' in self.source

    def test_propagates_to_offline_dataset_config(self):
        assert 'cfg.offline_dataset.dataset_path' in self.source


# ===================================================================
# 3. CriticObsAct – residual_style selects ResidualMLP
# ===================================================================

class TestCriticResidualStyle:
    """``CriticObsAct(residual_style=True)`` must use ``ResidualMLP``."""

    def test_residual_style_true_uses_residual_mlp(self):
        import torch
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
            "residual_style=True should select ResidualMLP, got "
            f"{type(critic.Q1).__name__}"
        )
        assert isinstance(critic.Q2, ResidualMLP)

    def test_residual_style_false_uses_plain_mlp(self):
        import torch
        from model.common.critic import CriticObsAct
        from model.common.mlp import MLP

        critic = CriticObsAct(
            cond_dim=11,
            mlp_dims=[64, 64, 64],
            action_dim=3,
            action_steps=1,
            residual_style=False,
            double_q=True,
        )
        assert isinstance(critic.Q1, MLP)

    def test_forward_with_residual_critic(self):
        import torch
        from model.common.critic import CriticObsAct

        critic = CriticObsAct(
            cond_dim=11,
            mlp_dims=[64, 64, 64],
            action_dim=3,
            action_steps=4,
            residual_style=True,
            double_q=True,
        )
        cond = {"state": torch.randn(2, 1, 11)}
        action = torch.randn(2, 4, 3)
        q1, q2 = critic(cond, action)
        assert q1.shape == (2,)
        assert q2.shape == (2,)


# ===================================================================
# 4. Rollout metric accounting – firsts_trajs seeding
# ===================================================================

class TestRolloutMetricAccounting:
    """_summarize_rollout_metrics must count completed episodes accurately
    when firsts_trajs[0] is properly seeded."""

    @staticmethod
    def _summarize(firsts, rewards, n_envs, act_steps=1,
                   best_reward_threshold=3.0):
        """Pure-Python replica of _summarize_rollout_metrics logic.

        This avoids importing the full agent module (which needs hydra,
        wandb, etc.) while still testing the exact same algorithm.
        """
        episodes_start_end = []
        for env_ind in range(n_envs):
            env_steps = np.where(firsts[:, env_ind] == 1)[0]
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
            rewards[start: end + 1, env_ind]
            for env_ind, start, end in episodes_start_end
        ]
        episode_reward = np.array(
            [np.sum(r) for r in reward_trajs_split]
        )
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
            "success_rate": float(
                np.mean(episode_best_reward >= best_reward_threshold)
            ),
            "avg_traj_length": float(np.mean(episode_lengths)),
        }

    def test_one_completed_episode_with_seeded_first(self):
        """With firsts_trajs[0]=1 and a done at step 3, one complete
        episode (steps 0..2) should be counted."""
        n_envs = 1
        n_steps = 5
        firsts = np.zeros((n_steps + 1, n_envs))
        firsts[0, 0] = 1        # episode starts at step 0
        firsts[3, 0] = 1        # episode ends (done) at step 2, new one starts at 3
        rewards = np.zeros((n_steps, n_envs))
        rewards[0, 0] = 1.0
        rewards[1, 0] = 2.0
        rewards[2, 0] = 3.0

        metrics = self._summarize(firsts, rewards, n_envs)
        assert metrics["num_episode_finished"] == 1
        assert metrics["avg_episode_reward"] == pytest.approx(6.0)

    def test_no_episode_without_seed(self):
        """If firsts_trajs[0] is NOT seeded (all zeros), no episode
        boundaries are detected → zero episodes counted."""
        n_envs = 1
        n_steps = 5
        firsts = np.zeros((n_steps + 1, n_envs))
        # No seed at [0]!  Only done at step 3.
        firsts[3, 0] = 1
        rewards = np.ones((n_steps, n_envs))

        metrics = self._summarize(firsts, rewards, n_envs)
        assert metrics["num_episode_finished"] == 0

    def test_multiple_envs_partial_episodes(self):
        """Two envs: env0 has one complete episode, env1 has none."""
        n_envs = 2
        n_steps = 5
        firsts = np.zeros((n_steps + 1, n_envs))
        firsts[0, :] = 1        # both start fresh
        firsts[4, 0] = 1        # env0 done at step 3
        # env1 never finishes
        rewards = np.ones((n_steps, n_envs))

        metrics = self._summarize(firsts, rewards, n_envs)
        assert metrics["num_episode_finished"] == 1

    def test_firsts_seeding_code_in_collect_rollout(self):
        """Verify that _collect_rollout in the source code properly
        seeds firsts_trajs[0] based on self.done_venv."""
        agent_py = os.path.join(
            REPO_ROOT,
            "agent", "finetune", "drifting",
            "train_qguided_drifting_agent.py",
        )
        with open(agent_py, "r") as f:
            src = f.read()

        # Must seed firsts on first call
        assert "firsts_trajs[0] = 1" in src, (
            "_collect_rollout must seed firsts_trajs[0]=1 on first rollout"
        )
        # Must carry over done_venv on subsequent calls
        assert "self.done_venv" in src, (
            "_collect_rollout must store/use self.done_venv for subsequent "
            "rollout episode accounting"
        )
        # Must reset done_venv when env is reset
        assert "self.done_venv = None" in src, (
            "run() must reset self.done_venv=None after reset_env_all()"
        )


# ===================================================================
# 5. download_url helpers – _resolve_env_name / _resolve_dataset_path
# ===================================================================

class TestResolveHelpers:
    def test_resolve_env_name_from_env_name(self):
        from script.download_url import _resolve_env_name
        cfg = _FakeCfg(env_name="ant-medium-expert-v2")
        assert _resolve_env_name(cfg) == "ant-medium-expert-v2"

    def test_resolve_env_name_from_env_string(self):
        from script.download_url import _resolve_env_name
        cfg = _FakeCfg(env="walker2d-medium-v2")
        assert _resolve_env_name(cfg) == "walker2d-medium-v2"

    def test_resolve_env_name_from_env_dict(self):
        from script.download_url import _resolve_env_name
        env_obj = types.SimpleNamespace(name="lift")
        cfg = _FakeCfg(env=env_obj)
        assert _resolve_env_name(cfg) == "lift"

    def test_resolve_dataset_path_train(self):
        from script.download_url import _resolve_dataset_path
        cfg = _FakeCfg(train_dataset_path="/a/b/train.npz")
        assert _resolve_dataset_path(cfg) == "/a/b/train.npz"

    def test_resolve_dataset_path_offline(self):
        from script.download_url import _resolve_dataset_path
        cfg = _FakeCfg(offline_dataset_path="/c/d/train.npz")
        assert _resolve_dataset_path(cfg) == "/c/d/train.npz"

    def test_resolve_dataset_path_prefers_train(self):
        from script.download_url import _resolve_dataset_path
        cfg = _FakeCfg(
            train_dataset_path="/a/train.npz",
            offline_dataset_path="/b/train.npz",
        )
        # train_dataset_path is checked first
        assert _resolve_dataset_path(cfg) == "/a/train.npz"

    def test_resolve_dataset_path_empty(self):
        from script.download_url import _resolve_dataset_path
        cfg = _FakeCfg()
        assert _resolve_dataset_path(cfg) == ""
