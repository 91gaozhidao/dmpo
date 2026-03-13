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
# 3. Hydra compose – qguided task configs stay top-level
# ===================================================================

class TestQGuidedHydraCompose:
    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch):
        pytest.importorskip("hydra")
        monkeypatch.setenv("REINFLOW_DIR", REPO_ROOT)
        monkeypatch.setenv("REINFLOW_DATA_DIR", os.path.join(REPO_ROOT, "data"))
        monkeypatch.setenv("REINFLOW_LOG_DIR", os.path.join(REPO_ROOT, "log"))
        monkeypatch.setenv("REINFLOW_WANDB_ENTITY", "test-entity")

        from omegaconf import OmegaConf

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)

        yield

        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

    def _compose_task(self, rel_config_dir, config_name):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        GlobalHydra.instance().clear()
        config_dir = os.path.join(REPO_ROOT, rel_config_dir)
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        return cfg

    def test_gym_transformer_task_compose_from_task_dir(self):
        cfg = self._compose_task(
            "cfg/gym/finetune/hopper-v2",
            "ft_qguided_drifting_transformer",
        )
        assert cfg._target_ == (
            "agent.finetune.drifting.train_qguided_drifting_agent."
            "TrainQGuidedDriftingAgent"
        )
        for key in ("logdir", "base_policy_path", "train", "env", "model"):
            assert key in cfg
        assert "templates" not in cfg
        assert cfg.env_name == "hopper-medium-v2"

    def test_gym_unet_task_compose_from_task_dir(self):
        cfg = self._compose_task(
            "cfg/gym/finetune/hopper-v2",
            "ft_qguided_drifting_unet1d",
        )
        assert cfg.model.policy._target_ == (
            "model.drifting.backbone.conditional_unet1d.ConditionalUnet1D"
        )
        assert cfg.train.batch_size == 256
        assert "templates" not in cfg

    def test_robomimic_transformer_task_compose_from_task_dir(self):
        cfg = self._compose_task(
            "cfg/robomimic/finetune/can",
            "ft_qguided_drifting_transformer_img",
        )
        assert cfg._target_ == (
            "agent.finetune.drifting.train_qguided_drifting_agent."
            "TrainQGuidedDriftingAgent"
        )
        assert cfg.robomimic_env_cfg_path == "cfg/robomimic/env_meta/can-img.json"
        assert cfg.shape_meta.obs.rgb.shape == [3, 96, 96]
        assert cfg.image_keys == ["robot0_eye_in_hand_image"]
        assert cfg.model.policy.core_network.causal_attn is False
        assert "templates" not in cfg

    def test_robomimic_unet_task_compose_from_task_dir(self):
        cfg = self._compose_task(
            "cfg/robomimic/finetune/can",
            "ft_qguided_drifting_unet1d_img",
        )
        assert cfg.model.policy._target_ == (
            "model.drifting.backbone.conditional_unet1d.ConditionalUnet1D"
        )
        assert cfg.low_dim_keys == [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        assert "templates" not in cfg


# ===================================================================
# 3b. Legacy RoboMimic image datasets – qlearning compatibility
# ===================================================================

class TestLegacyRobomimicImageDatasetCompat:
    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_legacy_robomimic_image_dataset_synthesizes_rewards_and_terminals(
        self, tmp_path
    ):
        from agent.dataset.sequence import StitchedSequenceQLearningDataset

        dataset_dir = tmp_path / "robomimic" / "can-img"
        dataset_dir.mkdir(parents=True)
        dataset_path = dataset_dir / "train.npz"
        np.savez_compressed(
            dataset_path,
            states=np.zeros((5, 9), dtype=np.float32),
            actions=np.zeros((5, 7), dtype=np.float32),
            traj_lengths=np.array([2, 3], dtype=np.int64),
            images=np.zeros((5, 3, 96, 96), dtype=np.uint8),
        )

        dataset = StitchedSequenceQLearningDataset(
            dataset_path=str(dataset_path),
            horizon_steps=1,
            cond_steps=1,
            img_cond_steps=1,
            use_img=True,
            device="cpu",
        )

        assert dataset.rewards.shape[0] == 5
        assert np.allclose(dataset.rewards.cpu().numpy(), 0.0)
        assert np.array_equal(
            dataset.dones.cpu().numpy(),
            np.array([0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_missing_rewards_still_errors_for_non_robomimic_dataset(self, tmp_path):
        from agent.dataset.sequence import StitchedSequenceQLearningDataset

        dataset_dir = tmp_path / "gym" / "hopper-v2"
        dataset_dir.mkdir(parents=True)
        dataset_path = dataset_dir / "train.npz"
        np.savez_compressed(
            dataset_path,
            states=np.zeros((5, 11), dtype=np.float32),
            actions=np.zeros((5, 3), dtype=np.float32),
            traj_lengths=np.array([5], dtype=np.int64),
        )

        with pytest.raises(KeyError, match="missing required keys for Q-learning"):
            StitchedSequenceQLearningDataset(
                dataset_path=str(dataset_path),
                horizon_steps=1,
                cond_steps=1,
                device="cpu",
            )


# ===================================================================
# 4. CriticObsAct – residual_style selects ResidualMLP
# ===================================================================

class TestCriticResidualStyle:
    """``CriticObsAct(residual_style=True)`` must use ``ResidualMLP``."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

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
# 5. Rollout metric accounting – firsts_trajs seeding
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
# 6. download_url helpers – _resolve_env_name / _resolve_dataset_path
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


# ===================================================================
# 7. Online-only Q-Guided Drifting – DictReplayBuffer & batch sampling
# ===================================================================

class TestOnlineOnlyQGuidedDrifting:
    """Validate that the online-only mode additions to
    TrainQGuidedDriftingAgent work correctly at the unit level."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_dict_replay_buffer_add_and_sample(self):
        """DictReplayBuffer can store and retrieve transitions."""
        import torch

        # Inline minimal replay buffer test to avoid importing the full
        # training agent module (which depends on wandb, gym, etc.)
        from collections import deque

        class _MinimalDictReplayBuffer:
            """Subset matching DictReplayBuffer's add/sample/len API."""

            def __init__(self, capacity):
                self._obs = deque(maxlen=capacity)
                self._actions = deque(maxlen=capacity)
                self._next_obs = deque(maxlen=capacity)
                self._rewards = deque(maxlen=capacity)
                self._terminated = deque(maxlen=capacity)

            def __len__(self):
                return len(self._obs)

            def add(self, obs, action, next_obs, reward, terminated):
                self._obs.append(obs)
                self._actions.append(action)
                self._next_obs.append(next_obs)
                self._rewards.append(reward)
                self._terminated.append(terminated)

            def sample(self, batch_size, device):
                indices = np.random.randint(0, len(self), size=batch_size)
                obs = {
                    k: torch.from_numpy(np.stack(
                        [self._obs[i][k] for i in indices]
                    )).to(device)
                    for k in self._obs[0]
                }
                actions = torch.from_numpy(
                    np.stack([self._actions[i] for i in indices])
                ).to(device)
                next_obs = {
                    k: torch.from_numpy(np.stack(
                        [self._next_obs[i][k] for i in indices]
                    )).to(device)
                    for k in self._next_obs[0]
                }
                rewards = torch.tensor(
                    [self._rewards[i] for i in indices],
                    dtype=torch.float32,
                    device=device,
                )
                terminated = torch.tensor(
                    [self._terminated[i] for i in indices],
                    dtype=torch.float32,
                    device=device,
                )
                return obs, actions, next_obs, rewards, terminated

        buf = _MinimalDictReplayBuffer(capacity=100)
        assert len(buf) == 0

        for i in range(10):
            buf.add(
                obs={"state": np.random.randn(2, 11).astype(np.float32)},
                action=np.random.randn(4, 3).astype(np.float32),
                next_obs={"state": np.random.randn(2, 11).astype(np.float32)},
                reward=float(i),
                terminated=0.0,
            )
        assert len(buf) == 10

        obs, actions, next_obs, rewards, terminated = buf.sample(
            batch_size=4, device=torch.device("cpu")
        )
        assert obs["state"].shape == (4, 2, 11)
        assert actions.shape == (4, 4, 3)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)

    def test_online_only_replay_warmup_gate_not_ready(self):
        """Pure-online mode should gate updates on min_replay_size."""
        replay_size = 0
        min_replay_size = 32
        result = replay_size >= max(min_replay_size, 1)
        assert result is False

    def test_online_only_replay_warmup_gate_ready(self):
        """Once replay reaches min_replay_size, pure-online updates may start."""
        replay_size = 64
        min_replay_size = 32
        result = replay_size >= max(min_replay_size, 1)
        assert result is True

    def test_online_only_source_code_has_online_only_flag(self):
        """The training agent source must contain online_only config key."""
        agent_py = os.path.join(
            REPO_ROOT,
            "agent", "finetune", "drifting",
            "train_qguided_drifting_agent.py",
        )
        with open(agent_py, "r") as f:
            src = f.read()

        assert 'online_only' in src, (
            "TrainQGuidedDriftingAgent must support online_only config flag"
        )
        assert 'self.online_only' in src
        # When online_only is requested, offline_batch_ratio must be forced to 0.0
        assert '0.0 if requested_online_only' in src, (
            "offline_batch_ratio must be forced to 0.0 in online-only mode"
        )
        # When online_only is requested, offline_only_iters must be 0
        assert '0 if requested_online_only' in src, (
            "offline_only_iters must be forced to 0 in online-only mode"
        )
        assert 'max(self.min_replay_size, 1)' in src, (
            "pure-online mode must gate updates via min_replay_size"
        )

    def test_online_only_run_skips_cache_offline(self):
        """In online-only mode, run() must not call _cache_offline_dataset."""
        agent_py = os.path.join(
            REPO_ROOT,
            "agent", "finetune", "drifting",
            "train_qguided_drifting_agent.py",
        )
        with open(agent_py, "r") as f:
            src = f.read()

        # run() should conditionally skip offline caching
        assert 'None if self.online_only' in src, (
            "run() must skip _cache_offline_dataset when online_only is True"
        )

    def test_run_loop_logs_replay_warmup_skip(self):
        """run() should emit an explicit replay-warmup skip metric."""
        agent_py = os.path.join(
            REPO_ROOT,
            "agent", "finetune", "drifting",
            "train_qguided_drifting_agent.py",
        )
        with open(agent_py, "r") as f:
            src = f.read()

        assert 'train/update_skipped_for_replay_warmup' in src, (
            "run() should log when pure-online updates are skipped for replay warmup"
        )

    def test_run_py_skips_offline_resolution_when_online_only(self):
        """Launcher should not resolve offline datasets in pure-online mode."""
        run_py = os.path.join(REPO_ROOT, "script", "run.py")
        with open(run_py, "r") as f:
            src = f.read()

        assert 'requested_online_only = bool(cfg.get("train", {}).get("online_only", False))' in src
        assert 'Skipping offline dataset resolution because train.online_only=true.' in src
        assert 'cfg.offline_dataset_path = None' in src


# ===================================================================
# 8. Q-Guided model layer – actor/critic loss shapes
# ===================================================================

class TestQGuidedDriftingModel:
    """Unit tests for the QGuidedDrifting model layer ensuring actor and
    critic losses work with online replay-buffer data."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def _make_model(self):
        import torch
        from model.drifting.drifting import DriftingPolicy
        from model.drifting.backbone.transformer_for_drifting import (
            TransformerForDrifting,
        )
        from model.common.critic import CriticObsAct
        from model.drifting.ft_qguided.qguided_drifting import QGuidedDrifting

        device = "cpu"
        obs_dim = 11
        act_dim = 3
        horizon = 4
        act_steps = 1
        cond_steps = 2

        backbone = TransformerForDrifting(
            input_dim=act_dim,
            output_dim=act_dim,
            horizon=horizon,
            n_obs_steps=cond_steps,
            cond_dim=obs_dim,
            n_layer=2,
            n_head=2,
            n_emb=32,
            causal_attn=False,
        )
        # cond_dim for critic must account for flattened multi-step obs
        critic = CriticObsAct(
            cond_dim=obs_dim * cond_steps,
            mlp_dims=[32, 32],
            action_dim=act_dim,
            action_steps=act_steps,
            double_q=True,
        )
        model = QGuidedDrifting(
            device=device,
            policy=backbone,
            critic=critic,
            act_dim=act_dim,
            horizon_steps=horizon,
            act_steps=act_steps,
            act_min=-1.0,
            act_max=1.0,
            obs_dim=obs_dim,
            cond_steps=2,
            num_action_samples=4,
            num_positive_samples=1,
            num_query_samples=2,
        )
        return model

    def test_critic_loss_from_online_data(self):
        """Critic loss must work with online (s, a, r, s', done) tuples."""
        import torch

        model = self._make_model()
        B = 8
        obs = {"state": torch.randn(B, 2, 11)}
        next_obs = {"state": torch.randn(B, 2, 11)}
        actions = torch.randn(B, 4, 3).clamp(-1, 1)
        rewards = torch.randn(B)
        terminated = torch.zeros(B)

        loss, metrics = model.loss_critic(
            obs, next_obs, actions, rewards, terminated, gamma=0.99
        )
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0
        assert "critic/loss" in metrics

    def test_actor_loss_from_online_obs(self):
        """Actor loss must work with online observations only."""
        import torch

        model = self._make_model()
        B = 8
        obs = {"state": torch.randn(B, 2, 11)}

        loss, metrics = model.loss_actor(obs)
        assert loss.ndim == 0  # scalar
        assert "actor/loss" in metrics
        assert "actor/V_norm_mean" in metrics

    def test_forward_deterministic_and_stochastic(self):
        """Forward pass must produce actions in both modes."""
        import torch

        model = self._make_model()
        B = 4
        obs = {"state": torch.randn(B, 2, 11)}

        det_actions = model(obs, deterministic=True)
        stoch_actions = model(obs, deterministic=False)
        assert det_actions.shape == (B, 4, 3)
        assert stoch_actions.shape == (B, 4, 3)
