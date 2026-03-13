"""Unit tests for the latent-PPO drifting implementation."""

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_lowdim_latent_policy_zero_init_matches_standard_normal():
    from model.drifting.ft_latent_ppo.latent_ppo_drifting import (
        LatentPolicyHeadLowdim,
    )

    head = LatentPolicyHeadLowdim(
        cond_dim=11,
        horizon_steps=4,
        action_dim=3,
        mlp_dims=[32, 32],
        residual_style=False,
        min_std=0.05,
        max_std=2.0,
    )
    cond = {"state": torch.randn(5, 1, 11)}

    dist, mu, std = head.distribution(cond)

    assert mu.shape == (5, 4, 3)
    assert std.shape == (5, 4, 3)
    assert torch.allclose(mu, torch.zeros_like(mu))
    assert torch.allclose(std, torch.ones_like(std))
    sample = dist.rsample()
    assert sample.shape == (5, 4, 3)
    assert dist.log_prob(sample).shape == (5,)


def test_latent_buffer_uses_override_bootstrap_values():
    from agent.finetune.drifting.latent_ppo_buffer import PPODriftingLatentBuffer

    buffer = PPODriftingLatentBuffer(
        n_steps=1,
        n_envs=1,
        horizon_steps=2,
        act_steps=1,
        action_dim=2,
        save_full_observation=False,
        furniture_sparse_reward=False,
        best_reward_threshold_for_success=1,
        reward_scale_running=False,
        gamma=0.9,
        gae_lambda=1.0,
        reward_scale_const=1.0,
        device="cpu",
    )
    buffer.reset(initial_firsts=np.array([1.0], dtype=np.float32))
    buffer.add(
        step=0,
        obs_venv={"state": np.zeros((1, 1, 3), dtype=np.float32)},
        latent_venv=np.zeros((1, 2, 2), dtype=np.float32),
        action_venv=np.zeros((1, 2, 2), dtype=np.float32),
        reward_venv=np.array([1.0], dtype=np.float32),
        terminated_venv=np.array([0.0], dtype=np.float32),
        truncated_venv=np.array([1.0], dtype=np.float32),
        value_venv=np.array([0.0], dtype=np.float32),
        logprob_venv=np.array([0.0], dtype=np.float32),
        next_value_override_venv=np.array([5.0], dtype=np.float32),
    )
    buffer.set_last_values(np.array([0.0], dtype=np.float32))
    buffer.update()

    assert np.isclose(buffer.returns_trajs[0, 0], 1.0 + 0.9 * 5.0)
    assert np.isclose(buffer.advantages_trajs[0, 0], 1.0 + 0.9 * 5.0)


def test_latent_buffer_minibatch_shapes():
    from agent.finetune.drifting.latent_ppo_buffer import PPODriftingLatentBuffer

    buffer = PPODriftingLatentBuffer(
        n_steps=2,
        n_envs=2,
        horizon_steps=3,
        act_steps=1,
        action_dim=2,
        save_full_observation=False,
        furniture_sparse_reward=False,
        best_reward_threshold_for_success=1,
        reward_scale_running=False,
        gamma=0.99,
        gae_lambda=0.95,
        reward_scale_const=1.0,
        device="cpu",
    )
    buffer.reset(initial_firsts=np.ones((2,), dtype=np.float32))
    for step in range(2):
        buffer.add(
            step=step,
            obs_venv={"state": np.ones((2, 1, 4), dtype=np.float32) * step},
            latent_venv=np.zeros((2, 3, 2), dtype=np.float32),
            action_venv=np.zeros((2, 3, 2), dtype=np.float32),
            reward_venv=np.ones((2,), dtype=np.float32),
            terminated_venv=np.zeros((2,), dtype=np.float32),
            truncated_venv=np.zeros((2,), dtype=np.float32),
            value_venv=np.zeros((2,), dtype=np.float32),
            logprob_venv=np.zeros((2,), dtype=np.float32),
        )
    buffer.set_last_values(np.zeros((2,), dtype=np.float32))
    buffer.update()

    minibatch = next(
        buffer.iter_minibatches(
            batch_size=3,
            update_epochs=1,
            device="cpu",
            normalize_advantages=True,
        )
    )
    assert minibatch["obs"]["state"].shape == (3, 1, 4)
    assert minibatch["latents"].shape == (3, 3, 2)
    assert minibatch["actions"].shape == (3, 3, 2)
    assert minibatch["returns"].shape == (3,)
    assert minibatch["advantages"].shape == (3,)
    assert minibatch["logprobs"].shape == (3,)


def test_latent_ppo_model_freezes_generator_and_uses_zero_mean_eval():
    from model.common.critic import CriticObs
    from model.drifting.backbone.conditional_unet1d import ConditionalUnet1D
    from model.drifting.ft_latent_ppo.latent_ppo_drifting import (
        LatentPolicyHeadLowdim,
        LatentPPODrifting,
    )

    backbone = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=3,
        down_dims=[8, 16],
        kernel_size=3,
        n_groups=1,
        cond_predict_scale=False,
    )
    latent_head = LatentPolicyHeadLowdim(
        cond_dim=3,
        horizon_steps=2,
        action_dim=2,
        mlp_dims=[16, 16],
        residual_style=False,
        min_std=0.05,
        max_std=2.0,
    )
    critic = CriticObs(
        cond_dim=3,
        mlp_dims=[16, 16],
        activation_type="Mish",
        residual_style=False,
    )
    model = LatentPPODrifting(
        device="cpu",
        policy=backbone,
        latent_policy=latent_head,
        critic=critic,
        act_dim=2,
        horizon_steps=2,
        act_steps=1,
        act_min=-1,
        act_max=1,
        obs_dim=3,
        cond_steps=1,
        actor_policy_path=None,
        seed=0,
    )

    cond = {"state": torch.randn(4, 1, 3)}
    deterministic_actions = model(cond=cond, deterministic=True)
    zero_latent_actions = model.action_from_latent(
        cond,
        torch.zeros((4, 2, 2), dtype=torch.float32),
    )

    assert all(not param.requires_grad for param in model.actor.parameters())
    assert torch.allclose(deterministic_actions, zero_latent_actions)


def test_image_latent_policy_head_shapes():
    from model.common.vit import VitEncoder, VitEncoderConfig
    from model.drifting.ft_latent_ppo.latent_ppo_drifting import LatentPolicyHeadImage

    backbone = VitEncoder(
        obs_shape=[3, 16, 16],
        cfg=VitEncoderConfig(
            patch_size=8,
            depth=1,
            embed_dim=16,
            num_heads=4,
            embed_style="embed2",
            embed_norm=0,
        ),
        num_channel=3,
        img_h=16,
        img_w=16,
    )
    head = LatentPolicyHeadImage(
        backbone=backbone,
        cond_dim=9,
        horizon_steps=4,
        action_dim=7,
        mlp_dims=[32, 32],
        img_cond_steps=1,
        spatial_emb=16,
        augment=False,
        residual_style=False,
        min_std=0.05,
        max_std=2.0,
    )
    cond = {
        "state": torch.randn(2, 1, 9),
        "rgb": torch.randint(0, 255, (2, 1, 3, 16, 16), dtype=torch.uint8),
    }

    out = head.sample(cond)

    assert out["latents"].shape == (2, 4, 7)
    assert out["latent_mean"].shape == (2, 4, 7)
    assert out["latent_std"].shape == (2, 4, 7)
    assert out["logprob"].shape == (2,)


class TestLatentPPODriftingHydraCompose:
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

    def test_gym_task_compose(self):
        cfg = self._compose_task(
            "cfg/gym/finetune/hopper-v2",
            "ft_latent_ppo_drifting_unet1d",
        )
        assert cfg._target_ == (
            "agent.finetune.drifting.train_latent_ppo_drifting_agent."
            "TrainLatentPPODriftingAgent"
        )
        assert cfg.model._target_ == (
            "model.drifting.ft_latent_ppo.latent_ppo_drifting."
            "LatentPPODrifting"
        )
        assert cfg.train.n_steps == 500
        assert cfg.env_name == "hopper-medium-v2"

    def test_robomimic_task_compose(self):
        cfg = self._compose_task(
            "cfg/robomimic/finetune/can",
            "ft_latent_ppo_drifting_transformer_img",
        )
        assert cfg._target_ == (
            "agent.finetune.drifting.train_latent_ppo_drifting_agent."
            "TrainLatentPPODriftingAgent"
        )
        assert cfg.model.latent_policy._target_ == (
            "model.drifting.ft_latent_ppo.latent_ppo_drifting."
            "LatentPolicyHeadImage"
        )
        assert cfg.shape_meta.obs.rgb.shape == [3, 96, 96]
        assert cfg.image_keys == ["robot0_eye_in_hand_image"]
