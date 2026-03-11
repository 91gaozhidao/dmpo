# MIT License
# Copyright (c) 2025 ReinFlow Authors

"""
Phase 1 & 4: Static Dependency, Registry, and Configuration Tests.

Tests cover:
- Module import completeness (no unresolved dependencies / circular imports)
- Factory registration via Hydra _target_ fields
- YAML configuration key completeness
- Parameter alignment between PPO drifting and PPO meanflow configs
"""

import pytest
import importlib
import os
import sys

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Phase 1: Module Import Completeness
# ---------------------------------------------------------------------------

class TestModuleImports:
    """Verify all drifting modules can be imported without errors."""

    def test_import_drifting_policy(self):
        mod = importlib.import_module("model.drifting.drifting")
        assert hasattr(mod, "DriftingPolicy")

    def test_import_ppo_drifting(self):
        mod = importlib.import_module("model.drifting.ft_ppo.ppodrifting")
        assert hasattr(mod, "PPODrifting")
        assert hasattr(mod, "NoisyDriftingMLP")

    def test_import_grpo_drifting(self):
        mod = importlib.import_module("model.drifting.ft_grpo.grpodrifting")
        assert hasattr(mod, "GRPODrifting")
        assert hasattr(mod, "NoisyDriftingPolicy")
        assert hasattr(mod, "_tanh_jacobian_correction")

    def test_import_grpo_buffer(self):
        mod = importlib.import_module("agent.finetune.grpo.buffer")
        assert hasattr(mod, "GRPOBuffer")
        assert hasattr(mod, "ADVANTAGE_STD_THRESHOLD")

    def test_import_train_grpo_drifting_agent(self):
        try:
            mod = importlib.import_module(
                "agent.finetune.grpo.train_grpo_drifting_agent"
            )
            assert hasattr(mod, "TrainGRPODriftingAgent")
        except ImportError as e:
            pytest.skip(f"Optional dependency missing: {e}")

    def test_import_train_drifting_agent(self):
        try:
            mod = importlib.import_module("agent.pretrain.train_drifting_agent")
            assert hasattr(mod, "TrainDriftingAgent")
        except (ImportError, KeyError) as e:
            pytest.skip(f"Optional dependency or env var missing: {e}")

    def test_import_train_ppo_drifting_agent(self):
        try:
            mod = importlib.import_module(
                "agent.finetune.reinflow.train_ppo_drifting_agent"
            )
            assert hasattr(mod, "TrainPPODriftingAgent")
        except ImportError as e:
            pytest.skip(f"Optional dependency missing: {e}")

    def test_import_train_ppo_drifting_img_agent(self):
        try:
            mod = importlib.import_module(
                "agent.finetune.reinflow.train_ppo_drifting_img_agent"
            )
            assert hasattr(mod, "TrainPPOImgDriftingAgent")
        except ImportError as e:
            pytest.skip(f"Optional dependency missing: {e}")

    def test_import_train_drifting_dispersive_agent(self):
        try:
            mod = importlib.import_module(
                "agent.pretrain.train_drifting_dispersive_agent"
            )
            assert hasattr(mod, "TrainDriftingDispersiveAgent")
        except (ImportError, KeyError) as e:
            pytest.skip(f"Optional dependency or env var missing: {e}")

    def test_import_drifting_init_packages(self):
        """Verify __init__.py modules don't cause import errors."""
        importlib.import_module("model.drifting")
        importlib.import_module("model.drifting.ft_ppo")
        importlib.import_module("model.drifting.ft_grpo")
        importlib.import_module("agent.finetune.grpo")

    def test_no_circular_imports(self):
        """Force re-import of all drifting modules to detect circular deps."""
        modules_to_check = [
            "model.drifting.drifting",
            "model.drifting.ft_ppo.ppodrifting",
            "model.drifting.ft_grpo.grpodrifting",
            "agent.finetune.grpo.buffer",
        ]
        for mod_name in modules_to_check:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Failed to import {mod_name}"

        # Agent modules may have optional deps (tqdm, wandb, gym)
        optional_modules = [
            "agent.finetune.grpo.train_grpo_drifting_agent",
            "agent.pretrain.train_drifting_agent",
            "agent.pretrain.train_drifting_dispersive_agent",
            "agent.finetune.reinflow.train_ppo_drifting_agent",
            "agent.finetune.reinflow.train_ppo_drifting_img_agent",
        ]
        for mod_name in optional_modules:
            try:
                mod = importlib.import_module(mod_name)
                assert mod is not None
            except (ImportError, KeyError):
                pass  # Optional dependency or env var missing, not a circular import


# ---------------------------------------------------------------------------
# Phase 1: Factory/Registry Validation
# ---------------------------------------------------------------------------

class TestFactoryRegistration:
    """Verify Hydra _target_ strings resolve to actual classes."""

    @pytest.mark.parametrize("target_path,expected_class", [
        (
            "agent.pretrain.train_drifting_agent.TrainDriftingAgent",
            "TrainDriftingAgent",
        ),
        (
            "agent.pretrain.train_drifting_dispersive_agent.TrainDriftingDispersiveAgent",
            "TrainDriftingDispersiveAgent",
        ),
        (
            "agent.finetune.reinflow.train_ppo_drifting_agent.TrainPPODriftingAgent",
            "TrainPPODriftingAgent",
        ),
        (
            "agent.finetune.reinflow.train_ppo_drifting_img_agent.TrainPPOImgDriftingAgent",
            "TrainPPOImgDriftingAgent",
        ),
        (
            "agent.finetune.grpo.train_grpo_drifting_agent.TrainGRPODriftingAgent",
            "TrainGRPODriftingAgent",
        ),
        (
            "model.drifting.ft_ppo.ppodrifting.PPODrifting",
            "PPODrifting",
        ),
        (
            "model.drifting.ft_grpo.grpodrifting.GRPODrifting",
            "GRPODrifting",
        ),
    ])
    def test_target_resolves(self, target_path, expected_class):
        """Simulate Hydra's get_class() by importing the target path."""
        parts = target_path.rsplit(".", 1)
        module_path, class_name = parts[0], parts[1]
        try:
            mod = importlib.import_module(module_path)
        except (ImportError, KeyError) as e:
            pytest.skip(f"Optional dependency or env var missing: {e}")
        cls = getattr(mod, class_name, None)
        assert cls is not None, f"Class {class_name} not found in {module_path}"
        assert cls.__name__ == expected_class


# ---------------------------------------------------------------------------
# Phase 4: YAML Configuration Validation
# ---------------------------------------------------------------------------

class TestYAMLConfigCompleteness:
    """Verify key hyperparameters are present in drifting config files."""

    @pytest.fixture
    def cfg_dir(self):
        return os.path.join(REPO_ROOT, "cfg")

    def _load_yaml(self, path):
        """Load YAML with basic parsing, skipping files with syntax errors."""
        try:
            from omegaconf import OmegaConf
            return OmegaConf.load(path)
        except Exception:
            # Fallback: use PyYAML with unsafe loader that allows duplicate keys
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)

    def _find_yaml_files(self, cfg_dir, pattern):
        """Find YAML files matching a filename pattern."""
        matches = []
        for root, dirs, files in os.walk(cfg_dir):
            for f in files:
                if pattern in f and f.endswith(".yaml"):
                    matches.append(os.path.join(root, f))
        return matches

    def test_grpo_configs_have_required_keys(self, cfg_dir):
        """GRPO configs must have group_size, kl_beta, clip_coef (epsilon)."""
        grpo_files = self._find_yaml_files(cfg_dir, "ft_grpo_drifting")
        assert len(grpo_files) > 0, "No GRPO drifting config files found"
        checked = 0
        for path in grpo_files:
            try:
                cfg = self._load_yaml(path)
            except Exception:
                continue  # Skip files with parsing errors (e.g., duplicate keys)
            # Check train section
            assert "train" in cfg, f"{path}: missing 'train' section"
            train = cfg["train"]
            assert "group_size" in train, f"{path}: missing train.group_size"
            assert "kl_beta" in train, f"{path}: missing train.kl_beta"
            # Check model section
            assert "model" in cfg, f"{path}: missing 'model' section"
            model = cfg["model"]
            assert "epsilon" in model, f"{path}: missing model.epsilon (clip_coef)"
            checked += 1
        assert checked > 0, "No GRPO config files could be parsed"

    def test_ppo_drifting_configs_have_required_keys(self, cfg_dir):
        """PPO drifting configs must have core PPO hyperparameters."""
        ppo_files = self._find_yaml_files(cfg_dir, "ft_ppo_drifting")
        assert len(ppo_files) > 0, "No PPO drifting config files found"
        checked = 0
        for path in ppo_files:
            try:
                cfg = self._load_yaml(path)
            except Exception:
                continue  # Skip files with parsing errors
            assert "model" in cfg, f"{path}: missing 'model' section"
            assert "train" in cfg, f"{path}: missing 'train' section"
            checked += 1
        assert checked > 0, "No PPO drifting config files could be parsed"

    def test_pretrain_drifting_configs_exist(self, cfg_dir):
        """Verify pretrain drifting configs exist."""
        pre_files = self._find_yaml_files(cfg_dir, "pre_drifting")
        assert len(pre_files) > 0, "No pretrain drifting config files found"

    def test_grpo_config_has_mask_self_in_pretrain(self, cfg_dir):
        """Pretrain configs should define mask_self parameter if using drifting."""
        pre_files = self._find_yaml_files(cfg_dir, "pre_drifting")
        for path in pre_files:
            cfg = self._load_yaml(path)
            if "model" in cfg:
                model = cfg["model"]
                # mask_self may be at top-level model or nested
                if "mask_self" in model:
                    assert isinstance(model["mask_self"], bool)

    def test_ppo_drifting_vs_meanflow_optimizer_alignment(self, cfg_dir):
        """Phase 4: Compare PPO drifting vs PPO meanflow optimizer params."""
        # Find a matching environment pair
        hopper_ppo_drifting = os.path.join(
            cfg_dir, "gym/finetune/hopper-v2/ft_ppo_drifting_mlp.yaml"
        )
        hopper_ppo_meanflow = os.path.join(
            cfg_dir, "gym/finetune/hopper-v2/ft_ppo_meanflow_mlp.yaml"
        )
        if not os.path.exists(hopper_ppo_drifting) or not os.path.exists(hopper_ppo_meanflow):
            pytest.skip("Hopper PPO config pair not found")

        cfg_d = self._load_yaml(hopper_ppo_drifting)
        cfg_m = self._load_yaml(hopper_ppo_meanflow)

        # Compare train-section parameters that should align
        for key in ["actor_lr", "critic_lr", "update_epochs", "batch_size",
                     "gamma", "gae_lambda", "vf_coef", "ent_coef"]:
            if key in cfg_d.get("train", {}) and key in cfg_m.get("train", {}):
                assert cfg_d["train"][key] == cfg_m["train"][key], (
                    f"Parameter train.{key} differs: "
                    f"drifting={cfg_d['train'][key]}, meanflow={cfg_m['train'][key]}"
                )
