"""
Minimal robomimic rendering smoke test.

This script isolates robosuite / EGL / OSMesa environment issues without
booting a full training job. It mirrors the repository's robomimic env
creation path closely enough to reproduce most environment-side failures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from gym import spaces
import numpy as np

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


TASK_DEFAULTS = {
    "can": {
        "env_meta": "cfg/robomimic/env_meta/can-img.json",
        "low_dim_keys": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        "image_keys": ["robot0_eye_in_hand_image"],
    },
    "lift": {
        "env_meta": "cfg/robomimic/env_meta/lift-img.json",
        "low_dim_keys": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        "image_keys": ["robot0_eye_in_hand_image"],
    },
    "square": {
        "env_meta": "cfg/robomimic/env_meta/square-img.json",
        "low_dim_keys": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        "image_keys": ["agentview_image"],
    },
    "transport": {
        "env_meta": "cfg/robomimic/env_meta/transport-img.json",
        "low_dim_keys": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot1_eef_pos",
            "robot1_eef_quat",
            "robot1_gripper_qpos",
        ],
        "image_keys": ["shouldercamera0_image", "shouldercamera1_image"],
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test robomimic rendering.")
    parser.add_argument(
        "--task",
        choices=sorted(TASK_DEFAULTS),
        default="can",
        help="Named robomimic task to test.",
    )
    parser.add_argument(
        "--env-meta",
        type=Path,
        default=None,
        help="Optional path to an env_meta json. Overrides --task default file.",
    )
    parser.add_argument(
        "--backend",
        choices=["egl", "osmesa"],
        default=os.environ.get("MUJOCO_GL", "egl"),
        help="Rendering backend to test.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id passed into the robomimic env metadata.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of random-action steps to run after reset.",
    )
    parser.add_argument(
        "--resets",
        type=int,
        default=2,
        help="Number of env resets to run.",
    )
    return parser.parse_args()


def _resolve_env_meta(args: argparse.Namespace) -> tuple[Path, dict, dict]:
    task_defaults = TASK_DEFAULTS[args.task]
    env_meta_path = args.env_meta or (_repo_root() / task_defaults["env_meta"])
    with env_meta_path.open("r", encoding="utf-8") as handle:
        env_meta = json.load(handle)
    env_meta["reward_shaping"] = False
    env_meta["env_kwargs"]["render_gpu_device_id"] = args.gpu_id
    return env_meta_path, env_meta, task_defaults


def _init_obs_utils(task_defaults: dict) -> None:
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {
            "low_dim": task_defaults["low_dim_keys"],
            "rgb": task_defaults["image_keys"],
        }
    )


def _print_obs_summary(obs: dict) -> None:
    print("Observation keys:")
    for key, value in obs.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"  - {key}: shape={shape}, dtype={dtype}")


def _resolve_action_space(env) -> spaces.Box:
    if hasattr(env, "action_spec"):
        spec = env.action_spec() if callable(env.action_spec) else env.action_spec
        low, high = spec
        return spaces.Box(low=low, high=high)

    if hasattr(env, "action_dimension"):
        action_dim = (
            env.action_dimension()
            if callable(env.action_dimension)
            else env.action_dimension
        )
        low = -np.ones(action_dim, dtype=np.float32)
        high = np.ones(action_dim, dtype=np.float32)
        return spaces.Box(low=low, high=high)

    if hasattr(env, "env") and hasattr(env.env, "action_spec"):
        spec = env.env.action_spec() if callable(env.env.action_spec) else env.env.action_spec
        low, high = spec
        return spaces.Box(low=low, high=high)

    raise AttributeError(
        f"Could not infer action space from env type {type(env).__name__}."
    )


def main() -> int:
    args = _parse_args()
    env_meta_path, env_meta, task_defaults = _resolve_env_meta(args)
    os.environ["MUJOCO_GL"] = args.backend

    print(f"Testing task={args.task}")
    print(f"Using env_meta={env_meta_path}")
    print(f"Using backend={args.backend}")
    print(f"Using render_gpu_device_id={args.gpu_id}")

    _init_obs_utils(task_defaults)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )

    # Match the training path to reduce unnecessary memory churn.
    env.env.hard_reset = False
    action_space = _resolve_action_space(env)

    for reset_idx in range(args.resets):
        start = time.time()
        obs = env.reset()
        duration = time.time() - start
        print(f"Reset {reset_idx + 1}/{args.resets} finished in {duration:.3f}s")
        if reset_idx == 0:
            _print_obs_summary(obs)

        done = False
        steps = 0
        while not done and steps < args.steps:
            obs, reward, done, info = env.step(action_space.sample())
            steps += 1
        print(
            f"Rollout {reset_idx + 1}: steps={steps}, done={done}, "
            f"last_reward={reward}"
        )

    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()
    elif hasattr(env, "env") and callable(getattr(env.env, "close", None)):
        env.env.close()
    print("Robomimic render smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"Robomimic render smoke test failed: {exc}", file=sys.stderr)
        raise
