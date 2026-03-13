# MIT License
#
# Copyright (c) 2026 ReinFlow Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Critic-guided drifting fine-tuning.

This module keeps drifting's native actor update:
    x <- x + V_{p,q}(x)

The critic is only used to identify which policy samples should populate the
high-value positive set p(a | s) and the current policy set q(a | s).
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Dict, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.drifting.drifting import DriftingPolicy

log = logging.getLogger(__name__)


def _find_run_config_path(checkpoint_path: str) -> str:
    checkpoint_path = os.path.abspath(checkpoint_path)
    search_dir = os.path.dirname(checkpoint_path)
    checked = set()

    for _ in range(4):
        candidate = os.path.join(search_dir, ".hydra", "config.yaml")
        if candidate not in checked and os.path.exists(candidate):
            return candidate
        checked.add(candidate)
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:
            break
        search_dir = parent_dir

    raise FileNotFoundError(
        f"Could not find a sibling Hydra config for checkpoint: {checkpoint_path}"
    )


def _build_drifting_policy_cfg(run_cfg, device: torch.device):
    model_cfg = run_cfg.model
    model_target = model_cfg.get("_target_")

    if model_target == "model.drifting.drifting.DriftingPolicy":
        drifting_cfg = OmegaConf.create(
            OmegaConf.to_container(model_cfg, resolve=True)
        )
    elif model_target in {
        "model.drifting.ft_ppo.ppodrifting.PPODrifting",
        "model.drifting.ft_grpo.grpodrifting.GRPODrifting",
    }:
        backbone_cfg = model_cfg.get("policy", model_cfg.get("backbone"))
        drifting_cfg = OmegaConf.create(
            {
                "_target_": "model.drifting.drifting.DriftingPolicy",
                "network": OmegaConf.to_container(backbone_cfg, resolve=True),
                "device": str(device),
                "horizon_steps": model_cfg.horizon_steps,
                "action_dim": model_cfg.act_dim,
                "act_min": model_cfg.act_min,
                "act_max": model_cfg.act_max,
                "obs_dim": model_cfg.obs_dim,
                "max_denoising_steps": 1,
                "seed": run_cfg.get("seed", 0),
            }
        )
    else:
        raise ValueError(
            "Q-guided drifting can only bootstrap from DriftingPolicy or "
            f"legacy drifting RL checkpoints, but found model target: {model_target}"
        )

    drifting_cfg["device"] = str(device)
    return drifting_cfg


def _extract_prefixed_state_dict(state_dict, prefix: str, replacement: str = ""):
    extracted = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            extracted[f"{replacement}{key[len(prefix):]}"] = value
    return extracted if extracted else None


def _extract_drifting_state_dict(checkpoint, use_ema: bool):
    state_dict_candidates = []
    if use_ema and isinstance(checkpoint.get("ema"), dict):
        state_dict_candidates.append(("ema", checkpoint["ema"]))
    if isinstance(checkpoint.get("model"), dict):
        state_dict_candidates.append(("model", checkpoint["model"]))
    if isinstance(checkpoint, dict) and any(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict_candidates.append(("checkpoint", checkpoint))

    for source_name, state_dict in state_dict_candidates:
        if any(key.startswith("network.") for key in state_dict):
            return state_dict, source_name

        for prefix, replacement in (
            ("actor.drifting_policy.", ""),
            ("actor_ft.policy.", "network."),
            ("actor_old.", "network."),
            ("actor_ref.drifting_policy.", ""),
            ("actor.", ""),
        ):
            extracted = _extract_prefixed_state_dict(
                state_dict, prefix=prefix, replacement=replacement
            )
            if extracted is not None:
                return extracted, f"{source_name}:{prefix}"

    raise ValueError(
        "Could not recover a DriftingPolicy state dict from checkpoint. "
        f"Available top-level keys: {list(checkpoint.keys())}"
    )


def load_drifting_policy_from_checkpoint(
    actor_policy_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> DriftingPolicy:
    checkpoint = torch.load(actor_policy_path, map_location=device)
    run_cfg = OmegaConf.load(_find_run_config_path(actor_policy_path))
    drifting_cfg = _build_drifting_policy_cfg(run_cfg, device=device)
    drifting_policy = hydra.utils.instantiate(drifting_cfg).to(device)

    drifting_state_dict, source_name = _extract_drifting_state_dict(
        checkpoint, use_ema=use_ema
    )
    missing_keys, unexpected_keys = drifting_policy.load_state_dict(
        drifting_state_dict,
        strict=False,
    )
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Failed to load drifting backbone from checkpoint "
            f"`{actor_policy_path}` ({source_name}). "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )

    log.info(
        "Loaded DriftingPolicy backbone for Q-guided fine-tuning from %s (%s)",
        actor_policy_path,
        source_name,
    )
    return drifting_policy.eval()


def _repeat_condition(cond: Dict[str, torch.Tensor], repeats: int):
    return {
        key: value.repeat_interleave(repeats, dim=0)
        for key, value in cond.items()
    }


def _gather_action_samples(action_samples: torch.Tensor, indices: torch.Tensor):
    gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(
        -1,
        -1,
        action_samples.shape[2],
        action_samples.shape[3],
    )
    return torch.gather(action_samples, dim=1, index=gather_index)


def compute_conditioned_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: torch.Tensor,
):
    """
    Batched conditional version of drifting's V_{p,q}.

    Args:
        x: [B, Q, D] differentiable query samples.
        y_pos: [B, P, D] high-value positive samples.
        y_neg: [B, N, D] current policy samples.
        temperature: scalar temperature tensor.
    """
    n_neg = y_neg.shape[1]
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)
    dist = torch.cat([dist_neg, dist_pos], dim=-1)

    log_kernel = (-dist / temperature).clamp(min=-80.0)
    kernel = log_kernel.exp()

    row_sum = kernel.sum(dim=-1, keepdim=True)
    col_sum = kernel.sum(dim=-2, keepdim=True)
    normalizer = (row_sum * col_sum).clamp_min(1e-12).sqrt()
    affinity = kernel / normalizer

    affinity_neg = affinity[:, :, :n_neg]
    affinity_pos = affinity[:, :, n_neg:]

    weight_pos = affinity_pos * affinity_neg.sum(dim=-1, keepdim=True)
    weight_neg = affinity_neg * affinity_pos.sum(dim=-1, keepdim=True)

    drift_pos = torch.einsum("bqp,bpd->bqd", weight_pos, y_pos)
    drift_neg = torch.einsum("bqn,bnd->bqd", weight_neg, y_neg)
    drift = drift_pos - drift_neg
    return drift, dist_pos, dist_neg


def compute_conditioned_drifting_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures,
):
    """
    Conditional drifting loss that preserves the paper's x -> x + V update form.
    """
    with torch.no_grad():
        mean_dist = torch.cdist(x, y_pos).mean()

    metrics = {
        "actor/mean_cross_dist": mean_dist.item(),
    }
    drift_total = torch.zeros_like(x)
    last_dist_pos = None
    last_dist_neg = None

    for base_temperature in temperatures:
        adaptive_temperature = (base_temperature * mean_dist).detach().clamp_min(1e-6)
        drift_t, dist_pos, dist_neg = compute_conditioned_V(
            x=x,
            y_pos=y_pos,
            y_neg=y_neg,
            temperature=adaptive_temperature,
        )
        drift_total = drift_total + drift_t
        last_dist_pos = dist_pos
        last_dist_neg = dist_neg

        with torch.no_grad():
            drift_rms_sq = torch.mean(torch.sum(drift_t**2, dim=-1)) / x.shape[-1]
            lambda_t = torch.sqrt(drift_rms_sq + 1e-8)
            metrics[f"actor/drifting_lambda_T{base_temperature}"] = lambda_t.item()

    target = (x + drift_total).detach()
    loss = F.mse_loss(x, target)

    with torch.no_grad():
        drift_norms = torch.norm(drift_total, dim=-1)
        metrics["actor/V_norm_mean"] = drift_norms.mean().item()
        metrics["actor/V_norm_max"] = drift_norms.max().item()
        metrics["actor/V_norm_std"] = drift_norms.std().item()
        if last_dist_pos is not None:
            metrics["actor/dist_to_pos_mean"] = last_dist_pos.mean().item()
            metrics["actor/dist_to_neg_mean"] = last_dist_neg.mean().item()

    return loss, metrics


class QGuidedDrifting(nn.Module):
    """
    Off-policy critic with drifting-native actor improvement.
    """

    def __init__(
        self,
        device,
        policy,
        critic,
        act_dim,
        horizon_steps,
        act_steps,
        act_min,
        act_max,
        obs_dim,
        cond_steps,
        actor_policy_path=None,
        seed=0,
        temperatures=(0.1,),
        mask_self=False,
        num_action_samples=8,
        num_positive_samples=2,
        num_query_samples=4,
        sample_latent_scale=1.0,
        eval_latent_scale=0.0,
        reference_anchor_coeff=0.0,
        use_ema_checkpoint=True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_steps = act_steps
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        self.act_min = act_min
        self.act_max = act_max
        self.temperatures = tuple(temperatures)
        self.mask_self = mask_self
        self.num_action_samples = int(num_action_samples)
        self.num_positive_samples = int(num_positive_samples)
        self.num_query_samples = int(num_query_samples)
        self.sample_latent_scale = float(sample_latent_scale)
        self.eval_latent_scale = float(eval_latent_scale)
        self.reference_anchor_coeff = float(reference_anchor_coeff)
        self.use_ema_checkpoint = use_ema_checkpoint

        if self.num_action_samples <= 0:
            raise ValueError("num_action_samples must be positive")
        if self.num_positive_samples <= 0:
            raise ValueError("num_positive_samples must be positive")
        if self.num_positive_samples > self.num_action_samples:
            raise ValueError("num_positive_samples must be <= num_action_samples")
        if self.num_query_samples <= 0:
            raise ValueError("num_query_samples must be positive")
        if self.act_steps > self.horizon_steps:
            raise ValueError("act_steps must be <= horizon_steps")

        if isinstance(policy, DriftingPolicy):
            self.actor = policy.to(self.device)
        else:
            self.actor = DriftingPolicy(
                network=policy,
                device=self.device,
                horizon_steps=horizon_steps,
                action_dim=act_dim,
                act_min=act_min,
                act_max=act_max,
                obs_dim=obs_dim,
                max_denoising_steps=1,
                seed=seed,
                temperatures=list(self.temperatures),
                mask_self=mask_self,
            ).to(self.device)

        if actor_policy_path:
            loaded_actor = load_drifting_policy_from_checkpoint(
                actor_policy_path,
                device=self.device,
                use_ema=self.use_ema_checkpoint,
            )
            self.actor.load_state_dict(loaded_actor.state_dict(), strict=True)

        self.reference_actor = copy.deepcopy(self.actor).eval()
        for param in self.reference_actor.parameters():
            param.requires_grad = False

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.report_network_params()

    def report_network_params(self):
        log.info(
            "Number of network parameters: Total %.3f M | Actor %.3f M | Critic %.3f M",
            sum(p.numel() for p in self.parameters()) / 1e6,
            sum(p.numel() for p in self.actor.parameters()) / 1e6,
            sum(p.numel() for p in self.critic.parameters()) / 1e6,
        )

    def _sample_actor_actions(
        self,
        cond: Dict[str, torch.Tensor],
        num_samples: int,
        deterministic: bool,
        latent_scale: float,
        detach: bool,
        actor_override: DriftingPolicy | None = None,
    ):
        actor = self.actor if actor_override is None else actor_override
        batch_size = cond["state"].shape[0]
        repeated_cond = _repeat_condition(cond, num_samples)
        if deterministic:
            latents = torch.zeros(
                batch_size * num_samples,
                self.horizon_steps,
                self.action_dim,
                device=self.device,
            )
        else:
            latents = torch.randn(
                batch_size * num_samples,
                self.horizon_steps,
                self.action_dim,
                device=self.device,
            ) * latent_scale
        actions = actor.predict(
            cond=repeated_cond,
            z=latents,
            clip_actions=True,
        ).view(batch_size, num_samples, self.horizon_steps, self.action_dim)
        latents = latents.view(batch_size, num_samples, self.horizon_steps, self.action_dim)
        if detach:
            actions = actions.detach()
            latents = latents.detach()
        return actions, latents

    def _min_q(self, cond: Dict[str, torch.Tensor], actions: torch.Tensor):
        q_values = self.critic(cond, actions)
        if isinstance(q_values, tuple):
            q1, q2 = q_values
            return torch.min(q1, q2), q1, q2
        return q_values, q_values, q_values

    def _target_min_q(self, cond: Dict[str, torch.Tensor], actions: torch.Tensor):
        q_values = self.target_critic(cond, actions)
        if isinstance(q_values, tuple):
            q1, q2 = q_values
            return torch.min(q1, q2)
        return q_values

    @torch.no_grad()
    def forward(self, cond: Dict[str, torch.Tensor], deterministic: bool = False):
        actions, _ = self._sample_actor_actions(
            cond=cond,
            num_samples=1,
            deterministic=deterministic,
            latent_scale=self.eval_latent_scale if deterministic else self.sample_latent_scale,
            detach=True,
        )
        return actions[:, 0]

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        rewards,
        terminated,
        gamma,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        executed_actions = actions[:, : self.act_steps]
        current_q1, current_q2 = self.critic(obs, executed_actions)

        with torch.no_grad():
            next_actions = self.forward(next_obs, deterministic=False)[:, : self.act_steps]
            next_q = self._target_min_q(next_obs, next_actions)
            rewards = rewards.view(-1)
            terminated = terminated.view(-1)
            target_q = rewards + gamma * (1 - terminated) * next_q

        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        critic_loss = critic_loss_1 + critic_loss_2
        metrics = {
            "critic/loss": critic_loss.item(),
            "critic/loss_q1": critic_loss_1.item(),
            "critic/loss_q2": critic_loss_2.item(),
            "critic/q1_mean": current_q1.mean().item(),
            "critic/q2_mean": current_q2.mean().item(),
            "critic/target_mean": target_q.mean().item(),
        }
        return critic_loss, metrics

    def loss_actor(self, obs) -> Tuple[torch.Tensor, Dict[str, float]]:
        query_actions, query_latents = self._sample_actor_actions(
            cond=obs,
            num_samples=self.num_query_samples,
            deterministic=False,
            latent_scale=self.sample_latent_scale,
            detach=False,
        )

        with torch.no_grad():
            candidate_actions, _ = self._sample_actor_actions(
                cond=obs,
                num_samples=self.num_action_samples,
                deterministic=False,
                latent_scale=self.sample_latent_scale,
                detach=True,
            )

            batch_size = candidate_actions.shape[0]
            repeated_obs = _repeat_condition(obs, self.num_action_samples)
            candidate_q, _, _ = self._min_q(
                repeated_obs,
                candidate_actions[:, :, : self.act_steps].reshape(
                    batch_size * self.num_action_samples,
                    self.act_steps,
                    self.action_dim,
                ),
            )
            candidate_q = candidate_q.view(batch_size, self.num_action_samples)
            top_indices = candidate_q.topk(
                k=self.num_positive_samples,
                dim=1,
                largest=True,
            ).indices
            positive_actions = _gather_action_samples(candidate_actions, top_indices)

        query_actions_flat = query_actions.reshape(
            query_actions.shape[0],
            self.num_query_samples,
            -1,
        )
        positive_actions_flat = positive_actions.reshape(
            positive_actions.shape[0],
            self.num_positive_samples,
            -1,
        )
        negative_actions_flat = candidate_actions.reshape(
            candidate_actions.shape[0],
            self.num_action_samples,
            -1,
        )

        actor_loss, metrics = compute_conditioned_drifting_loss(
            x=query_actions_flat,
            y_pos=positive_actions_flat,
            y_neg=negative_actions_flat,
            temperatures=self.temperatures,
        )

        anchor_loss = torch.tensor(0.0, device=self.device)
        if self.reference_anchor_coeff > 0:
            repeated_obs = _repeat_condition(obs, self.num_query_samples)
            reference_actions = self.reference_actor.predict(
                cond=repeated_obs,
                z=query_latents.reshape(
                    query_latents.shape[0] * query_latents.shape[1],
                    self.horizon_steps,
                    self.action_dim,
                ),
                clip_actions=True,
            ).view_as(query_actions)
            anchor_loss = F.mse_loss(query_actions, reference_actions.detach())
            actor_loss = actor_loss + self.reference_anchor_coeff * anchor_loss
            metrics["actor/reference_anchor_loss"] = anchor_loss.item()

        with torch.no_grad():
            batch_size = query_actions.shape[0]
            repeated_obs = _repeat_condition(obs, self.num_query_samples)
            query_q, _, _ = self._min_q(
                repeated_obs,
                query_actions[:, :, : self.act_steps].reshape(
                    batch_size * self.num_query_samples,
                    self.act_steps,
                    self.action_dim,
                ),
            )
            query_q = query_q.view(batch_size, self.num_query_samples)
            metrics.update(
                {
                    "actor/loss": actor_loss.item(),
                    "actor/query_q_mean": query_q.mean().item(),
                    "actor/query_q_max": query_q.max().item(),
                    "actor/positive_q_mean": candidate_q.topk(
                        k=self.num_positive_samples,
                        dim=1,
                        largest=True,
                    ).values.mean().item(),
                    "actor/candidate_q_mean": candidate_q.mean().item(),
                    "actor/candidate_q_min": candidate_q.min().item(),
                    "actor/candidate_q_max": candidate_q.max().item(),
                }
            )

        return actor_loss, metrics

    def update_target_critic(self, tau: float):
        for target_param, source_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )
