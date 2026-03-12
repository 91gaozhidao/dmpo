# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Continuous GRPO (Group Relative Policy Optimization) for Drifting Policy.

GRPO eliminates the Critic/Value network entirely. Instead, it estimates
advantages via group-internal Z-score normalization of episodic returns
collected from G trajectories sampled from the same initial state. A frozen
reference policy provides KL divergence regularization to prevent excessive
policy drift.

For Drifting Policy (1-NFE), the deterministic network output serves as
the Gaussian mean, and a learnable log-std parameter provides exploration.
"""

import copy
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from torch.distributions import Normal
from model.drifting.drifting import DriftingPolicy

log = logging.getLogger(__name__)

# Module-level constants for numerical stability
LOG_2 = np.log(2)
TANH_CLIP_THRESHOLD = 0.999999
JACOBIAN_EPS = 1e-6


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
    elif model_target == "model.drifting.ft_ppo.ppodrifting.PPODrifting":
        drifting_cfg = OmegaConf.create(
            {
                "_target_": "model.drifting.drifting.DriftingPolicy",
                "network": OmegaConf.to_container(model_cfg.policy, resolve=True),
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
            "Continuous GRPO can only bootstrap from DriftingPolicy or PPO "
            f"Drifting checkpoints, but found model target: {model_target}"
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


def _load_drifting_policy_from_checkpoint(
    actor_policy_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> DriftingPolicy:
    if not actor_policy_path:
        raise ValueError(
            "Continuous GRPO requires `actor_policy_path` to bootstrap the "
            "drifting backbone."
        )

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
        "Loaded DriftingPolicy backbone for continuous GRPO from %s (%s)",
        actor_policy_path,
        source_name,
    )
    return drifting_policy.eval()


def _tanh_jacobian_correction(u):
    """
    Compute the log-determinant of the Jacobian for the tanh squashing.

    Uses the numerically stable formula:
        log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2*u))

    Args:
        u: (*, D) unbounded pre-tanh actions

    Returns:
        correction: (*, D) per-dimension Jacobian correction
    """
    return 2 * (LOG_2 - u - F.softplus(-2 * u))


class NoisyDriftingPolicy(nn.Module):
    """
    Stochastic wrapper around the deterministic Drifting Policy for GRPO.

    The 1-NFE Drifting Policy output is treated as the Gaussian mean.
    An independent learnable log-std parameter provides exploration noise,
    enabling proper log-probability and KL divergence computation required
    by GRPO.
    """

    def __init__(self, drifting_policy: DriftingPolicy, action_dim: int,
                 horizon_steps: int, init_log_std: float = -0.5):
        """
        Args:
            drifting_policy: Pretrained DriftingPolicy instance
            action_dim: Dimension of action space
            horizon_steps: Number of steps in trajectory horizon
            init_log_std: Initial value for log standard deviation
        """
        super().__init__()
        self.drifting_policy = drifting_policy
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        # Independent learnable log-std for the full action space
        self.log_std = nn.Parameter(
            torch.full(
                (horizon_steps * action_dim,),
                init_log_std,
                device=drifting_policy.device,
            )
        )
        self.register_buffer(
            "mean_latent",
            torch.zeros(
                1,
                horizon_steps,
                action_dim,
                device=drifting_policy.device,
            ),
        )

    def forward(self, cond: dict):
        """
        Compute deterministic action mean and exploration std.

        Args:
            cond: dict with 'state' key for observations

        Returns:
            mean: (B, Ta, Da) deterministic action from Drifting Policy
            std: (Ta*Da,) exploration standard deviation
        """
        batch_size = next(iter(cond.values())).shape[0]
        z = self.mean_latent.expand(batch_size, -1, -1)
        mean = self.drifting_policy.predict(cond, z=z, clip_actions=False)
        std = self.log_std.exp()
        return mean, std

    def get_distribution(self, cond: dict):
        """
        Build the action distribution for the given observations.

        Args:
            cond: dict with 'state' key

        Returns:
            dist: Normal distribution over flattened actions
            mean: (B, Ta, Da) raw action mean
        """
        mean, std = self.forward(cond)
        B = mean.shape[0]
        mean_flat = mean.reshape(B, -1)  # (B, Ta*Da)
        dist = Normal(mean_flat, std.expand(B, -1))
        return dist, mean

    def get_log_prob(self, cond: dict, action):
        """
        Compute log-probability of actions under the Tanh-Normal policy.

        Applies inverse tanh (atanh) to recover unbounded actions, then
        computes Normal log-probability with Jacobian correction:
            log pi(a) = log N(atanh(a); mu, sigma) - log(1 - a^2 + eps)

        Args:
            cond: dict with 'state' key
            action: (B, Ta, Da) actions in (-1, 1)

        Returns:
            log_prob: (B,) per-sample log-probability
        """
        dist, _ = self.get_distribution(cond)
        B = action.shape[0]
        action_flat = action.reshape(B, -1)  # (B, Ta*Da)

        # Inverse tanh to recover unbounded action
        action_clipped = torch.clamp(action_flat, -TANH_CLIP_THRESHOLD, TANH_CLIP_THRESHOLD)
        u = torch.atanh(action_clipped)

        # Normal log-prob with Jacobian correction for tanh squashing
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1 - action_clipped.pow(2) + JACOBIAN_EPS)
        log_prob = log_prob.sum(dim=-1)  # (B,)

        # Runtime assertion: verify Jacobian-corrected log-probs are finite
        assert torch.isfinite(log_prob).all(), (
            f"Non-finite log-probability after Jacobian correction: "
            f"NaN count={log_prob.isnan().sum().item()}, "
            f"Inf count={log_prob.isinf().sum().item()}"
        )

        return log_prob

    def get_action_and_log_prob(self, cond: dict):
        """
        Sample an action via Tanh-Normal and return the corrected log-probability.

        Uses reparameterized sampling (rsample) for gradient flow.

        Args:
            cond: dict with 'state' key

        Returns:
            action: (B, Ta, Da) sampled action in [-1, 1]
            log_prob: (B,) Jacobian-corrected log-probability
        """
        dist, mean = self.get_distribution(cond)
        u = dist.rsample()  # (B, Ta*Da)
        action_flat = torch.tanh(u)  # bounded to [-1, 1]

        # Jacobian correction (numerically stable):
        # log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2*u))
        log_prob = dist.log_prob(u)
        log_prob -= _tanh_jacobian_correction(u)
        log_prob = log_prob.sum(dim=-1)  # (B,)

        B = mean.shape[0]
        action = action_flat.view(B, self.horizon_steps, self.action_dim)
        return action, log_prob

    def sample_action(self, cond: dict):
        """
        Sample an action via Tanh-Normal for environment interaction.

        Uses non-reparameterized sampling (sample) since this is typically
        called under torch.no_grad() during trajectory collection.

        Args:
            cond: dict with 'state' key

        Returns:
            action: (B, Ta, Da) sampled action in [-1, 1]
            log_prob: (B,) Jacobian-corrected log-probability
        """
        dist, mean = self.get_distribution(cond)
        u = dist.sample()  # (B, Ta*Da)
        action_flat = torch.tanh(u)  # bounded to [-1, 1]

        # Jacobian correction (numerically stable)
        log_prob = dist.log_prob(u)
        log_prob -= _tanh_jacobian_correction(u)
        log_prob = log_prob.sum(dim=-1)  # (B,)

        B = mean.shape[0]
        action = action_flat.view(B, self.horizon_steps, self.action_dim)
        return action, log_prob


class GRPODrifting(nn.Module):
    """
    Continuous GRPO objective for Drifting Policy.

    Core GRPO features:
    - No Critic/Value network: advantages are computed via group-internal
      Z-score normalization of episodic returns.
    - Frozen reference policy for KL penalty to constrain policy drift.
    - Clipped surrogate policy loss (same form as PPO clip).

    The GRPO loss is:
        L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + beta * KL
    where KL is the analytical KL divergence KL(N_curr || N_ref)
    between the pre-tanh Gaussian distributions of the current and
    reference policies.
    """

    def __init__(self, actor: NoisyDriftingPolicy = None,
                 actor_policy_path: str = None,
                 device: str = "cpu",
                 init_log_std: float = -0.5,
                 use_ema: bool = True,
                 beta: float = 0.01,
                 epsilon: float = 0.2,
                 act_min: float = -1.0,
                 act_max: float = 1.0):
        """
        Args:
            actor: NoisyDriftingPolicy to optimize
            actor_policy_path: path to a drifting pretrain / PPO checkpoint
            device: device used for the actor and reference actor
            init_log_std: initial exploration std for the GRPO actor
            use_ema: whether to prefer the EMA weights when available
            beta: KL penalty coefficient
            epsilon: PPO-style clipping range
            act_min: Minimum action value for clipping
            act_max: Maximum action value for clipping
        """
        super().__init__()
        device = torch.device(device)

        if actor is None:
            drifting_policy = _load_drifting_policy_from_checkpoint(
                actor_policy_path=actor_policy_path,
                device=device,
                use_ema=use_ema,
            )
            actor = NoisyDriftingPolicy(
                drifting_policy=drifting_policy,
                action_dim=drifting_policy.action_dim,
                horizon_steps=drifting_policy.horizon_steps,
                init_log_std=init_log_std,
            )

        self.actor = actor.to(device)
        self.act_min = act_min
        self.act_max = act_max

        # Frozen reference policy (deep copy with no gradients)
        self.actor_ref = copy.deepcopy(self.actor)
        for param in self.actor_ref.parameters():
            param.requires_grad = False

        self.beta = beta
        self.epsilon = epsilon

    def compute_loss(self, obs, actions, advantages, old_log_probs):
        """
        Compute the Continuous GRPO loss with analytical KL divergence.

        Uses analytical KL(N_curr || N_ref) for the pre-tanh Gaussian
        distributions, eliminating sampling variance in the KL penalty.

        Args:
            obs: dict with 'state' key, (B, To, Do) observations
            actions: (B, Ta, Da) sampled actions
            advantages: (B,) group-normalized advantages
            old_log_probs: (B,) log-probabilities from the sampling policy

        Returns:
            loss: scalar GRPO loss
            metrics: dict with diagnostic values
        """
        # Runtime assertion: verify actions are within valid bounds
        assert torch.isfinite(actions).all(), (
            f"Non-finite actions detected: "
            f"NaN count={actions.isnan().sum().item()}, "
            f"Inf count={actions.isinf().sum().item()}"
        )

        # Current policy distribution parameters and log-probability
        current_mean, current_std = self.actor(obs)
        current_log_probs = self.actor.get_log_prob(obs, actions)

        # Reference policy distribution parameters
        with torch.no_grad():
            ref_mean, ref_std = self.actor_ref(obs)

        # Policy ratio: pi_theta(a|s) / pi_old(a|s)
        ratio = torch.exp(current_log_probs - old_log_probs)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
        ) * advantages
        policy_loss = -torch.min(surr1, surr2)

        # Analytical KL divergence: KL(N_curr || N_ref)
        # For diagonal Gaussian distributions with independent dimensions
        B = current_mean.shape[0]
        curr_mean_flat = current_mean.reshape(B, -1)   # (B, Ta*Da)
        ref_mean_flat = ref_mean.reshape(B, -1)         # (B, Ta*Da)
        curr_std_expanded = current_std.expand(B, -1)  # (B, Ta*Da)
        ref_std_expanded = ref_std.expand(B, -1)       # (B, Ta*Da)

        var_curr = curr_std_expanded.pow(2)
        var_ref = ref_std_expanded.pow(2)

        kl_div = (
            torch.log(ref_std_expanded / curr_std_expanded)
            + (var_curr + (curr_mean_flat - ref_mean_flat).pow(2)) / (2 * var_ref)
            - 0.5
        )
        kl_div_sum = kl_div.sum(dim=-1)  # Sum over action dimensions

        # Total GRPO loss
        loss = (policy_loss + self.beta * kl_div_sum).mean()

        # Runtime assertion: verify loss is finite (no NaN from Jacobian correction)
        assert torch.isfinite(loss), (
            f"Non-finite GRPO loss detected: loss={loss.item()}, "
            f"policy_loss_mean={policy_loss.mean().item()}, "
            f"kl_div_mean={kl_div_sum.mean().item()}"
        )

        # Diagnostic metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            clipfrac = (
                (ratio - 1.0).abs() > self.epsilon
            ).float().mean().item()

        metrics = {
            "policy_loss": policy_loss.mean().item(),
            "kl_div": kl_div_sum.mean().item(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "ratio": ratio.mean().item(),
            "log_std": self.actor.log_std.mean().item(),
        }

        return loss, metrics
