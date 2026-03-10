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
PPO fine-tuning for Drifting Policy.

Uses Gaussian policy approximation: the deterministic single-step output of the
Drifting Policy serves as the mean, with a learned exploration noise standard
deviation on top. This converts the 1-NFE Drifting Policy into a standard
Gaussian policy suitable for PPO log-probability computation.
"""

import logging
log = logging.getLogger(__name__)
from model.flow.mlp_meanflow import MeanFlowMLP
from model.flow.ft_ppo.ppoflow import PPOFlow
import torch
from torch import Tensor
from torch.distributions import Normal


class NoisyDriftingMLP(torch.nn.Module):
    """
    Noisy version of the Drifting Policy network for PPO fine-tuning.
    
    Wraps the pretrained MeanFlowMLP and adds learnable exploration noise.
    The policy uses t=1.0, r=0.0 for single-step inference, then adds
    Gaussian exploration noise for PPO training.
    """
    def __init__(self,
                 policy: MeanFlowMLP,
                 denoising_steps,
                 learn_explore_noise_from,
                 inital_noise_scheduler_type,
                 min_logprob_denoising_std,
                 max_logprob_denoising_std,
                 learn_explore_time_embedding,
                 time_dim_explore,
                 use_time_independent_noise,
                 device,
                 noise_hidden_dims,
                 activation_type):
        super().__init__()

        self.policy = policy
        self.denoising_steps = denoising_steps
        self.learn_explore_noise_from = learn_explore_noise_from
        self.device = device

        # Noise parameters for stability
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(min_logprob_denoising_std**2, device=device)),
            requires_grad=False
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(max_logprob_denoising_std**2, device=device)),
            requires_grad=False
        )

        # Time embedding for exploration noise
        if learn_explore_time_embedding:
            from model.diffusion.modules import SinusoidalPosEmb
            self.time_embedding_explore = torch.nn.Sequential(
                SinusoidalPosEmb(time_dim_explore),
                torch.nn.Linear(time_dim_explore, time_dim_explore * 2),
                torch.nn.Mish(),
                torch.nn.Linear(time_dim_explore * 2, time_dim_explore),
            ).to(device)
        else:
            self.time_embedding_explore = None

        # MLP for noise prediction
        self.use_time_independent_noise = use_time_independent_noise
        if use_time_independent_noise:
            input_dim = policy.act_dim_total + policy.cond_enc_dim
        else:
            input_dim = policy.act_dim_total + policy.cond_enc_dim + time_dim_explore

        from model.common.mlp import MLP
        self.mlp_logvar = MLP(
            [input_dim] + noise_hidden_dims + [policy.act_dim_total],
            activation_type=activation_type,
            out_activation_type="Identity",
        ).to(device)

    def forward(self, z, cond, learn_exploration_noise=True, step=None):
        """
        Forward pass for noisy Drifting Policy.

        Uses t=1.0 and r=0.0 for single-step generation, then predicts
        exploration noise standard deviation.

        Args:
            z: (B, Ta, Da) input noise
            cond: condition dict with 'state' key
            learn_exploration_noise: whether to predict exploration noise
            step: current step index (used for noise learning schedule)

        Returns:
            action: (B, Ta, Da) predicted action (deterministic mean)
            noise_std: (B, Ta*Da) exploration noise standard deviation
        """
        B, Ta, Da = z.shape

        # Single-step generation: t=1.0, r=0.0
        t = torch.ones(B, device=self.device)
        r = torch.zeros(B, device=self.device)

        # Get deterministic action from the policy
        action = self.policy(z, t, r, cond)

        # Predict exploration noise
        if learn_exploration_noise and step is not None and step >= self.learn_explore_noise_from:
            action_flat = action.view(B, -1)
            cond_encoded = self.policy.forward_encoder(cond)

            if self.use_time_independent_noise:
                noise_input = torch.cat([action_flat, cond_encoded], dim=-1)
            else:
                if self.time_embedding_explore is not None:
                    time_tensor = torch.ones((B, 1), device=self.device)
                    time_emb = self.time_embedding_explore(time_tensor).view(B, -1)
                    noise_input = torch.cat([action_flat, cond_encoded, time_emb], dim=-1)
                else:
                    noise_input = torch.cat([action_flat, cond_encoded], dim=-1)

            logvar = self.mlp_logvar(noise_input)
            logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
            noise_std = torch.exp(0.5 * logvar)
        else:
            noise_std = torch.exp(0.5 * self.logvar_min).expand(B, Ta * Da)

        return action, noise_std


class PPODrifting(PPOFlow):
    """
    PPO fine-tuning for Drifting Policy.
    
    Since Drifting Policy uses 1-NFE inference, the PPO log-probability
    computation simplifies to a single Gaussian transition:
        p(a | s) = N(a | mu(z, s), sigma^2)
    where mu is the deterministic Drifting Policy output and sigma is learned.
    """

    def __init__(self,
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min,
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type
                 ):

        super().__init__(
            device,
            policy,
            critic,
            actor_policy_path,
            act_dim,
            horizon_steps,
            act_min,
            act_max,
            obs_dim,
            cond_steps,
            noise_scheduler_type,
            inference_steps,
            ft_denoising_steps,
            randn_clip_value,
            min_sampling_denoising_std,
            min_logprob_denoising_std,
            logprob_min,
            logprob_max,
            clip_ploss_coef,
            clip_ploss_coef_base,
            clip_ploss_coef_rate,
            clip_vloss_coef,
            denoised_clip_value,
            max_logprob_denoising_std,
            time_dim_explore,
            learn_explore_time_embedding,
            use_time_independent_noise,
            noise_hidden_dims,
            logprob_debug_sample,
            logprob_debug_recalculate,
            explore_net_activation_type
        )

    def init_actor_ft(self, policy_copy):
        """Initialize fine-tuning actor with noisy Drifting Policy."""
        self.actor_ft = NoisyDriftingMLP(
            policy=policy_copy,
            denoising_steps=self.inference_steps,
            learn_explore_noise_from=self.inference_steps - self.ft_denoising_steps,
            inital_noise_scheduler_type=self.noise_scheduler_type,
            min_logprob_denoising_std=self.min_logprob_denoising_std,
            max_logprob_denoising_std=self.max_logprob_denoising_std,
            learn_explore_time_embedding=self.learn_explore_time_embedding,
            time_dim_explore=self.time_dim_explore,
            use_time_independent_noise=self.use_time_independent_noise,
            device=self.device,
            noise_hidden_dims=self.noise_hidden_dims,
            activation_type=self.explore_net_activation_type
        )

    def get_logprobs(self,
                     cond: dict,
                     x_chain: Tensor,
                     get_entropy=False,
                     normalize_denoising_horizon=False,
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True
                     ):
        """
        Compute log probabilities for the Drifting Policy.

        Since Drifting Policy is 1-NFE, the chain has exactly 2 elements:
        x_chain[:, 0] = initial noise z ~ N(0, I)
        x_chain[:, 1] = generated action

        The transition probability is:
            p(action | z, s) = N(action | mu(z, s), sigma^2)
        """
        logprob = 0.0
        joint_entropy = 0.0
        entropy_rate_est = 0.0
        logprob_steps = 0

        B = x_chain.shape[0]

        # Initial probability p(z) ~ N(0, 1)
        init_dist = Normal(
            torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device), 1.0
        )
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)
        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1

        # Single-step transition: z -> action
        z = x_chain[:, 0]  # (B, Ta, Da)
        action_target = x_chain[:, -1].reshape(B, -1)  # (B, Ta*Da)

        # Forward through the noisy actor
        action_mean, noise_std = self.actor_ft.forward(z, cond, True, 0)
        action_mean_flat = action_mean.reshape(B, -1)  # (B, Ta*Da)

        if clip_intermediate_actions:
            action_mean_flat = action_mean_flat.clamp(
                -self.denoised_clip_value, self.denoised_clip_value
            )

        # Transition distribution
        dist = Normal(action_mean_flat, noise_std)
        logprob_trans = dist.log_prob(action_target).sum(-1)  # (B,)
        logprob += logprob_trans
        logprob_steps += 1

        if get_entropy:
            entropy_trans = dist.entropy().sum(-1)
            joint_entropy += entropy_trans

        if get_entropy:
            entropy_rate_est = joint_entropy / logprob_steps
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
        if normalize_act_space_dimension:
            logprob = logprob / self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est / self.act_dim_total

        if verbose_entropy_stats and get_entropy:
            log.info(
                f"entropy_rate_est={entropy_rate_est.shape} Entropy Percentiles: "
                f"10%={entropy_rate_est.quantile(0.1):.2f}, "
                f"50%={entropy_rate_est.median():.2f}, "
                f"90%={entropy_rate_est.quantile(0.9):.2f}"
            )

        chains_stds_mean = noise_std.mean()

        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds_mean
            return logprob, entropy_rate_est,
        else:
            if get_chains_stds:
                return logprob, chains_stds_mean
            return logprob

    @torch.no_grad()
    def get_actions(self,
                    cond: dict,
                    eval_mode: bool,
                    save_chains=False,
                    normalize_denoising_horizon=False,
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True,
                    ret_logprob=True
                    ):
        """
        Generate actions using the Drifting Policy (1 NFE).

        The sampling procedure is:
            1. Sample z ~ N(0, I)
            2. Compute action_mean = network(z, cond) with t=1.0, r=0.0
            3. Add exploration noise: action = action_mean + sigma * eps
        """
        B = cond["state"].shape[0]

        if save_chains:
            x_chain = torch.zeros(
                (B, 2, self.horizon_steps, self.action_dim), device=self.device
            )
        if ret_logprob:
            log_prob = 0.0
            log_prob_steps = 0
            if self.logprob_debug_sample:
                log_prob_list = []

        # Sample initial noise
        xt, log_prob_init = self.sample_first_point(B)
        if ret_logprob and account_for_initial_stochasticity:
            log_prob += log_prob_init
            log_prob_steps += 1
            if self.logprob_debug_sample:
                log_prob_list.append(log_prob_init.mean().item())

        if save_chains:
            x_chain[:, 0] = xt

        # Single-step generation
        action_mean, noise_std = self.actor_ft.forward(
            xt, cond, learn_exploration_noise=not eval_mode, step=0
        )

        if clip_intermediate_actions:
            action_mean = action_mean.clamp(
                -self.denoised_clip_value, self.denoised_clip_value
            )

        # Add exploration noise
        std = noise_std.unsqueeze(-1).reshape(action_mean.shape)
        std = torch.clamp(std, min=self.min_sampling_denoising_std)
        dist = Normal(action_mean, std)

        if not eval_mode:
            xt = dist.sample().clamp_(
                dist.loc - self.randn_clip_value * dist.scale,
                dist.loc + self.randn_clip_value * dist.scale
            ).to(self.device)
        else:
            xt = action_mean

        # Final action clipping
        xt = xt.clamp_(self.act_min, self.act_max)

        if ret_logprob:
            logprob_transition = dist.log_prob(xt).sum(dim=(-2, -1)).to(self.device)
            if self.logprob_debug_sample:
                log_prob_list.append(logprob_transition.mean().item())
            log_prob += logprob_transition
            log_prob_steps += 1

        if save_chains:
            x_chain[:, 1] = xt

        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob / log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob / self.act_dim_total
            if self.logprob_debug_sample:
                transform_logprob = torch.log(
                    1 - torch.tanh(xt) ** 2 + 1e-7
                ).sum(dim=(-2, -1)).mean().item()
                print(f"log_prob_list={log_prob_list}, transform={transform_logprob}")

        if ret_logprob:
            if save_chains:
                return (xt, x_chain, log_prob)
            return (xt, log_prob)
        else:
            if save_chains:
                return (xt, x_chain)
            return xt
