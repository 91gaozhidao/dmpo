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
PPO fine-tuning for Drifting Policy with low-dimensional observations (gym).

Since Drifting Policy is 1-NFE, the inference chain is always of length 2
(initial noise + generated action). This simplifies the PPO update compared
to multi-step flow models.
"""
import os
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
import numpy as np
import torch
from agent.finetune.reinflow.train_ppo_shortcut_agent import TrainPPOShortCutAgent
from model.drifting.ft_ppo.ppodrifting import PPODrifting
from agent.finetune.reinflow.buffer import PPOFlowBuffer


class TrainPPODriftingAgent(TrainPPOShortCutAgent):
    """
    Training agent for PPO fine-tuning of Drifting Policy with low-dimensional
    observations.

    Inherits from TrainPPOShortCutAgent and overrides Drifting-specific components.
    Since Drifting Policy is 1-NFE, inference_steps is forced to 1.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        log.info("Initialized Drifting Policy PPO training agent with low-dim observations")

        # Drifting Policy uses 1 inference step (1 NFE)
        self.inference_steps = 1

        self.initial_ratio_error_threshold = 1e-5

    def agent_update(self, verbose=True):
        """
        Agent update for Drifting Policy PPO.

        The PPO loss computation is the same as the parent, but the underlying
        policy is 1-NFE Drifting Policy.
        """
        clipfracs_list = []
        noise_std_list = []
        actor_norm = 0.0
        critic_norm = 0.0

        for update_epoch, batch_id, minibatch in self.minibatch_generator() if not self.repeat_samples else self.minibatch_generator_repeat():
            self.model: PPODrifting

            pg_loss, entropy_loss, v_loss, bc_loss, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
                newlogprob_min, newlogprob_max, newlogprob_std, \
                noise_std, newQ_values = self.model.loss(*minibatch,
                                                    use_bc_loss=self.use_bc_loss,
                                                    bc_loss_type=self.bc_loss_type,
                                                    normalize_denoising_horizon=self.normalize_denoising_horizon,
                                                    normalize_act_space_dimension=self.normalize_act_space_dim,
                                                    verbose=verbose,
                                                    clip_intermediate_actions=self.clip_intermediate_actions,
                                                    account_for_initial_stochasticity=self.account_for_initial_stochasticity)
            self.approx_kl = approx_kl

            if verbose:
                log.info(f"Drifting update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")

            if update_epoch == 0 and batch_id == 0 and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold:
                log.warning(f"Warning: Drifting ratio={ratio} not 1.00 when update_epoch==0 and batch_id==0, there might be bugs in the implementation!")

            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)

            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff

            clipfracs_list += [clipfrac]
            noise_std_list += [noise_std]

            loss.backward()

            if (batch_id + 1) % self.grad_accumulate == 0:
                actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
                critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
                if verbose:
                    log.info(f"Drifting before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}")

                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                if verbose:
                    log.info(f"Drifting grad update at batch {batch_id}")
                    log.info(f"Drifting approx_kl: {approx_kl}, update_epoch: {update_epoch}/{self.update_epochs}, num_batch: {self.total_steps // self.batch_size}")

        clip_fracs = np.mean(clipfracs_list)
        noise_stds = np.mean(noise_std_list)

        self.train_ret_dict = {
            "loss": loss,
            "pg loss": pg_loss,
            "value loss": v_loss,
            "entropy_loss": entropy_loss,
            "bc_loss": bc_loss,
            "approx kl": self.approx_kl,
            "ratio": ratio,
            "clipfrac": clip_fracs,
            "explained variance": self.explained_var,
            "old_logprob_min": oldlogprob_min,
            "old_logprob_max": oldlogprob_max,
            "old_logprob_std": oldlogprob_std,
            "new_logprob_min": newlogprob_min,
            "new_logprob_max": newlogprob_max,
            "new_logprob_std": newlogprob_std,
            "actor_norm": actor_norm,
            "critic_norm": critic_norm,
            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic lr": self.critic_optimizer.param_groups[0]["lr"],
            "min_logprob_noise_std": self.model.min_logprob_denoising_std,
            "min_sampling_noise_std": self.model.min_sampling_denoising_std,
            "noise_std": noise_stds,
            "Q_values": self.Q_values
        }

    def run(self):
        """
        Main training loop for Drifting Policy PPO fine-tuning.
        """
        log.info("Starting Drifting Policy PPO fine-tuning training loop (1-NFE)")

        self.init_buffer()
        self.prepare_run()
        self.buffer.reset()

        if self.resume:
            self.resume_training()

        train_itr_pbar = tqdm(
            total=self.n_train_itr,
            desc="Drifting Policy Training Iterations",
            unit="itr",
            dynamic_ncols=True,
            ascii=True,
            initial=self.itr
        )

        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env(buffer_device=self.buffer_device)
            self.buffer.update_full_obs()

            for step in tqdm(range(self.n_steps), desc="Collecting samples", leave=False) if self.verbose else range(self.n_steps):
                if not self.verbose and step % 100 == 0:
                    print(f"Drifting processed {step} of {self.n_steps}")

                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(self.prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }

                    action_samples, chains_venv = self.get_samples(
                        cond=cond,
                        ret_device=self.buffer_device,
                        normalize_denoising_horizon=self.normalize_denoising_horizon,
                        normalize_act_space_dimension=self.normalize_act_space_dim,
                        clip_intermediate_actions=self.clip_intermediate_actions,
                        account_for_initial_stochasticity=self.account_for_initial_stochasticity
                    )

                action_venv = action_samples[:, :self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)

                self.buffer.add(step, self.prev_obs_venv, chains_venv, reward_venv, terminated_venv, truncated_venv)

                self.prev_obs_venv = obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0

            self.buffer.summarize_episode_reward()

            if not self.eval_mode:
                self.buffer: PPOFlowBuffer
                self.buffer.update(obs_venv, self.model)
                self.agent_update(verbose=self.verbose)

            self.log()
            self.update_lr()
            self.update_bc_coeff()
            self.adjust_finetune_schedule()
            self.save_model()

            if self.itr % self.log_freq == 0:
                train_itr_pbar.set_postfix({
                    'mode': 'Eval' if self.eval_mode else 'Train',
                    'reward': f'{self.buffer.avg_episode_reward:.2f}',
                    'success': f'{self.buffer.success_rate*100:.1f}%'
                })
            train_itr_pbar.update(1)

            self.itr += 1

            if self.use_early_stop and (self.buffer.success_rate < 0.05 or self.buffer.avg_episode_reward < 2.0):
                log.error(f"Drifting finetuning failed. success_rate={self.buffer.success_rate*100:.2f}% and avg_episode_reward={self.buffer.avg_episode_reward:.2f}")
                train_itr_pbar.close()
                exit()

            self.clear_cache()
            self.inspect_memory()

        train_itr_pbar.close()
        log.info("Drifting Policy PPO fine-tuning completed successfully!")
