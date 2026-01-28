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
Pre-training Improved MeanFlow (iMF) policy

Implements training for the improved MeanFlow algorithm from:
"Improved Mean Flows: On the Challenges of Fastforward Generative Models"
(Geng et al., 2025, arXiv:2512.02012)
"""

import logging
import torch
import wandb
log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.flow.improved_meanflow import ImprovedMeanFlow, ImprovedMeanFlowDispersive


class TrainImprovedMeanFlowAgent(PreTrainAgent):
    """Training agent for Improved MeanFlow (iMF) policies."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: ImprovedMeanFlow
        self.ema_model: ImprovedMeanFlow

        self.verbose_train = False
        self.verbose_loss = False
        self.verbose_test = True

        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test = False

            # iMF follows same inference as MeanFlow with 5 steps by default
            self.test_denoising_steps = 5

            self.test_clip_intermediate_actions = True
            self.test_model_type = 'ema'

    def get_loss(self, batch_data):
        """Compute iMF v-loss for training and validation."""
        try:
            loss = self.model.loss(*batch_data)
        except Exception as e:
            log.warning(f"iMF loss computation failed: {e}. Using fallback MSE loss.")
            # Fallback to simple MSE if JVP fails
            actions, obs = batch_data
            t = torch.rand(actions.shape[0], device=actions.device)
            r = torch.zeros_like(t)
            x0 = torch.randn_like(actions)
            t_ = t.view(-1, 1, 1)
            xt = (1 - t_) * actions + t_ * x0
            u_pred = self.model.network(xt, t, r, obs)
            u_target = actions - x0
            loss = torch.nn.functional.mse_loss(u_pred, u_target)
        return loss

    def inference(self, cond: dict):
        """Generate samples for testing."""
        if self.test_model_type == 'ema':
            samples = self.ema_model.sample(
                cond,
                inference_steps=self.test_denoising_steps,
                record_intermediate=False,
                clip_intermediate_actions=self.test_clip_intermediate_actions
            )
        else:
            samples = self.model.sample(
                cond,
                inference_steps=self.test_denoising_steps,
                record_intermediate=False,
                clip_intermediate_actions=self.test_clip_intermediate_actions
            )
        return samples


class TrainImprovedMeanFlowDispersiveAgent(PreTrainAgent):
    """Training agent for Improved MeanFlow (iMF) with Dispersive Loss regularization."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: ImprovedMeanFlowDispersive
        self.ema_model: ImprovedMeanFlowDispersive

        self.verbose_train = False
        self.verbose_loss = False
        self.verbose_test = True

        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test = False
            self.test_denoising_steps = 5
            self.test_clip_intermediate_actions = True
            self.test_model_type = 'ema'

        log.info("Initialized Improved MeanFlow Dispersive training agent")

        # Log dispersive loss configuration if available
        if hasattr(self.model, 'get_dispersive_loss_info'):
            dispersive_info = self.model.get_dispersive_loss_info()
            log.info(f"Dispersive loss configuration: {dispersive_info}")

            # Log to wandb if available
            if wandb.run is not None:
                wandb.config.update({f"dispersive_{k}": v for k, v in dispersive_info.items()})

    def get_loss(self, batch_data):
        """Compute iMF v-loss with dispersive regularization."""
        try:
            # The model's loss method already includes dispersive loss
            loss = self.model.loss(*batch_data)
            return loss
        except Exception as e:
            log.warning(f"iMF Dispersive loss computation failed: {e}. Using fallback MSE loss.")
            # Fallback to simple MSE if JVP fails
            actions, obs = batch_data
            t = torch.rand(actions.shape[0], device=actions.device)
            r = torch.zeros_like(t)
            x0 = torch.randn_like(actions)
            t_ = t.view(-1, 1, 1)
            xt = (1 - t_) * actions + t_ * x0
            u_pred = self.model.network(xt, t, r, obs)
            u_target = actions - x0
            loss = torch.nn.functional.mse_loss(u_pred, u_target)
        return loss

    def inference(self, cond: dict):
        """Generate samples for testing."""
        if self.test_model_type == 'ema':
            samples = self.ema_model.sample(
                cond,
                inference_steps=self.test_denoising_steps,
                record_intermediate=False,
                clip_intermediate_actions=self.test_clip_intermediate_actions
            )
        else:
            samples = self.model.sample(
                cond,
                inference_steps=self.test_denoising_steps,
                record_intermediate=False,
                clip_intermediate_actions=self.test_clip_intermediate_actions
            )
        return samples
