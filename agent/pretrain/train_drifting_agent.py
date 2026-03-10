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
Pre-training Drifting Policy agent.

Drifting Policy uses a drifting field to train a single-step (1 NFE) generative
model. At inference, only one forward pass is needed.
"""

import logging
import torch
log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.drifting.drifting import DriftingPolicy


class TrainDriftingAgent(PreTrainAgent):
    """Training agent for Drifting Policy following the PreTrainAgent structure."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: DriftingPolicy
        self.ema_model: DriftingPolicy

        self.verbose_train = False
        self.verbose_loss = False
        self.verbose_test = True

        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test = False

            # Drifting Policy uses 1 NFE by design
            self.test_denoising_steps = 1

            self.test_clip_intermediate_actions = True
            self.test_model_type = 'ema'

    def get_loss(self, batch_data):
        """Compute Drifting Policy loss for training and validation."""
        # batch_data = (actions, observation) from StitchedSequenceDataset
        actions, obs = batch_data
        cond = {"state": obs}
        loss = self.model.loss(x1=actions, cond=cond)
        return loss

    def inference(self, cond: dict):
        """Generate samples for testing using 1 NFE."""
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
