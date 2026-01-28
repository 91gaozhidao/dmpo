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
Pre-training Consistency Model policy via distillation from Reflow teacher.

Implements Consistency Distillation (CD) where a pre-trained Reflow model
serves as the teacher to guide the consistency model training.
"""

import logging
import torch
import hydra
log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.flow.consistency import ConsistencyModel


class TrainConsistencyAgent(PreTrainAgent):
    """Training agent for Consistency Model policies with Reflow teacher distillation."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: ConsistencyModel
        self.ema_model: ConsistencyModel

        self.verbose_train = False
        self.verbose_loss = False
        self.verbose_test = True

        # Load teacher model for distillation
        self.teacher_path = cfg.get('teacher_path', None)
        if self.teacher_path is None:
            raise ValueError("teacher_path must be specified for Consistency Distillation!")

        self._load_teacher(cfg)

        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test = False
            # Consistency models can work with 1 step
            self.test_denoising_steps = cfg.get('test_denoising_steps', 1)
            self.test_clip_intermediate_actions = True
            self.test_model_type = 'ema'

    def _load_teacher(self, cfg):
        """Load pre-trained Reflow teacher model."""
        log.info(f"Loading teacher model from {self.teacher_path}")

        # Load teacher checkpoint
        teacher_checkpoint = torch.load(self.teacher_path, map_location=self.device)

        # Create teacher model with same architecture
        teacher_model = hydra.utils.instantiate(cfg.teacher_model)

        # Load weights (support both 'model' and 'ema' keys)
        if 'ema' in teacher_checkpoint:
            teacher_model.load_state_dict(teacher_checkpoint['ema'])
            log.info("Loaded teacher from EMA weights")
        elif 'model' in teacher_checkpoint:
            teacher_model.load_state_dict(teacher_checkpoint['model'])
            log.info("Loaded teacher from model weights")
        else:
            teacher_model.load_state_dict(teacher_checkpoint)
            log.info("Loaded teacher from raw checkpoint")

        # Set teacher in consistency model
        self.model.set_teacher(teacher_model)
        self.ema_model.set_teacher(teacher_model)

        log.info("Teacher model loaded successfully for consistency distillation")

    def get_loss(self, batch_data):
        """Compute Consistency Distillation loss."""
        # batch_data = (actions, cond) from StitchedSequenceDataset
        actions, cond = batch_data
        loss = self.model.loss(actions, cond)
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
