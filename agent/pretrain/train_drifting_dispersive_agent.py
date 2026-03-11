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
Training agent for Drifting Policy with Dispersive Loss regularization.

Dispersive loss adds a diversity-promoting regularization term to the
standard drifting field loss, preventing mode collapse during pre-training.
"""

import logging
import wandb
from agent.pretrain.train_drifting_agent import TrainDriftingAgent

log = logging.getLogger(__name__)


class TrainDriftingDispersiveAgent(TrainDriftingAgent):
    """Training agent for Drifting Policy with Dispersive Loss regularization."""

    def __init__(self, cfg):
        super().__init__(cfg)
        log.info("Initialized Drifting Policy Dispersive training agent")

        # Log dispersive loss configuration if available
        if hasattr(self.model, 'get_dispersive_loss_info'):
            dispersive_info = self.model.get_dispersive_loss_info()
            log.info(f"Dispersive loss configuration: {dispersive_info}")

            # Log to wandb if available
            if wandb.run is not None:
                wandb.config.update({f"dispersive_{k}": v for k, v in dispersive_info.items()})

    def get_loss(self, batch_data):
        """
        Compute Drifting Policy loss with dispersive regularization.

        The model's loss method already includes the dispersive loss term
        when the model is configured with dispersive regularization.
        """
        try:
            actions, obs = batch_data
            cond = obs
            loss = self.model.loss(x1=actions, cond=cond)
            return loss
        except Exception as e:
            log.error(f"Error computing drifting dispersive loss: {e}")
            # Fallback to standard drifting loss computation
            return super().get_loss(batch_data)
