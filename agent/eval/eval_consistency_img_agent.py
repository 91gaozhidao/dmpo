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
Evaluate pre-trained/fine-tuned Consistency Model policy.
"""
import logging
import torch
log = logging.getLogger(__name__)
from agent.eval.eval_agent_img_base import EvalImgAgent
from model.flow.consistency import ConsistencyModel
from util.timer import Timer


class EvalImgConsistencyAgent(EvalImgAgent):
    """Evaluation agent for Consistency Model policies."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ################################################      overload        #########################################################
        self.load_ema = cfg.get('load_ema', True)  # Usually True for consistency models
        self.clip_intermediate_actions = True
        # Consistency models can work with 1 step, but we test multiple steps
        self.denoising_steps = cfg.get("denoising_step_list", [1, 2, 3, 4, 5])
        self.plot_scale = 'standard'
        self.render_onscreen = False
        self.record_video = False
        self.record_env_index = 0
        self.denoising_steps_trained = self.model.max_denoising_steps
        ####################################################################################
        log.info(f"Consistency Evaluation: load_ema={self.load_ema}, clip_intermediate_actions={self.clip_intermediate_actions}")

    def infer(self, cond: dict, num_denoising_steps: int):
        """
        Perform inference using the Consistency Model.

        Args:
            cond: Condition dictionary with 'state' and optionally 'rgb'
            num_denoising_steps: Number of denoising steps (1 for single-step generation)

        Returns:
            samples: Sample namedtuple with trajectories
            duration: Time taken for inference
        """
        self.model: ConsistencyModel
        timer = Timer()
        samples = self.model.sample(
            cond=cond,
            inference_steps=num_denoising_steps,
            record_intermediate=False,
            clip_intermediate_actions=self.clip_intermediate_actions
        )
        duration = timer()

        # Debug: Check if actions contain NaN or extreme values
        actions = samples.trajectories
        if torch.isnan(actions).any():
            log.warning("Consistency Model output contains NaN values!")
        if torch.isinf(actions).any():
            log.warning("Consistency Model output contains Inf values!")

        action_min, action_max = actions.min().item(), actions.max().item()
        action_mean = actions.mean().item()
        log.info(f"Consistency action stats: min={action_min:.3f}, max={action_max:.3f}, mean={action_mean:.3f}")

        # samples.trajectories: (n_envs, self.horizon_steps, self.action_dim)
        return samples, duration
