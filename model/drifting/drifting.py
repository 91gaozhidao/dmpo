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
Drifting Policy Implementation

Implements the Drifting Policy algorithm which achieves single-step (1 NFE)
inference by front-loading the iterative generation process into the training
phase via a drifting field. This is particularly advantageous for high-frequency
real-time robotic control.

Core algorithm follows the paper "Generative Modeling via Drifting" (Kaiming He):
- Exp kernel: k(x,y) = exp(-||x - y|| / T)
- Bi-directional normalization: sqrt(rowsum × colsum)
- Second-order weighting for balanced positive/negative contributions
- Temperature auto-scaled by mean pairwise distance for robustness
"""

import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from model.drifting.backbone.transformer_for_drifting import TransformerForDrifting
from model.drifting.backbone.conditional_unet1d import ConditionalUnet1D
from model.drifting.backbone.vit_wrapper import DriftingViTWrapper

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


def compute_V(x, y_pos, y_neg, T, mask_self=True):
    """
    Compute the mean-shift Drifting Field V_{p,q} based on positive and negative samples.
    
    In principle, V_{p,q} can be a wide range of vector fields, as long as it satisfies V_{p,p} = 0.
    In this paper, an instantiation inspired by mean-shift is used:
        V_{p,q}(x) := V_p^{+}(x) - V_q^{-}(x),
    where
        V_p^{+}(x) := 1 / Z_p(x) * E_{y^{+} ~ p} [ k(x,y^{+}) (y^{+} - x) ]
        V_q^{-}(x) := 1 / Z_q(x) * E_{y^{-} ~ q} [ k(x,y^{-}) (y^{-} - x) ]
    
    The normalizers are:
        Z_p(x) := E_{y^{+} ~ p} [ k(x,y^{+}) ]
        Z_q(x) := E_{y^{-} ~ q} [ k(x,y^{-}) ]
        
    Substituting into V_{p,q} = V_p^{+} - V_q^{-}, we obtain the compact form:
        V_{p,q}(x) = 1 / (Z_p(x) Z_q(x)) * E_{y^{+} ~ p, y^{-} ~ q} [ k(x,y^{+}) k(x,y^{-}) (y^{+} - y^{-}) ]

    Implementation (batch-normalized Monte Carlo estimate):
    In practice, we approximate the above expectation using mini-batches:
        V(x) := E [ K_B(x,y^{+}) K_B(x,y^{-}) (y^{+} - y^{-}) ]
    where K_B is a batch-normalized kernel: the kernel k is normalized over samples in batch B.
    This construction guarantees V_{p,p} = 0: when p=q, the term (y^{+} - y^{-}) is anti-symmetric
    and cancels out in expectation. The resulting field can be efficiently estimated via Monte Carlo 
    over mini-batches.

    Follows Kaiming He's demo implementation exactly:
    - Exp kernel: k(x,y) = exp(-||x - y|| / T)
    - K_B implements normalization across both x and y dimensions (bi-directional normalization: sqrt(rowsum × colsum)),
      which slightly improves performance.
    - Second-order weighting for balanced positive/negative contributions.

    Args:
        x: [N, D] generated samples (x in formulas)
        y_pos: [N_pos, D] real/positive samples (y^{+} from data distribution p)
        y_neg: [N_neg, D] negative samples (y^{-} from generated distribution q, often same as x)
        T: temperature (tau) for the exp kernel
        mask_self: if True, add diagonal mask to dist_neg to ignore self-matches
                   (required when y_neg is x itself; should be False when y_neg
                   is an independent set like shuffled positives)
                   
    Returns:
        V: [N, D] estimated drift vectors for each sample in x
        dist_pos: [N, N_pos] distance matrix to positives
        dist_neg: [N, N_neg] distance matrix to negatives
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]

    # Compute pairwise Euclidean distance matrix
    # ||x - y^{+}|| and ||x - y^{-}||
    dist_pos = torch.cdist(x, y_pos)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N, N_neg]

    # Ignore self (only when y_neg is x itself)
    # Mask self-matches to avoid artificially high self-similarity
    if mask_self:
        dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

    # Compute logits and concat for joint normalization
    # Conceptually, treats targets as a [N, N_neg + N_pos] pool of candidates
    dist = torch.cat([dist_neg, dist_pos], dim=1)  # [N, N_neg + N_pos]

    # ── Exp kernel (paper original, NOT softmax) ──
    # Evaluate k(x, y) = exp(-||x - y|| / T)
    log_kernel = (-dist / T).clamp(min=-80.0)
    kernel = log_kernel.exp()

    # ── Bi-directional normalization: sqrt(rowsum × colsum) ──
    # Normalize the kernel across both dimensions to yield K_B.
    # Normalizing along both dimensions was found to slightly improve performance.
    row_sum = kernel.sum(dim=-1, keepdim=True)    # sum over both pos and neg targets [N, 1]
    col_sum = kernel.sum(dim=-2, keepdim=True)    # sum over queries [1, N_neg + N_pos]
    normalizer = (row_sum * col_sum).clamp_min(1e-12).sqrt()
    A = kernel / normalizer # A corresponds to K_B(x, y)

    # Split the normalized kernel into parts matching negative and positive samples
    A_neg = A[:, :N_neg]  # K_B(x, y^{-})
    A_pos = A[:, N_neg:]  # K_B(x, y^{+})

    # Construct weights (second-order scaling per paper)
    # W_pos corresponds to K_B(x, y^{+}) * \sum K_B(x, y^{-}) with respect to y^{-}.
    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)
    # W_neg corresponds to K_B(x, y^{-}) * \sum K_B(x, y^{+}) with respect to y^{+}.
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)

    # Calculate expected drifting vectors weighted by the coefficients
    drift_pos = W_pos @ y_pos  # E_{y^{+}}[ K_B(x, y^{+}) K_B(x, y^{-}) y^{+} ] -> [N, D]
    drift_neg = W_neg @ y_neg  # E_{y^{-}}[ K_B(x, y^{+}) K_B(x, y^{-}) y^{-} ] -> [N, D]

    # Compute final V_{p,q}(x) = drift_pos - drift_neg
    # = E [ K_B(x,y^{+}) K_B(x,y^{-}) (y^{+} - y^{-}) ]
    V = drift_pos - drift_neg
    return V, dist_pos, dist_neg


def compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.1], mask_self=True):
    """
    Compute drifting loss following the Kaiming He demo formulation.
    
    Training Objective:
    Given a generator f_theta that maps noise epsilon to samples, the training loss is:
        L = E_epsilon [ || f_theta(epsilon) - stopgrad(f_theta(epsilon) + V_{p,q}(f_theta(epsilon))) ||^2 ]
    
    We move predictions toward their drifted versions. The drifting field V_{p,q}, depending on the 
    data distribution p and the generated distribution q, tells each generated sample where to go.

    Key differences from previous implementation:
    - No S_j feature normalization (not needed for single-scale action space)
    - No lambda_j drift normalization (was making loss ≡ 1.0)
    - Temperature auto-scaled by mean pairwise distance for robustness
    - Loss = MSE(x, sg(x + V)) which naturally decreases as model improves

    Args:
        x: [N, D] generated samples (f_theta(epsilon))
        y_pos: [N_pos, D] data samples (from true distribution p)
        y_neg: [N_neg, D] negative samples (from generated distribution q, typically x itself)
        temperatures: list of base temperatures to compute multi-scale drifts over
        mask_self: if True, mask diagonal in dist_neg (when y_neg is x).
                   Set to False when y_neg is an independent set (e.g.
                   shuffled positives) so no valid pairs are excluded.
                   
    Returns:
        loss: a scalar tensor representing the regression MSE loss
        metrics: a dictionary containing logging metrics
    """
    B, D = x.shape

    with torch.no_grad():
        dist_cross = torch.cdist(x, y_pos)  # [N, N_pos]
        mean_dist = dist_cross.mean()

    metrics = {
        "train/mean_cross_dist": mean_dist.item(),
    }

    V_total = torch.zeros_like(x)
    last_dist_pos = None
    last_dist_neg = None
    
    # Calculate and accumulate the drift field V across various temperatures
    for T in temperatures:
        # Auto-scale temperature by mean pairwise distance so that
        # T=0.1 means "10% of the mean distance" — robust across tasks
        adaptive_T = (T * mean_dist).detach()
        
        # Calculate the drift field V_t given the current adaptive temperature
        V_t, dist_pos, dist_neg = compute_V(x, y_pos, y_neg, adaptive_T,
                                              mask_self=mask_self)
        V_total = V_total + V_t
        last_dist_pos = dist_pos
        last_dist_neg = dist_neg

        # Diagnostic: lambda_j (drift RMS per sqrt-dim)
        with torch.no_grad():
            V_rms_sq = torch.mean(torch.sum(V_t**2, dim=-1)) / D
            lambda_j = torch.sqrt(V_rms_sq + 1e-8)
            metrics[f"train/drifting_lambda_T{T}"] = lambda_j.item()
            metrics[f"train/drift_magnitude_T{T}"] = lambda_j.item()

    # Loss: MSE(x, sg(x + V))
    # This precisely matches the training objective defined in the paper.
    # We move predictions x toward their drifted versions (x + V).
    # Unlike the old lambda-normalized loss (≡ 1.0), this loss decreases
    # as the model improves, giving the optimizer meaningful signal.
    target = (x + V_total).detach()
    loss = F.mse_loss(x, target)

    # Training monitoring metrics
    with torch.no_grad():
        V_norms = torch.norm(V_total, dim=-1)
        metrics["train/V_norm_mean"] = V_norms.mean().item()
        metrics["train/V_norm_max"] = V_norms.max().item()
        metrics["train/V_norm_std"] = V_norms.std().item()

        # raw_mse: actual MSE between generated and real samples
        if x.shape[0] == y_pos.shape[0]:
            metrics["train/raw_mse"] = F.mse_loss(x, y_pos).item()
        # drift_magnitude: RMS of V in data space
        metrics["train/drift_magnitude"] = V_norms.mean().item()

        # Distance statistics
        if B <= 512 and last_dist_pos is not None:
            dist_pos_mean = last_dist_pos.mean().item()
            valid_neg = last_dist_neg[last_dist_neg < 1e6]
            dist_neg_mean = (
                valid_neg.mean().item() if valid_neg.numel() > 0 else 0.0)
            metrics["train/dist_to_pos_mean"] = dist_pos_mean
            metrics["train/dist_to_neg_mean"] = dist_neg_mean
            metrics["train/pos_neg_dist_ratio"] = (
                dist_pos_mean / (dist_neg_mean + 1e-8))

    return loss, metrics


class DriftingPolicy(nn.Module):
    """
    Drifting Policy implementing drifting-field-based single-step generation.
    
    The core idea is to compute a drifting field V from positive (expert) 
    and optional negative samples, then train the network so that its output
    converges toward the drifted target. At inference time, only a single 
    forward pass (1 NFE) is needed.

    Uses the paper-correct algorithm:
    - Exp kernel with bi-directional normalization
    - Second-order weighting for balanced pos/neg contributions
    - Multi-temperature drift field accumulation
    - Temperature auto-scaled by mean pairwise distance
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        # Drifting-specific parameters
        temperatures: list = [0.1],
        mask_self: bool = True,
        # Legacy parameters (accepted but ignored for backward compatibility)
        drift_coef: float = 1.0,
        neg_drift_coef: float = 0.5,
        bandwidth: float = 1.0,
    ):
        """
        Initialize DriftingPolicy.

        Args:
            network: Backbone network for action prediction.
                     Supported: TransformerForDrifting, ConditionalUnet1D,
                     or any nn.Module with compatible forward signature.
            device: Device to run the model on
            horizon_steps: Number of steps in trajectory horizon
            action_dim: Dimension of action space
            act_min: Minimum action value for clipping
            act_max: Maximum action value for clipping
            obs_dim: Dimension of observation space
            max_denoising_steps: Maximum denoising steps (for compatibility)
            seed: Random seed for reproducibility
            temperatures: List of base temperatures for multi-scale drift fields
            mask_self: Whether to mask self-interaction in drift field
            drift_coef: (Legacy, ignored) Coefficient for positive drift field
            neg_drift_coef: (Legacy, ignored) Coefficient for negative drift field
            bandwidth: (Legacy, ignored) Bandwidth parameter for drift field kernel
        """
        super().__init__()

        if int(max_denoising_steps) <= 0:
            raise ValueError('max_denoising_steps must be a positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.horizon_steps, self.action_dim)
        self.act_range = (act_min, act_max)
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)

        # Drifting-specific parameters
        self.temperatures = temperatures
        self.mask_self = mask_self

        # Detect backbone type for dispatch
        self._is_transformer = isinstance(network, TransformerForDrifting)
        self._is_unet1d = isinstance(network, ConditionalUnet1D)
        self._is_vit_wrapper = isinstance(network, DriftingViTWrapper)

    def _call_network(self, x: Tensor, cond: dict) -> Tensor:
        """
        Dispatch forward pass to the backbone network with the correct signature.

        Drifting Policy is 1-NFE, so t and r are always constants (1.0 and 0.0).
        Different backbones have different forward() signatures:
        - TransformerForDrifting: forward(sample, cond=obs_tensor)
        - ConditionalUnet1D: forward(sample, timestep=None, global_cond=obs_flat)
        - MeanFlowMLP (legacy fallback): forward(action, time, r, cond)

        Args:
            x: (B, Ta, Da) input noise or action
            cond: dict with 'state' key for observations

        Returns:
            (B, Ta, Da) predicted action
        """
        B = x.shape[0]

        if self._is_transformer:
            # TransformerForDrifting: forward(sample, cond)
            # cond expects (B, T_cond, cond_dim)
            obs = cond['state']  # (B, To, Do)
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)  # (B, 1, Do)
            return self.network(x, cond=obs)

        elif self._is_unet1d:
            # ConditionalUnet1D: forward(sample, local_cond, global_cond)
            # Drifting has no timestep; pass obs as global_cond
            obs_flat = cond['state'].view(B, -1)  # (B, Do*To)
            return self.network(x, global_cond=obs_flat)

        elif self._is_vit_wrapper:
            # Wrapper handles visual features itself
            return self.network(x, cond=cond)

        else:
            raise ValueError(f"Unsupported backbone type: {type(self.network)}. "
                             f"DriftingPolicy only supports TransformerForDrifting, ConditionalUnet1D, or DriftingViTWrapper.")

    def compute_V_field(
        self, x: Tensor, y_pos: Tensor, y_neg: Tensor, T: float, mask_self: bool = True
    ) -> tuple:
        """
        Compute the drifting field V for a single temperature.

        Wrapper around the module-level compute_V that handles the
        (B, Ta, Da) -> (B, Ta*Da) flattening and unflattening.

        Args:
            x: (B, Ta, Da) generated action predictions
            y_pos: (B, Ta, Da) or (N_pos, Ta*Da) positive (expert) action targets
            y_neg: (B, Ta, Da) or (N_neg, Ta*Da) negative action targets
            T: temperature for the exp kernel
            mask_self: whether to mask self-interaction

        Returns:
            V: (B, Ta, Da) drift field
            dist_pos: distance matrix to positives
            dist_neg: distance matrix to negatives
        """
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        y_pos_flat = y_pos.reshape(y_pos.shape[0], -1)
        y_neg_flat = y_neg.reshape(y_neg.shape[0], -1)

        V, dist_pos, dist_neg = compute_V(x_flat, y_pos_flat, y_neg_flat, T, mask_self=mask_self)
        return V.reshape(B, self.horizon_steps, self.action_dim), dist_pos, dist_neg

    def loss(self, x1: Tensor, cond: dict, y_neg: Tensor = None) -> Tensor:
        """
        Compute Drifting Policy loss for offline pretraining.

        Uses the paper-correct algorithm with:
        - Exp kernel with bi-directional normalization
        - Second-order weighting
        - Multi-temperature drift field accumulation
        - Temperature auto-scaled by mean pairwise distance

        In pure behavior cloning (BC) mode, y_pos is the expert action (x1),
        and y_neg defaults to the generated samples themselves (x).

        Args:
            x1: (B, Ta, Da) expert action trajectories (used as y_pos)
            cond: dict with 'state' key for observations
            y_neg: (B, Ta, Da) or None, negative action samples for regularization

        Returns:
            Scalar loss tensor
        """
        B = x1.shape[0]
        y_pos = torch.clamp(x1, *self.act_range)

        # Generate initial noise as input
        x_gen = torch.randn((B,) + self.data_shape, device=self.device)

        # Network prediction (the current generated action)
        x = self._call_network(x_gen, cond)

        # Flatten to (B, Ta*Da) for drift field computation
        x_flat = x.reshape(B, -1)
        y_pos_flat = y_pos.reshape(B, -1)

        # Use generated samples as y_neg if not provided (standard drifting)
        if y_neg is not None:
            y_neg_flat = y_neg.reshape(y_neg.shape[0], -1)
        else:
            y_neg_flat = x_flat

        # Compute drifting loss using the paper-correct algorithm
        loss, _metrics = compute_drifting_loss(
            x_flat, y_pos_flat, y_neg_flat,
            temperatures=self.temperatures,
            mask_self=self.mask_self,
        )
        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int = 1,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None,
    ) -> Sample:
        """
        1 NFE sampling: only a single forward pass is needed.

        Args:
            cond: Condition dictionary with 'state' key
            inference_steps: Number of inference steps (forced to 1 for Drifting Policy)
            record_intermediate: Whether to record intermediate steps
            clip_intermediate_actions: Whether to clip actions to valid range
            z: Initial noise (if None, sample from Gaussian)

        Returns:
            Sample namedtuple with trajectories and optional chains
        """
        B = cond['state'].shape[0]

        if z is None:
            z = torch.randn((B,) + self.data_shape, device=self.device)

        # Single-step forward pass
        action = self._call_network(z, cond)
        action = action.clamp(*self.act_range)

        # Runtime assertion: verify action boundaries before env interaction
        assert action.min() >= self.act_range[0] and action.max() <= self.act_range[1], (
            f"Action boundary violation after clamp: "
            f"min={action.min().item():.6f}, max={action.max().item():.6f}, "
            f"expected range=[{self.act_range[0]}, {self.act_range[1]}]"
        )

        chains = None
        if record_intermediate:
            chains = action.unsqueeze(0)

        return Sample(trajectories=action, chains=chains)
