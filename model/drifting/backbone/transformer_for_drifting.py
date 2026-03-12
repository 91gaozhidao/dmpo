"""Transformer backbone designed for Drifting Policy (one-step generation).

Key differences from ``TransformerForDiffusion``:

* **No time/timestep embedding** – Drifting is single-step (NFE=1) and does
  not iterate over diffusion steps.  Removing the time token saves parameters,
  reduces sequence length, and avoids passing dummy ``timestep=None``.
* **Small random output head** – uses the default init (std=0.02) so that the
  initial generator produces diverse outputs.  This is critical for the drifting
  kernel: identical outputs (e.g. all zeros from zero-init) make inter-sample
  distances zero, causing the exp kernel to degenerate and V≈0.
* **Clean forward signature** – ``forward(sample, cond=None)`` with no timestep
  argument, making call sites clearer.
"""

from typing import Optional, Tuple
import logging
import torch
import torch.nn as nn
from model.diffusion.modules import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class TransformerForDrifting(nn.Module):
    """Transformer backbone for Drifting Policy.

    Supports two modes:
    * **Encoder-decoder** (``obs_as_cond=True``): observation tokens are
      processed by an encoder; the decoder cross-attends to the encoder
      output while processing the noisy action sequence.
    * **Encoder-only / BERT** (``obs_as_cond=False``): only the noisy action
      sequence is processed by a transformer encoder.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon  # action sequence length
        # Derive obs_as_cond from cond_dim (matching TransformerForDiffusion convention)
        obs_as_cond = cond_dim > 0

        # ── input embedding ──
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # ── condition encoder (observations) ──
        self.cond_obs_emb = None
        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        if obs_as_cond:
            T_cond = n_obs_steps
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers,
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb),
                )

            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer,
            )
        else:
            # encoder-only (BERT)
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer,
            )

        # ── causal attention mask ──
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if obs_as_cond:
                T_cond = n_obs_steps
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(T_cond),
                    indexing='ij',
                )
                # No +1 offset (no time token occupying slot 0 in cond)
                mem_mask = t >= s
                mem_mask = mem_mask.float().masked_fill(mem_mask == 0, float('-inf')).masked_fill(mem_mask == 1, float(0.0))
                self.register_buffer('memory_mask', mem_mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # ── output head ──
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # ── bookkeeping ──
        self.T = T
        self.horizon = horizon
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only
        self.n_emb = n_emb

        # ── weight initialisation ──
        self.apply(self._init_weights)
        # Use small random init for the output head (std=0.02 from _init_weights).
        # Zero-init was causing all model outputs to be identical at startup,
        # making the drifting kernel completely degenerate (dist_neg=0 → V≈0 →
        # no learning signal).  Small random init gives diverse outputs from
        # step 1, enabling the kernel to discriminate and produce useful V.

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    # ------------------------------------------------------------------ init
    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDrifting):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    # --------------------------------------------------------- optimiser API
    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Separate parameters into weight-decay / no-decay groups."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % str(inter_params)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params))

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)],
             "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)],
             "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    # ------------------------------------------------------------- forward
    def forward(
        self,
        sample: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            sample: (B, T, input_dim) — noisy action sequence.
            cond:   (B, T_cond, cond_dim) — observation condition
                    (required when obs_as_cond=True).
        Returns:
            (B, T, output_dim)
        """
        # input embedding
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT-style: just process input tokens
            t = input_emb.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(input_emb + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
        else:
            # Encoder-decoder
            cond_emb = self.cond_obs_emb(cond)
            tc = cond_emb.shape[1]
            cond_pos = self.cond_pos_emb[:, :tc, :]
            memory = self.drop(cond_emb + cond_pos)
            memory = self.encoder(memory)

            t = input_emb.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(input_emb + position_embeddings)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
            )

        # output head
        x = self.ln_f(x)
        x = self.head(x)
        return x
