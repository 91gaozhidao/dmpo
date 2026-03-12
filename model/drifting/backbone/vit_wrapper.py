import torch
import torch.nn as nn
from typing import Dict, Any
import einops

class DriftingViTWrapper(nn.Module):
    """
    Vision wrapper for Drifting Policy backbones.
    
    Extracts image features using a ViT backbone, compresses them,
    concatenates with low-dim state, and passes the result as a flat 
    condition representation to the core sequence generation backbone.
    """
    def __init__(
        self,
        core_network: nn.Module,
        vision_encoder: nn.Module,
        cond_dim: int,
        spatial_emb: int = 128,
        visual_feature_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.core_network = core_network
        self.vision_encoder = vision_encoder
        self.cond_dim = cond_dim
        self.use_spatial_emb = spatial_emb > 0
        
        # Visual feature processing
        if self.use_spatial_emb:
            from model.common.modules import SpatialEmb
            self.compress = SpatialEmb(
                num_patch=self.vision_encoder.num_patch,
                patch_dim=self.vision_encoder.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.visual_feature_dim = spatial_emb
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.vision_encoder.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
            self.visual_feature_dim = visual_feature_dim
            
        self.cond_enc_dim = self.visual_feature_dim + self.cond_dim

    def forward_encoder(self, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode image and state into a single condition vector."""
        if "rgb" not in cond:
            return cond["state"].view(cond["state"].shape[0], -1)
            
        img = cond["rgb"]
        B, T, C, H, W = img.shape
        state = cond["state"].view(B, -1)

        # Match the pretraining/eval image pipelines: concatenate the image
        # history along channels before the ViT.
        img = einops.rearrange(img, "b t c h w -> b (t c) h w").float()
        feat = self.vision_encoder(img)

        if self.use_spatial_emb:
            visual_feat = self.compress.forward(feat, state)
        else:
            visual_feat = self.compress(feat.flatten(1, -1))

        return torch.cat([visual_feat, state], dim=-1)

    def forward(self, x: torch.Tensor, cond: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Forward pass. Extracts features and delegates to core network.
        This signature matches what DriftingPolicy._call_network sends to custom wrappers:
        wrapper(x, cond=cond)
        """
        # Extract visual + low-dim features
        cond_emb = self.forward_encoder(cond)  # (B, cond_enc_dim)
        
        # The core network expects 'state' to be these features
        pseudo_cond = {"state": cond_emb}
        
        # We delegate back to DriftingPolicy's knowledge of the core_network type!
        # Wait, the wrapper doesn't know _call_network's logic. Let's replicate the
        # specific backbone dispatch here.
        B = x.shape[0]
        from model.drifting.backbone.transformer_for_drifting import TransformerForDrifting
        from model.drifting.backbone.conditional_unet1d import ConditionalUnet1D
        
        if isinstance(self.core_network, TransformerForDrifting):
            obs = pseudo_cond['state']
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
            return self.core_network(x, cond=obs)
        elif isinstance(self.core_network, ConditionalUnet1D):
            obs_flat = pseudo_cond['state'].view(B, -1)
            return self.core_network(x, global_cond=obs_flat)
        else:
            raise ValueError(f"Unknown core_network in DriftingViTWrapper: {type(self.core_network)}")
