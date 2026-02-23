# -*- coding: utf-8 -*-
"""
Z-Image Forward Pass

Model-specific forward pass handling for ZImageTransformer2DModel.

Supports two modes:
1. text2img: Standard text-to-image training
   - x: List[Tensor(C,1,H,W)] — one per batch item
   - cap_feats: List[Tensor(SeqLen, Dim)] — variable-length text embeddings

2. omni: Multi-condition training with SigLIP
   - x: List[List[Tensor(C,1,H,W)]] — [cond₁, cond₂, ..., target] per batch item
   - cap_feats: List[Tensor(SeqLen, Dim)]
   - siglip_feats: List[List[Tensor(Hs*Ws, Cs)]]
   - image_noise_mask: List[List[int]] — 0=clean condition, 1=noisy target

Note: ZImageTransformer2DModel internally negates output for v-prediction,
so we apply -output here to match the training target convention:
  target_velocity = noise - latents (from ACRFTrainer.sample_batch)
"""

import torch
from typing import List, Optional


def forward_text2img(
    transformer,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    cap_feats: List[torch.Tensor],
) -> torch.Tensor:
    """
    Text2Img forward pass.

    Args:
        transformer: ZImageTransformer2DModel
        noisy_latents: (B, C, H, W) noisy latent images
        timesteps: (B,) normalized timesteps in [0, 1] range
                   where 0 ≈ pure noise, 1 ≈ clean image
                   Formula: t = 1 - sigma = (1000 - acrf_timesteps) / 1000
        cap_feats: List of (SeqLen_i, Dim) text embeddings, one per batch item

    Returns:
        v_pred: (B, C, H, W) predicted velocity (negated for training target)
    """
    B = noisy_latents.shape[0]

    # Convert (B, C, H, W) → List[Tensor(C, 1, H, W)]
    x_list = [noisy_latents[i].unsqueeze(1) for i in range(B)]

    model_out_list = transformer(
        x=x_list,
        t=timesteps,
        cap_feats=cap_feats,
        return_dict=False,
    )[0]

    # Stack outputs and squeeze frame dim: List[(C,1,H,W)] → (B, C, H, W)
    # Negate for v-prediction training target
    return -torch.stack(model_out_list, dim=0).squeeze(2)


def forward_omni(
    transformer,
    noisy_target: torch.Tensor,
    cond_latents: List[torch.Tensor],
    timesteps: torch.Tensor,
    cap_feats: List[torch.Tensor],
    siglip_feats: List[List[torch.Tensor]],
    image_noise_mask: List[List[int]],
) -> torch.Tensor:
    """
    Omni forward pass with multi-image conditioning.

    In omni mode, the transformer processes condition images (clean)
    together with the target image (noisy) in a single forward pass.
    The noise mask tells the transformer which images are clean vs noisy.

    Args:
        transformer: ZImageTransformer2DModel
        noisy_target: (B, C, H, W) noisy target latent
        cond_latents: List of (B, C, H, W) condition image latents (clean, no noise)
                      Length = number of condition images
        timesteps: (B,) normalized timesteps in [0, 1]
        cap_feats: List of (SeqLen_i, Dim) text embeddings per batch item
        siglip_feats: List (batch) of List (per-condition) of Tensor(Hs*Ws, Cs)
                      siglip_feats[b][c] = SigLIP embedding for condition c in batch b
        image_noise_mask: List (batch) of List[int]
                          [0, 0, ..., 1] — 0=clean condition, 1=noisy target

    Returns:
        v_pred: (B, C, H, W) predicted velocity for the TARGET image only
    """
    B = noisy_target.shape[0]
    num_conds = len(cond_latents)

    # Build x_combined: List[List[Tensor(C,1,H,W)]]
    # Each batch item = [cond_0, cond_1, ..., target]
    x_combined = []
    for i in range(B):
        item = [cond_latents[c][i].unsqueeze(1) for c in range(num_conds)]
        item.append(noisy_target[i].unsqueeze(1))
        x_combined.append(item)

    model_out_list = transformer(
        x=x_combined,
        t=timesteps,
        cap_feats=cap_feats,
        siglip_feats=siglip_feats,
        image_noise_mask=image_noise_mask,
        return_dict=False,
    )[0]

    # Extract only the target output (last element per batch item)
    # Condition outputs are discarded during training
    target_outputs = [out[-1] for out in model_out_list]

    # Stack and squeeze frame dim, negate for v-prediction
    return -torch.stack(target_outputs, dim=0).squeeze(2)


__all__ = [
    "forward_text2img",
    "forward_omni",
]
