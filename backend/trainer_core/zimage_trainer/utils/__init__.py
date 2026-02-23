# -*- coding: utf-8 -*-
"""
Z-Image Specific Utilities

Only Z-Image specific utilities remain here.
Generic utilities have been moved to shared/utils/.

Backward compatibility: re-exports from shared/ are provided below.
"""

# Z-Image specific
from .vae_utils import load_vae, decode_latents_to_pixels, encode_pixels_to_latents
from .latent_utils import pack_latents, unpack_latents

# Backward compatibility re-exports from shared/
from shared.utils import (
    get_optimizer,
    get_scheduler,
)
from shared.utils.model_hooks import (
    BlockSwapperHook,
    apply_block_swapper,
    apply_attention_optimization,
    enable_gradient_checkpointing,
    apply_all_optimizations,
)
from shared.gradient import clip_grad_norm
from shared.memory import ModuleOffloader

__all__ = [
    # Z-Image specific
    "load_vae",
    "decode_latents_to_pixels",
    "encode_pixels_to_latents",
    "pack_latents",
    "unpack_latents",
    # Backward compat (deprecated — use shared.utils / shared.gradient / shared.memory directly)
    "get_optimizer",
    "get_scheduler",
    "BlockSwapperHook",
    "apply_block_swapper",
    "apply_attention_optimization",
    "enable_gradient_checkpointing",
    "apply_all_optimizations",
    "clip_grad_norm",
    "ModuleOffloader",
]
