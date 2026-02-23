# -*- coding: utf-8 -*-
"""
Z-Image Trainer — LoRA / ControlNet Training for Z-Image

Z-Image specific components. Generic flow matching utilities
are in shared/flow_matching/ and shared/losses/.

Usage:
    from zimage_trainer import LoRANetwork
    from shared.flow_matching import get_z_t, get_noisy_model_input
    from shared.flow_matching.samplers import create_sampler
    from shared.losses import StandardLoss, FrequencyAwareLoss
"""

__version__ = "0.4.0"

# Models
from .utils.zimage_utils import (
    load_transformer as load_zimage_model,
    load_vae,
    load_text_encoder_and_tokenizer,
    load_scheduler,
    create_pipeline_from_components,
)

# Networks
from .networks import LoRANetwork, create_network

# Inference
from .inference import ZImagePipeline

# Z-Image specific training components
from .training import compute_norm_opt_scale, compute_target_with_schedule

# ---- Backward compatibility re-exports (from shared/) ----
# These are deprecated; import directly from shared/flow_matching/ instead.
from shared.flow_matching import (
    get_z_t,
    get_noisy_model_input,
    compute_loss_weighting,
)
from shared.flow_matching.samplers import create_sampler

__all__ = [
    # Version
    "__version__",
    # Models
    "load_zimage_model",
    "load_vae",
    "load_text_encoder_and_tokenizer",
    "load_scheduler",
    "create_pipeline_from_components",
    # Networks
    "LoRANetwork",
    "create_network",
    # Inference
    "ZImagePipeline",
    # Z-Image Training
    "compute_norm_opt_scale",
    "compute_target_with_schedule",
    # Backward compat (deprecated)
    "get_z_t",
    "get_noisy_model_input",
    "compute_loss_weighting",
    "create_sampler",
]
