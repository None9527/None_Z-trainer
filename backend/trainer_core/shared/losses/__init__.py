# -*- coding: utf-8 -*-
"""
Shared Loss Functions — Model-agnostic

All loss modules operate on pure tensors (pred_v, target_v, etc.)
and contain no model-specific assumptions. Reusable across all
flow matching and v-prediction based models.

Available losses:
- StandardLoss: Charbonnier + Cosine (recommended default)
- MSELoss: Standard L2/MSE
- CharbonnierLoss: Robust L1
- L1CosineLoss: Charbonnier + Cosine
- L2CosineLoss: L2 + Cosine
- FrequencyAwareLoss: Frequency-domain separated loss (HF L1 + LF Cosine)
- AdaptiveFrequencyLoss: Adaptive frequency loss with warmup
- StyleStructureLoss: SSIM structure lock + Lab style transfer
- LatentStyleStructureLoss: Latent-space approximation (saves VRAM)
- DPOLoss: Direct Preference Optimization
- DPOLossWithSNR: DPO with SNR weighting
"""

from .standard_loss import StandardLoss, compute_standard_loss
from .mse_loss import (
    MSELoss,
    CharbonnierLoss,
    L2CosineLoss,
    L1CosineLoss,
    compute_mse_loss,
    compute_charbonnier_loss,
    compute_cosine_loss,
)
from .frequency_aware_loss import FrequencyAwareLoss, AdaptiveFrequencyLoss
from .style_structure_loss import StyleStructureLoss, LatentStyleStructureLoss
from .dpo_loss import DPOLoss, DPOLossWithSNR

__all__ = [
    # Standard (recommended)
    "StandardLoss",
    "compute_standard_loss",
    # Basic
    "MSELoss",
    "CharbonnierLoss",
    "L2CosineLoss",
    "L1CosineLoss",
    "compute_mse_loss",
    "compute_charbonnier_loss",
    "compute_cosine_loss",
    # Frequency-domain
    "FrequencyAwareLoss",
    "AdaptiveFrequencyLoss",
    # Style-structure
    "StyleStructureLoss",
    "LatentStyleStructureLoss",
    # DPO
    "DPOLoss",
    "DPOLossWithSNR",
]
