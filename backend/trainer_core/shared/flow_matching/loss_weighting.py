# -*- coding: utf-8 -*-
"""
Flow Matching Loss Weighting — Model-agnostic

Provides timestep-dependent loss weighting schemes and the basic
flow matching MSE loss. Applicable to any flow matching model.
"""

import math

import torch
import torch.nn.functional as F


def compute_loss_weighting(
    weighting_scheme: str,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Compute loss weighting based on timesteps.

    Args:
        weighting_scheme: One of "none", "sigma_sqrt", "cosmap"
        timesteps: Timesteps in [0, 1] range (B,)

    Returns:
        Loss weights (B,)
    """
    if weighting_scheme == "sigma_sqrt":
        return torch.sqrt(timesteps)
    elif weighting_scheme == "cosmap":
        return 1.0 - torch.cos(timesteps * math.pi / 2)
    else:
        # "none" or unknown → uniform weighting
        return torch.ones_like(timesteps)


def flow_matching_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute standard flow matching MSE loss.

    Standard flow matching:
        target = velocity = noise - z_0

    Z-Image variant:
        target = -z_t (with Norm_opt scaling)
    """
    return F.mse_loss(model_pred, target, reduction=reduction)
