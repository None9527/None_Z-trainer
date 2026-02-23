# -*- coding: utf-8 -*-
"""
Norm_opt Scaling — Z-Image Specific

Compute optimal scaling factor k using Norm_opt to minimize
the normalized loss for Z-Image's proxy target training.

Formula:
    k = E[model_pred²] / E[model_pred · target]
    Loss = MSE(model_pred, target × k) / k²
"""

import torch


def compute_norm_opt_scale(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    min_scale: float = 5.0,
    max_scale: float = 100.0,
) -> float:
    """
    Compute optimal scaling factor k using Norm_opt.

    Args:
        model_pred: Model's prediction tensor
        target: Training target tensor
        min_scale: Minimum allowed scale
        max_scale: Maximum allowed scale

    Returns:
        k: Optimal scaling factor
    """
    with torch.no_grad():
        mp_flat = model_pred.flatten().float()
        tgt_flat = target.flatten().float()

        mp_sq = (mp_flat ** 2).mean()
        mp_tgt = (mp_flat * tgt_flat).mean()

        if mp_tgt.abs() > 1e-8:
            k = (mp_sq / mp_tgt).item()
        else:
            k = 50.0

        k = max(min_scale, min(max_scale, k))

    return k
