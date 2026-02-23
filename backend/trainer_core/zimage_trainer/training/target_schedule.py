# -*- coding: utf-8 -*-
"""
Target Schedule — Z-Image Specific

Compute time-dependent training targets for Z-Image's
content/quality mode blending strategy.
"""

from typing import Tuple

import torch


def compute_target_with_schedule(
    latents: torch.Tensor,
    z_t: torch.Tensor,
    timesteps: torch.Tensor,
    schedule_mode: str = "content",
) -> Tuple[torch.Tensor, float]:
    """
    Compute target using time-dependent schedule.

    Args:
        latents: Clean latents z_0
        z_t: Noisy latents
        timesteps: Timesteps in Z-Image format (1 - sigma)
        schedule_mode: "content" or "quality"

    Returns:
        target: Training target
        alpha_t: Schedule factor
    """
    if schedule_mode == "quality":
        # Quality mode: pure proxy target
        target = -z_t
        alpha_t = 1.0
    else:
        # Content mode: time-dependent blending
        alpha_t = timesteps.mean().item()
        T_content = -latents  # -z_0
        T_proxy = -z_t        # -z_t
        target = alpha_t * T_content + (1.0 - alpha_t) * T_proxy

    return target, alpha_t
