# -*- coding: utf-8 -*-
"""
Flow Matching Noising — Model-agnostic

Core linear interpolation formula for flow matching:
    z_t = (1 - t) * z_0 + t * noise

Provides:
    - get_z_t: Basic noisy latent creation
    - get_noisy_model_input: z_t + timestep normalization
    - compute_velocity_target: v = noise - z_0
"""

from typing import Tuple

import torch


def get_z_t(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """
    Create noisy latent z_t via linear interpolation.

    Formula: z_t = (1 - sigma) * z_0 + sigma * noise

    Args:
        latents: Clean latents z_0 (B, C, H, W) or (B, C, F, H, W)
        noise: Random noise (same shape as latents)
        sigmas: Noise levels in [0, 1] range (B,)

    Returns:
        z_t: Noisy latent
    """
    if latents.dim() == 4:
        s = sigmas.view(-1, 1, 1, 1)
    elif latents.dim() == 5:
        s = sigmas.view(-1, 1, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported latent dim: {latents.dim()}")

    return (1.0 - s) * latents + s * noise


def get_noisy_model_input(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create noisy model input and normalized timesteps.

    Args:
        latents: Clean latents z_0
        noise: Random noise
        sigmas: Noise levels in [0, 1] range (B,)

    Returns:
        z_t: Noisy latent
        timesteps_normalized: (1 - sigma), for models using t_scale=1000
    """
    z_t = get_z_t(latents, noise, sigmas)
    timesteps_normalized = 1.0 - sigmas
    return z_t, timesteps_normalized


def compute_velocity_target(
    latents: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Compute velocity target for rectified flow.

    Formula: v = noise - z_0

    This is the straight-line ODE: dx/dt = v(x, t)
    """
    return noise - latents
