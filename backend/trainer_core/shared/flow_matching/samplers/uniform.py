# -*- coding: utf-8 -*-
"""
Uniform Timestep Sampling — Model-agnostic

Supports:
    - Pure uniform sampling in [0, 1]
    - Shift-transformed uniform (Z-Image / FLUX style)
    - Dynamic shift based on image resolution

The shift transform concentrates samples toward higher noise levels,
which is beneficial for models using the flow matching straight-line ODE.
"""

import math
from typing import Optional, Tuple

import torch


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """
    Compute resolution-dependent dynamic shift value.

    Linear interpolation between base_shift and max_shift based on
    the image sequence length. Matches the Pipeline's calculate_shift.

    Args:
        image_seq_len: (latent_h // 2) * (latent_w // 2)
        base_seq_len: Base sequence length (256 → 512×512 input)
        max_seq_len: Max sequence length (4096 → 2048×2048 input)
        base_shift: Shift at base resolution (0.5)
        max_shift: Shift at max resolution (1.15)

    Returns:
        Computed shift value (mu)
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def get_lin_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
):
    """Create linear interpolation function (generalized form)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def sample_uniform(
    batch_size: int,
    shift: float = 0.0,
    dynamic_shift: bool = False,
    latent_shape: Optional[Tuple[int, ...]] = None,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    min_sigma: float = 0.001,
    max_sigma: float = 0.999,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Sample sigmas uniformly in [0, 1], optionally with shift transform.

    Args:
        batch_size: Number of samples
        shift: Fixed shift factor (>0 enables shift transform)
        dynamic_shift: Whether to compute shift from latent_shape
        latent_shape: Latent shape for dynamic shift (B, C, H, W)
        base_seq_len: Dynamic shift base sequence length
        max_seq_len: Dynamic shift max sequence length
        base_shift: Dynamic shift base value
        max_shift: Dynamic shift max value
        min_sigma: Minimum sigma (clamp)
        max_sigma: Maximum sigma (clamp)
        device: Target device
        dtype: Target dtype

    Returns:
        sigmas: Sampled noise levels in [min_sigma, max_sigma] (B,)
    """
    if device is None:
        device = torch.device("cpu")

    sigmas = torch.rand(batch_size, device=device)

    # Compute effective shift
    effective_shift = shift
    if dynamic_shift and latent_shape is not None:
        h, w = latent_shape[-2:]
        image_seq_len = (h // 2) * (w // 2)
        effective_shift = calculate_shift(
            image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift
        )

    # Apply shift transform: sigma' = (sigma * s) / (1 + (s - 1) * sigma)
    if effective_shift > 0:
        sigmas = (sigmas * effective_shift) / (1 + (effective_shift - 1) * sigmas)

    sigmas = sigmas.clamp(min_sigma, max_sigma)

    if dtype is not None:
        sigmas = sigmas.to(dtype=dtype)

    return sigmas
