# -*- coding: utf-8 -*-
"""
Logit-Normal Timestep Sampling — Model-agnostic

From SD3 paper: https://arxiv.org/abs/2403.03206v1

Samples timesteps from the logit-normal distribution, which concentrates
samples around the middle of the noise schedule. This is beneficial for
models that benefit from balanced training across noise levels.

Also supports:
    - Mode sampling (cosine-weighted toward middle)
    - Sigmoid sampling (for legacy compatibility)
"""

import math

import torch


def sample_logit_normal(
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    min_sigma: float = 0.001,
    max_sigma: float = 0.999,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Sample sigmas from the logit-normal distribution.

    sigma = sigmoid(N(logit_mean, logit_std^2))

    Args:
        batch_size: Number of samples
        logit_mean: Mean of the normal in logit space
        logit_std: Std of the normal in logit space
        min_sigma: Minimum sigma
        max_sigma: Maximum sigma
        device: Target device
        dtype: Target dtype

    Returns:
        sigmas: Sampled noise levels (B,)
    """
    if device is None:
        device = torch.device("cpu")

    u = torch.normal(
        mean=logit_mean, std=logit_std,
        size=(batch_size,), device=device,
    )
    sigmas = torch.sigmoid(u)
    sigmas = sigmas.clamp(min_sigma, max_sigma)

    if dtype is not None:
        sigmas = sigmas.to(dtype=dtype)

    return sigmas


def sample_mode(
    batch_size: int,
    mode_scale: float = 1.29,
    min_sigma: float = 0.001,
    max_sigma: float = 0.999,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Sample sigmas using mode-weighted distribution.

    Concentrates samples near the mode of the noise schedule.
    """
    if device is None:
        device = torch.device("cpu")

    u = torch.rand(batch_size, device=device)
    sigmas = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    sigmas = sigmas.clamp(min_sigma, max_sigma)

    if dtype is not None:
        sigmas = sigmas.to(dtype=dtype)

    return sigmas


def sample_logsnr(
    batch_size: int,
    min_sigma: float = 0.001,
    max_sigma: float = 0.999,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Sample sigmas from Log-SNR distribution.

    From: https://arxiv.org/abs/2411.14793v3
    """
    if device is None:
        device = torch.device("cpu")

    logsnr = torch.randn(batch_size, device=device) * 1.0
    sigmas = torch.sigmoid(-logsnr / 2)
    sigmas = sigmas.clamp(min_sigma, max_sigma)

    if dtype is not None:
        sigmas = sigmas.to(dtype=dtype)

    return sigmas
