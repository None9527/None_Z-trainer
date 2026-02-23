# -*- coding: utf-8 -*-
"""
SNR (Signal-to-Noise Ratio) Weighting Utilities — Model-agnostic

Provides Min-SNR gamma weighting and Floored Min-SNR for flow matching training.
Pure math on timesteps, no model-specific logic.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_snr(
    timesteps: torch.Tensor,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """
    Compute SNR (Signal-to-Noise Ratio).

    For Rectified Flow:
        sigma = timestep / 1000
        SNR = ((1 - sigma) / sigma)^2

    Args:
        timesteps: Timestep values (B,)
        num_train_timesteps: Total training timesteps

    Returns:
        snr: SNR values (B,)
    """
    sigmas = timesteps.float() / num_train_timesteps
    sigmas = sigmas.clamp(min=0.001, max=0.999)
    snr = ((1 - sigmas) / sigmas) ** 2
    return snr


def compute_snr_weights(
    timesteps: torch.Tensor,
    num_train_timesteps: int = 1000,
    snr_gamma: float = 5.0,
    snr_floor: float = 0.1,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """
    Compute Floored Min-SNR weights.

    Args:
        timesteps: Timestep values (B,)
        num_train_timesteps: Total training timesteps
        snr_gamma: SNR clamp value (recommended 5.0)
        snr_floor: Floor weight (recommended 0.1)
        prediction_type: "v_prediction" or "epsilon"

    Returns:
        weights: Weight coefficients (B, 1, 1, 1)
    """
    if snr_gamma <= 0:
        return torch.ones(timesteps.shape[0], 1, 1, 1, device=timesteps.device, dtype=torch.float32)

    snr = compute_snr(timesteps, num_train_timesteps)
    clipped_snr = torch.clamp(snr, max=snr_gamma)

    if prediction_type == "v_prediction":
        weights = clipped_snr / (snr + 1)
    else:
        weights = clipped_snr / snr.clamp(min=0.001)

    if snr_floor > 0:
        weights = torch.maximum(weights, torch.tensor(snr_floor, device=weights.device))

    weights = weights.view(-1, 1, 1, 1)
    return weights


def print_anchor_snr_weights(
    turbo_steps: int = 10,
    shift: float = 3.0,
    snr_gamma: float = 5.0,
    snr_floor: float = 0.1,
):
    """Print anchor SNR weight distribution (for debugging)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=shift,
    )
    scheduler.set_timesteps(num_inference_steps=turbo_steps, device="cpu")

    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas[:-1]

    logger.info(f"\n{'='*60}")
    logger.info(f"Anchor SNR weights (steps={turbo_steps}, shift={shift}, gamma={snr_gamma}, floor={snr_floor})")
    logger.info(f"{'='*60}")

    for i, (t, s) in enumerate(zip(timesteps, sigmas)):
        snr = ((1 - s) / s) ** 2
        std_weight = min(snr.item(), snr_gamma) / (snr.item() + 1)
        floored_weight = max(std_weight, snr_floor)
        logger.info(f"  Anchor {i}: t={t.item():.1f}, σ={s.item():.4f}, SNR={snr.item():.4f}, w={floored_weight:.4f}")


__all__ = [
    "compute_snr",
    "compute_snr_weights",
    "print_anchor_snr_weights",
]
