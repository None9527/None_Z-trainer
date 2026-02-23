# -*- coding: utf-8 -*-
"""
[MODEL_NAME] Sampling Strategy

Model-specific timestep sampling.
Different models use different sampling distributions and shift values.
"""

import torch


def sample_timesteps(batch_size: int, device: torch.device, config: dict) -> torch.Tensor:
    """
    Sample timesteps for training.

    Args:
        batch_size: Number of samples
        device: Target device
        config: Sampling config (method, shift, etc.)

    Returns:
        Timesteps tensor (B,)
    """
    raise NotImplementedError("Implement timestep sampling for your model")
