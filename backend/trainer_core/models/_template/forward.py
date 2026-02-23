# -*- coding: utf-8 -*-
"""
[MODEL_NAME] Forward Pass

Model-specific forward pass:
- Convert standard latents to model-specific input format
- Call model
- Convert output to standard (pred, target) format for shared losses
"""

import torch


def forward(model, noisy_latents, timesteps, text_embed, **kwargs) -> torch.Tensor:
    """
    Model-specific forward pass.

    Args:
        model: The transformer/unet model
        noisy_latents: Noisy input (format depends on model)
        timesteps: Timesteps
        text_embed: Text encoder output
        **kwargs: Model-specific extra args

    Returns:
        Model prediction tensor
    """
    raise NotImplementedError("Implement forward pass for your model")
