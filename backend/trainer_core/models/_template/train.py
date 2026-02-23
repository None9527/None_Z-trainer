# -*- coding: utf-8 -*-
"""
[MODEL_NAME] Training Loop

Assembles shared components (losses, optimizers) with model-specific
components (sampler, forward) into a complete training step.

This is the assembly point, NOT an abstraction layer.
Copy from zimage/train.py and modify as needed.
"""


def train_step(model, latents, noise, text_embed, config):
    """
    One training step.

    Assembly pattern:
        1. Sample timesteps (model-specific sampler)
        2. Create noisy input (model-specific)
        3. Forward pass (model-specific)
        4. Compute loss (shared losses)
        5. Return loss + components dict

    Args:
        model: The model being trained
        latents: Clean latents
        noise: Random noise
        text_embed: Text embeddings
        config: Training configuration

    Returns:
        Tuple of (loss, loss_components_dict)
    """
    raise NotImplementedError("Implement training step for your model")
