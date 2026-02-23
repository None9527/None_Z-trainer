# -*- coding: utf-8 -*-
"""
Gradient Utilities - Model-agnostic

Provides gradient clipping (norm and value based).
Consolidates the duplicated implementations from:
- flow_matching.clip_gradients
- training_utils.GradientClipper
- training_utils.clip_grad_norm

into a single canonical location.
"""

import torch
from torch import nn
from typing import Union, Iterable


def clip_grad_norm(
    parameters: Union[Iterable[nn.Parameter], torch.Tensor],
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradients by total norm.

    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm. Set 0 to disable.
        norm_type: Type of norm (default: L2)

    Returns:
        Original gradient norm before clipping
    """
    if max_norm <= 0:
        # Disabled - just compute and return current norm
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
            norm_type,
        )
        return total_norm.item()

    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type).item()


def clip_grad_value(
    parameters: Union[Iterable[nn.Parameter], torch.Tensor],
    clip_value: float = 1.0,
) -> None:
    """
    Clip gradients by value.

    Args:
        parameters: Model parameters
        clip_value: Maximum absolute value for gradients
    """
    torch.nn.utils.clip_grad_value_(parameters, clip_value)
