# -*- coding: utf-8 -*-
"""
Timestep Sampling Strategies — Factory + Registry

Supported modes:
    - "uniform": Uniform sampling with optional shift transform
    - "logit_normal": LogitNormal distribution (SD3 paper)
    - "acrf" / "anchor": Turbo anchor-based discrete sampling
    - "mode": Mode-weighted sampling
    - "logsnr": Log-SNR sampling
"""

from .uniform import sample_uniform, calculate_shift, get_lin_function
from .logit_normal import sample_logit_normal, sample_mode, sample_logsnr
from .anchor import sample_anchor, compute_anchors

_SAMPLER_MAP = {
    "uniform": sample_uniform,
    "logit_normal": sample_logit_normal,
    "acrf": sample_anchor,
    "anchor": sample_anchor,
    "mode": sample_mode,
    "logsnr": sample_logsnr,
}


def create_sampler(mode: str):
    """
    Factory: return the sampling function for the given mode.

    Args:
        mode: Sampling strategy name

    Returns:
        Callable that accepts (batch_size, ...) and returns sigmas (B,)

    Raises:
        ValueError: If mode is unknown
    """
    if mode not in _SAMPLER_MAP:
        raise ValueError(
            f"Unknown sampling mode: '{mode}'. "
            f"Available: {list(_SAMPLER_MAP.keys())}"
        )
    return _SAMPLER_MAP[mode]


__all__ = [
    "create_sampler",
    "sample_uniform",
    "sample_logit_normal",
    "sample_anchor",
    "sample_mode",
    "sample_logsnr",
    "calculate_shift",
    "get_lin_function",
    "compute_anchors",
]
