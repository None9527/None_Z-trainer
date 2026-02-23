# -*- coding: utf-8 -*-
"""
Z-Image Specific Training Components

Modules:
    norm_opt         — Norm_opt scaling for proxy target training
    target_schedule  — Content / quality mode blending
"""

from .norm_opt import compute_norm_opt_scale
from .target_schedule import compute_target_with_schedule

__all__ = [
    "compute_norm_opt_scale",
    "compute_target_with_schedule",
]
