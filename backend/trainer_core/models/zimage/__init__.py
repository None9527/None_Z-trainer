# -*- coding: utf-8 -*-
"""
Z-Image Model Package

Assembly point for Z-Image training:
- forward.py: Model-specific forward pass (text2img + omni)
- Components imported from zimage_trainer/ and shared/
"""

from .forward import forward_text2img, forward_omni

__all__ = [
    "forward_text2img",
    "forward_omni",
]
