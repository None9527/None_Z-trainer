# -*- coding: utf-8 -*-
"""
Shared Optimizers — Model-agnostic

Custom optimizer implementations for mixed-precision training:
- AdamWFP8: FP8 quantized AdamW
- AdamWBF16: BF16 optimized AdamW
- Muon: Momentum + Orthogonalization (RMSNorm immune)
- MuonFP8: Muon with FP8 momentum storage
"""

from .adamw_fp8 import AdamWFP8
from .adamw_bf16 import AdamWBF16
from .muon import Muon, MuonFP8

__all__ = ["AdamWFP8", "AdamWBF16", "Muon", "MuonFP8"]
