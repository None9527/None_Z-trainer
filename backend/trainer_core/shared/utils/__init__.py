# -*- coding: utf-8 -*-
"""
Shared Utilities — Model-agnostic

Training infrastructure reusable across all models:
- training_utils: Optimizer creation, LR schedulers, checkpoint I/O
- memory_optimizer: Block swap manager, activation checkpointing
- block_swapper: Forward-pass layer swapping for low VRAM
- model_hooks: PyTorch hook-based optimizations (block swap, attention backend)
- hardware_detector: GPU/CPU/xformers auto-detection
- degradation: Image degradation for img2img training
- timestep_aware_loss: Timestep-aware loss weight scheduling
- lr_schedulers: OneCycleLR + diffusers schedulers

Note: Gradient clipping → shared.gradient
      Module offloading → shared.memory
"""

from .training_utils import (
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    load_checkpoint,
)
from .lr_schedulers import get_scheduler_with_onecycle
from .hardware_detector import HardwareDetector
from .block_swapper import ForwardBlockSwapper, create_block_swapper
from .model_hooks import (
    BlockSwapperHook,
    apply_block_swapper,
    apply_attention_optimization,
    enable_gradient_checkpointing,
    apply_all_optimizations,
)
from .degradation import ImageDegradation, BatchDegradation, create_degradation_transform
from .timestep_aware_loss import TimestepAwareLossScheduler

__all__ = [
    # Training
    "get_optimizer",
    "get_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    # LR Schedulers
    "get_scheduler_with_onecycle",
    # Hardware
    "HardwareDetector",
    # Block Swap
    "ForwardBlockSwapper",
    "create_block_swapper",
    # Model Hooks
    "BlockSwapperHook",
    "apply_block_swapper",
    "apply_attention_optimization",
    "enable_gradient_checkpointing",
    "apply_all_optimizations",
    # Degradation
    "ImageDegradation",
    "BatchDegradation",
    "create_degradation_transform",
    # Timestep-aware loss
    "TimestepAwareLossScheduler",
]
