# -*- coding: utf-8 -*-
"""
Training Utilities — Model-agnostic

Provides:
- Optimizer creation (AdamW, AdamW8bit, AdamWFP8, AdamWBF16, Adafactor, Prodigy, Lion, Lion8bit, SGD)
- LR scheduler creation (constant, constant_with_warmup, linear, cosine, cosine_with_restarts, one_cycle)
- Checkpoint save/load (safetensors)

Note: Gradient clipping → shared/gradient.py
      Module offloading → shared/memory.py
"""

import logging
import math
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


# ============================================================================
# Optimizers
# ============================================================================

def get_optimizer(
    params: Union[List[Dict], List[nn.Parameter]],
    optimizer_type: str = "AdamW",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer.

    Supported optimizers:
    - AdamW: Standard PyTorch AdamW
    - AdamW8bit: 8-bit AdamW from bitsandbytes (memory efficient)
    - AdamWFP8: FP8 AdamW (custom, PyTorch native float8 + per-tensor scaling)
    - AdamWBF16: BF16 AdamW (custom, bfloat16 state storage)
    - Adafactor: Memory-efficient optimizer from transformers
    - Prodigy: Adaptive learning rate optimizer
    - Lion: Sign-based optimizer (low memory, fast)
    - Lion8bit: 8-bit Lion from bitsandbytes (lowest memory)
    - SGD: Standard SGD with momentum
    """
    opt_type = optimizer_type.lower().replace("_", "").replace("-", "")

    # --- AdamW ---
    if opt_type in ["adamw", "adam"]:
        logger.info("Using AdamW optimizer")
        return torch.optim.AdamW(
            params, lr=learning_rate, weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )

    # --- AdamW8bit ---
    elif opt_type in ["adamw8bit", "adam8bit"]:
        try:
            import bitsandbytes as bnb
            logger.info("Using AdamW8bit optimizer (8-bit quantized states)")
            return bnb.optim.AdamW8bit(
                params, lr=learning_rate, weight_decay=weight_decay,
            )
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to AdamW")
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # --- AdamWFP8 ---
    elif opt_type in ["adamwfp8", "adamfp8"]:
        from shared.optimizers import AdamWFP8
        logger.info("Using AdamWFP8 optimizer (FP8 quantized states, custom impl)")
        return AdamWFP8(
            params, lr=learning_rate, weight_decay=weight_decay,
        )

    # --- AdamWBF16 ---
    elif opt_type in ["adamwbf16", "adambf16"]:
        from shared.optimizers import AdamWBF16
        logger.info("Using AdamWBF16 optimizer (BF16 state storage, custom impl)")
        return AdamWBF16(
            params, lr=learning_rate, weight_decay=weight_decay,
        )

    # --- Adafactor ---
    elif opt_type in ["adafactor", "adafac"]:
        try:
            from transformers import Adafactor
            optimizer = Adafactor(
                params,
                lr=learning_rate,
                scale_parameter=kwargs.get("scale_parameter", False),
                relative_step=kwargs.get("relative_step", False),
                warmup_init=kwargs.get("warmup_init", False),
                weight_decay=weight_decay,
            )
            logger.info("Using Adafactor optimizer (memory efficient, no momentum states)")
            return optimizer
        except ImportError:
            logger.warning("transformers not available for Adafactor, falling back to AdamW")
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # --- Prodigy ---
    elif opt_type in ["prodigy"]:
        try:
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                params,
                lr=learning_rate if learning_rate != 1e-4 else 1.0,
                weight_decay=weight_decay,
                d_coef=kwargs.get("d_coef", 1.0),
            )
            logger.info("Using Prodigy optimizer (adaptive learning rate)")
            return optimizer
        except ImportError:
            logger.warning("prodigyopt not available, falling back to AdamW")
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # --- Lion ---
    elif opt_type in ["lion"]:
        try:
            import bitsandbytes as bnb
            logger.info("Using Lion optimizer (sign-based, low memory)")
            return bnb.optim.Lion(
                params, lr=learning_rate, weight_decay=weight_decay,
                betas=kwargs.get("betas", (0.9, 0.99)),
            )
        except ImportError:
            logger.warning("bitsandbytes not available for Lion, falling back to AdamW")
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # --- Lion8bit ---
    elif opt_type in ["lion8bit"]:
        try:
            import bitsandbytes as bnb
            logger.info("Using Lion8bit optimizer (8-bit sign-based, lowest memory)")
            return bnb.optim.Lion8bit(
                params, lr=learning_rate, weight_decay=weight_decay,
                betas=kwargs.get("betas", (0.9, 0.99)),
            )
        except ImportError:
            logger.warning("bitsandbytes not available for Lion8bit, falling back to AdamW")
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # --- SGD ---
    elif opt_type in ["sgd"]:
        logger.info("Using SGD optimizer")
        return torch.optim.SGD(
            params, lr=learning_rate, weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
        )

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# ============================================================================
# LR Schedulers
# ============================================================================

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 0,
    num_cycles: int = 1,
    **kwargs,
):
    """
    Create learning rate scheduler.

    Supported schedulers:
    - constant: Fixed LR (no decay)
    - constant_with_warmup: Fixed LR with linear warmup
    - linear: Linear decay after warmup
    - cosine: Cosine annealing after warmup
    - cosine_with_restarts: Cosine annealing with hard restarts
    - one_cycle: PyTorch OneCycleLR (peak → anneal)
    """
    from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

    sched_type = scheduler_type.lower().replace("-", "_").strip()

    # --- constant ---
    if sched_type == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)

    # --- constant_with_warmup ---
    elif sched_type == "constant_with_warmup":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return 1.0
        return LambdaLR(optimizer, lr_lambda)

    # --- linear ---
    elif sched_type == "linear":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return max(0.0, 1.0 - (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps))
        return LambdaLR(optimizer, lr_lambda)

    # --- cosine ---
    elif sched_type == "cosine":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(progress * math.pi)))
        return LambdaLR(optimizer, lr_lambda)

    # --- cosine_with_restarts ---
    elif sched_type == "cosine_with_restarts":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(progress * num_cycles * 2.0 * math.pi)))
        return LambdaLR(optimizer, lr_lambda)

    # --- one_cycle ---
    elif sched_type == "one_cycle":
        max_lr = optimizer.param_groups[0]["lr"]
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=num_training_steps,
            pct_start=kwargs.get("pct_start", 0.3),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4),
        )

    else:
        logger.warning(f"Unknown scheduler '{scheduler_type}', defaulting to constant")
        return LambdaLR(optimizer, lambda _: 1.0)


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def save_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    metadata: Optional[Dict[str, str]] = None,
):
    """Save checkpoint to safetensors."""
    converted = {k: v.to(dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    save_file(converted, path, metadata=metadata)
    logger.info(f"Saved: {path}")


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load checkpoint from safetensors."""
    from safetensors.torch import load_file
    return load_file(path, device=device)

