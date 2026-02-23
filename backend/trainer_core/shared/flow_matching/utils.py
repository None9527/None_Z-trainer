# -*- coding: utf-8 -*-
"""
Flow Matching Training Utilities — Model-agnostic
"""

import logging

logger = logging.getLogger(__name__)


def log_training_info(
    epoch: int,
    step: int,
    loss_dict: dict,
    lr: float,
    alpha_t: float = None,
):
    """Log training information in a standardized format."""
    loss_str = ", ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
    alpha_str = f", α(t)={alpha_t:.2f}" if alpha_t is not None else ""
    logger.info(f"[Epoch {epoch}][Step {step}] {loss_str}{alpha_str}, lr={lr:.2e}")
