# -*- coding: utf-8 -*-
"""
Flow Matching Training Utilities — Model-agnostic

Provides the core building blocks for flow matching based training:

    noising         — z_t creation, velocity target computation
    samplers        — Timestep sampling strategies (uniform, logit_normal, anchor)
    loss_weighting  — Timestep-dependent loss weighting schemes
    utils           — Training logging
"""

from .noising import get_z_t, get_noisy_model_input, compute_velocity_target
from .loss_weighting import compute_loss_weighting, flow_matching_loss
from .utils import log_training_info

__all__ = [
    # Noising
    "get_z_t",
    "get_noisy_model_input",
    "compute_velocity_target",
    # Loss weighting
    "compute_loss_weighting",
    "flow_matching_loss",
    # Utils
    "log_training_info",
]
