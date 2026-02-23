# -*- coding: utf-8 -*-
"""
Training Domain - Domain Services

Business logic that doesn't belong to any single entity.
"""

import logging
from typing import Dict, List, Tuple

from .value_objects import TrainingConfig, LossConfig, TimestepConfig

logger = logging.getLogger(__name__)


class TrainingConfigValidator:
    """Validate training configuration for consistency and correctness."""

    @staticmethod
    def validate(config: TrainingConfig) -> Tuple[bool, List[str]]:
        """
        Validate training config.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Must have at least one loss enabled
        if not config.loss.has_main_loss:
            errors.append("No loss function enabled: set lambda_l1 > 0, lambda_l2 > 0, or enable frequency/style loss")

        # Learning rate sanity check
        if config.scheduler.learning_rate <= 0:
            errors.append(f"Learning rate must be positive, got {config.scheduler.learning_rate}")

        if config.scheduler.learning_rate > 0.01:
            errors.append(f"Learning rate {config.scheduler.learning_rate} is unusually high for LoRA training")

        # LoRA dim must be positive
        if config.lora.network_dim <= 0:
            errors.append(f"LoRA network_dim must be positive, got {config.lora.network_dim}")

        # ACRF-specific checks
        if config.timestep.mode == "acrf" and config.timestep.acrf_steps <= 0:
            errors.append("ACRF mode enabled but acrf_steps <= 0")

        # Batch size
        if config.batch_size <= 0:
            errors.append(f"Batch size must be positive, got {config.batch_size}")

        return len(errors) == 0, errors
