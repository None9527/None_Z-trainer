# -*- coding: utf-8 -*-
"""
Training Domain - Repository Interfaces (Ports)

Abstract interfaces for training data persistence.
Implementations live in infrastructure/ layer.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from .entities import TrainingSession
from .value_objects import TrainingConfig


class ITrainingRepository(ABC):
    """
    Repository interface for training configuration persistence.

    Implementations:
    - TomlTrainingRepository (current TOML-based config)
    - JsonTrainingRepository (JSON-based config)
    """

    @abstractmethod
    def save_config(self, config: TrainingConfig, path: str) -> None:
        """Save training configuration to storage."""

    @abstractmethod
    def load_config(self, path: str) -> TrainingConfig:
        """Load training configuration from storage."""

    @abstractmethod
    def get_default_config(self) -> TrainingConfig:
        """Get default training configuration."""


class ITrainingSessionRepository(ABC):
    """Repository interface for training session state."""

    @abstractmethod
    def get_current_session(self) -> Optional[TrainingSession]:
        """Get the current training session, if any."""

    @abstractmethod
    def save_session(self, session: TrainingSession) -> None:
        """Persist training session state."""


class ITrainingRunner(ABC):
    """
    Interface for training process execution.

    Decouples the domain from how training is actually launched
    (subprocess, in-process, distributed, etc.)
    """

    @abstractmethod
    def start(self, config_path: str, mixed_precision: str = "bf16",
              num_gpus: int = 1, gpu_ids: str = "") -> int:
        """
        Start a training process.

        Args:
            config_path: Path to the generated TOML config file.
            mixed_precision: One of "bf16", "fp16", "no".
            num_gpus: Number of GPUs.
            gpu_ids: Comma-separated GPU IDs.

        Returns:
            Process ID or task ID
        """

    @abstractmethod
    def stop(self, process_id: int) -> None:
        """Stop a running training process."""

    @abstractmethod
    def is_running(self, process_id: int) -> bool:
        """Check if a training process is still running."""

    def get_output(self, process_id: int) -> Optional[str]:
        """Read latest output line (optional)."""
        return None
