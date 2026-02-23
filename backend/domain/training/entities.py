# -*- coding: utf-8 -*-
"""
Training Domain - Entities

Core business entities for training session management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime


class TrainingStatus(Enum):
    """Training session lifecycle states."""
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPING = "stopping"


class TrainingMode(Enum):
    """Training approach."""
    LORA = "lora"
    FULL_FINETUNE = "full_finetune"


class ModelType(Enum):
    """Supported model types (extensible per convention)."""
    ZIMAGE = "zimage"
    # Future: FLUX = "flux", SD3 = "sd3", etc.


@dataclass
class TrainingSession:
    """
    Aggregate Root - Training Session

    Represents a single training run with all its configuration and state.
    This is the central entity that orchestrates training lifecycle.
    """
    id: str
    status: TrainingStatus = TrainingStatus.IDLE
    model_type: ModelType = ModelType.ZIMAGE
    training_mode: TrainingMode = TrainingMode.LORA
    config: Optional["TrainingConfig"] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_epoch: int = 0
    current_step: int = 0
    total_steps: int = 0
    last_loss: float = 0.0
    loss_components: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    process_id: Optional[int] = None

    def start(self) -> None:
        """Transition to running state."""
        if self.status not in (TrainingStatus.IDLE, TrainingStatus.COMPLETED, TrainingStatus.FAILED):
            raise ValueError(f"Cannot start training from status: {self.status}")
        self.status = TrainingStatus.RUNNING
        self.started_at = datetime.now()
        self.error_message = None

    def stop(self) -> None:
        """Request training stop."""
        if self.status != TrainingStatus.RUNNING:
            raise ValueError(f"Cannot stop training from status: {self.status}")
        self.status = TrainingStatus.STOPPING

    def complete(self) -> None:
        """Mark training as completed."""
        self.status = TrainingStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """Mark training as failed."""
        self.status = TrainingStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now()

    def update_progress(self, epoch: int, step: int, loss: float,
                        components: Optional[Dict[str, float]] = None) -> None:
        """Update training progress."""
        self.current_epoch = epoch
        self.current_step = step
        self.last_loss = loss
        if components:
            self.loss_components = components

    @property
    def is_active(self) -> bool:
        """Check if training is currently active."""
        return self.status in (TrainingStatus.RUNNING, TrainingStatus.PREPARING)

    @property
    def progress_pct(self) -> float:
        """Get progress percentage."""
        if self.total_steps <= 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)
