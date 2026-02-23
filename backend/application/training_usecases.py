# -*- coding: utf-8 -*-
"""
Application Layer - Training Use Cases

Orchestrates domain objects to perform training operations.
Contains no business rules - delegates to domain entities and services.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional

from ..domain.training.entities import TrainingSession, TrainingStatus
from ..domain.training.value_objects import TrainingConfig
from ..domain.training.repositories import (
    ITrainingRepository,
    ITrainingSessionRepository,
    ITrainingRunner,
)
from ..domain.training.services import TrainingConfigValidator

logger = logging.getLogger(__name__)


class SaveConfigUseCase:
    """Save training configuration."""

    def __init__(self, repo: ITrainingRepository):
        self._repo = repo

    def execute(self, config: TrainingConfig, path: str) -> Tuple[bool, List[str]]:
        """
        Validate and save training config.

        Returns:
            Tuple of (success, errors)
        """
        is_valid, errors = TrainingConfigValidator.validate(config)
        if not is_valid:
            return False, errors

        self._repo.save_config(config, path)
        return True, []


class StartTrainingUseCase:
    """Start a training session."""

    def __init__(
        self,
        config_repo: ITrainingRepository,
        session_repo: ITrainingSessionRepository,
        runner: ITrainingRunner,
    ):
        self._config_repo = config_repo
        self._session_repo = session_repo
        self._runner = runner

    def execute(self, config: TrainingConfig) -> Tuple[bool, str]:
        """
        Start training with given config.

        Returns:
            Tuple of (success, message)
        """
        # Validate config
        is_valid, errors = TrainingConfigValidator.validate(config)
        if not is_valid:
            return False, "; ".join(errors)

        # Check no training is already running
        current = self._session_repo.get_current_session()
        if current and current.is_active:
            return False, "Training is already in progress"

        # Create new session
        import uuid
        session = TrainingSession(id=str(uuid.uuid4()))
        session.config = config

        try:
            session.start()
            process_id = self._runner.start(config)
            session.process_id = process_id
            self._session_repo.save_session(session)
            return True, f"Training started (pid={process_id})"
        except Exception as e:
            session.fail(str(e))
            self._session_repo.save_session(session)
            return False, f"Failed to start training: {e}"


class StopTrainingUseCase:
    """Stop a running training session."""

    def __init__(
        self,
        session_repo: ITrainingSessionRepository,
        runner: ITrainingRunner,
    ):
        self._session_repo = session_repo
        self._runner = runner

    def execute(self) -> Tuple[bool, str]:
        """
        Stop the current training session.

        Returns:
            Tuple of (success, message)
        """
        session = self._session_repo.get_current_session()
        if not session or not session.is_active:
            return False, "No active training to stop"

        try:
            session.stop()
            if session.process_id:
                self._runner.stop(session.process_id)
            self._session_repo.save_session(session)
            return True, "Training stop requested"
        except Exception as e:
            return False, f"Failed to stop training: {e}"


class GetTrainingStatusUseCase:
    """Get current training session status."""

    def __init__(self, session_repo: ITrainingSessionRepository):
        self._session_repo = session_repo

    def execute(self) -> Optional[Dict[str, Any]]:
        """
        Get current training status.

        Returns:
            Status dict or None if no session
        """
        session = self._session_repo.get_current_session()
        if not session:
            return None

        return {
            "id": session.id,
            "status": session.status.value,
            "epoch": session.current_epoch,
            "step": session.current_step,
            "total_steps": session.total_steps,
            "progress": session.progress_pct,
            "loss": session.last_loss,
            "loss_components": session.loss_components,
            "error": session.error_message,
        }
