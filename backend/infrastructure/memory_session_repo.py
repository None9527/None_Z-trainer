# -*- coding: utf-8 -*-
"""
Memory Session Repository - Infrastructure Implementation

In-memory storage for the current training session state.
"""

import logging
from typing import Optional

from ..domain.training.repositories import ITrainingSessionRepository
from ..domain.training.entities import TrainingSession

logger = logging.getLogger(__name__)


class MemorySessionRepository(ITrainingSessionRepository):
    """In-memory training session store (singleton per process)."""

    _current_session: Optional[TrainingSession] = None

    def get_current_session(self) -> Optional[TrainingSession]:
        """Get the current training session."""
        return self._current_session

    def save_session(self, session: TrainingSession) -> None:
        """Persist session state in memory."""
        MemorySessionRepository._current_session = session
        logger.debug(f"Session saved: status={session.status}")
