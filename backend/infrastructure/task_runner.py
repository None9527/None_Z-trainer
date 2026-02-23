# -*- coding: utf-8 -*-
"""
Async Task Runner - Shared background task infrastructure

Provides a reusable pattern for long-running tasks:
- Start a background asyncio task
- Poll status via API
- Cancel/stop gracefully
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class TaskStatus:
    """Status of a background task."""
    running: bool = False
    total: int = 0
    completed: int = 0
    current_file: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "total": self.total,
            "completed": self.completed,
            "current_file": self.current_file,
            "errors": self.errors,
        }

    def reset(self):
        self.running = False
        self.total = 0
        self.completed = 0
        self.current_file = ""
        self.errors = []


class AsyncTaskRunner:
    """Manages a single background async task with status tracking."""

    def __init__(self, name: str = "task"):
        self.name = name
        self.status = TaskStatus()
        self._task: Optional[asyncio.Task] = None
        self._cancelled = False

    @property
    def is_running(self) -> bool:
        return self.status.running

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def start(self, coro: Awaitable):
        """Start a background task from a coroutine."""
        if self._task and not self._task.done():
            raise RuntimeError(f"Task '{self.name}' is already running")

        self._cancelled = False
        self.status.reset()
        self.status.running = True

        self._task = asyncio.create_task(self._run_wrapper(coro))
        return self.status

    async def _run_wrapper(self, coro):
        """Wrapper that catches exceptions and updates status on completion."""
        try:
            await coro
        except asyncio.CancelledError:
            logger.info(f"Task '{self.name}' was cancelled")
        except Exception as e:
            logger.error(f"Task '{self.name}' failed: {e}", exc_info=True)
            self.status.errors.append(str(e))
        finally:
            self.status.running = False

    def stop(self):
        """Request task cancellation."""
        self._cancelled = True
        if self._task and not self._task.done():
            self._task.cancel()

    def get_status(self) -> dict:
        return self.status.to_dict()
