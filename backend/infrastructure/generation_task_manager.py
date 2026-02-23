# -*- coding: utf-8 -*-
"""
Generation Task Manager — In-memory task tracking

Allows generation tasks to survive SSE disconnections (page navigation).
Tasks run in background threads; clients can poll for status/results.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class TaskState(Enum):
    PENDING = "pending"
    LOADING = "loading"
    GENERATING = "generating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationTask:
    """A tracked generation task."""
    task_id: str
    state: TaskState = TaskState.PENDING
    step: int = 0
    total_steps: int = 0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "task_id": self.task_id,
            "state": self.state.value,
            "step": self.step,
            "total_steps": self.total_steps,
            "message": self.message,
        }
        if self.result:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d


class GenerationTaskManager:
    """Thread-safe in-memory task manager.

    Stores task state so clients can:
    - Start a task → get task_id
    - Poll task status by task_id (survives page navigation)
    - Retrieve result after completion
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, GenerationTask] = {}
                    cls._instance._tasks_lock = threading.Lock()
        return cls._instance

    def create_task(self, total_steps: int) -> GenerationTask:
        task = GenerationTask(
            task_id=str(uuid.uuid4())[:8],
            total_steps=total_steps,
            created_at=time.time(),
        )
        with self._tasks_lock:
            self._tasks[task.task_id] = task
            # Cleanup old completed tasks (keep last 20)
            self._cleanup_old_tasks()
        return task

    def get_task(self, task_id: str) -> Optional[GenerationTask]:
        with self._tasks_lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs) -> None:
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if task:
                for k, v in kwargs.items():
                    if hasattr(task, k):
                        setattr(task, k, v)

    def get_active_task(self) -> Optional[GenerationTask]:
        """Return the currently running task, if any."""
        with self._tasks_lock:
            for task in self._tasks.values():
                if task.state in (TaskState.PENDING, TaskState.LOADING,
                                  TaskState.GENERATING, TaskState.SAVING):
                    return task
        return None

    def _cleanup_old_tasks(self) -> None:
        """Remove completed tasks older than 10 minutes, keep last 20."""
        now = time.time()
        completed = [
            (tid, t) for tid, t in self._tasks.items()
            if t.state in (TaskState.COMPLETED, TaskState.FAILED)
        ]
        # Sort by completion time, remove oldest beyond 20
        completed.sort(key=lambda x: x[1].completed_at, reverse=True)
        for tid, t in completed[20:]:
            del self._tasks[tid]
        # Also remove tasks older than 10 minutes
        for tid, t in list(completed):
            if now - t.completed_at > 600:
                self._tasks.pop(tid, None)


# Singleton accessor
task_manager = GenerationTaskManager()
