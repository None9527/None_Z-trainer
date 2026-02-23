# -*- coding: utf-8 -*-
"""
Subprocess Training Runner - Infrastructure Implementation

Launches the v2 training script (trainer_core/zimage_trainer/train.py)
as a subprocess via Accelerate.
"""

import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

from ..domain.training.repositories import ITrainingRunner

logger = logging.getLogger(__name__)


class SubprocessTrainingRunner(ITrainingRunner):
    """Launches training scripts via subprocess."""

    def __init__(self):
        self._processes: Dict[int, subprocess.Popen] = {}
        self._next_id = 1

    def start(self, config_path: str, mixed_precision: str = "bf16",
              num_gpus: int = 1, gpu_ids: str = "") -> int:
        """
        Start training as a subprocess.

        Args:
            config_path: Absolute path to the generated TOML config file.
            mixed_precision: One of "bf16", "fp16", "no".
            num_gpus: Number of GPUs to use.
            gpu_ids: Comma-separated GPU IDs (e.g. "0,1").

        Returns:
            Internal process ID for tracking.
        """
        from .config import PROJECT_ROOT

        # V2 training script location
        train_script = (
            PROJECT_ROOT / "backend" / "trainer_core"
            / "zimage_trainer" / "train.py"
        )

        if not train_script.exists():
            raise FileNotFoundError(
                f"Training script not found: {train_script}"
            )

        # Build accelerate launch command
        # Use venv's accelerate binary if available, otherwise fall back to sys.executable
        venv_dir = PROJECT_ROOT / "venv"
        venv_python = venv_dir / "bin" / "python"
        venv_accelerate = venv_dir / "bin" / "accelerate"

        if venv_accelerate.exists():
            # Use the venv accelerate CLI directly
            cmd = [
                str(venv_accelerate), "launch",
                "--mixed_precision", mixed_precision,
            ]
        elif venv_python.exists():
            # Use venv python with accelerate module
            cmd = [
                str(venv_python), "-m", "accelerate", "launch",
                "--mixed_precision", mixed_precision,
            ]
        else:
            # Fallback to current python interpreter
            cmd = [
                sys.executable, "-m", "accelerate", "launch",
                "--mixed_precision", mixed_precision,
            ]

        # Multi-GPU support
        if num_gpus > 1:
            cmd.extend(["--num_processes", str(num_gpus)])
        if gpu_ids:
            cmd.extend(["--gpu_ids", gpu_ids])

        cmd.extend([
            str(train_script),
            "--config", config_path,
        ])

        # Environment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Ensure trainer_core is importable
        trainer_core = str(PROJECT_ROOT / "backend" / "trainer_core")
        backend = str(PROJECT_ROOT / "backend")
        existing_python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{trainer_core}:{backend}:{existing_python_path}"

        logger.info(f"Starting training: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )

        pid = self._next_id
        self._next_id += 1
        self._processes[pid] = proc

        logger.info(f"Training started with ID={pid}, PID={proc.pid}")
        return pid

    def start_from_config(self, config, **kwargs) -> int:
        """Legacy compatibility: start from TrainingConfig object."""
        raise NotImplementedError(
            "Use start(config_path=...) directly. "
            "The TOML file should be generated first via TomlTrainingRepository."
        )

    def stop(self, process_id: int) -> None:
        """Stop a running training process immediately."""
        import os, signal
        proc = self._processes.get(process_id)
        if proc and proc.poll() is None:
            try:
                # Kill entire process group to ensure child processes are also killed
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            logger.info(f"Training {process_id} killed")

    def is_running(self, process_id: int) -> bool:
        """Check if training is still running."""
        proc = self._processes.get(process_id)
        if proc is None:
            return False
        return proc.poll() is None

    def get_output(self, process_id: int) -> Optional[str]:
        """Read latest output from the training process (non-blocking)."""
        proc = self._processes.get(process_id)
        if proc is None or proc.stdout is None:
            return None
        import select
        if select.select([proc.stdout], [], [], 0)[0]:
            line = proc.stdout.readline()
            if line:
                return line.decode("utf-8", errors="replace").rstrip()
        return None
