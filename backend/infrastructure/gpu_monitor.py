# -*- coding: utf-8 -*-
"""
GPU Monitor - Infrastructure Implementation

Uses nvidia-smi to query GPU information.
"""

import logging
import subprocess
from typing import List

from ..domain.system.repositories import IGPUMonitor
from ..domain.system.entities import GPUInfo

logger = logging.getLogger(__name__)


class NvidiaSmiGPUMonitor(IGPUMonitor):
    """GPU monitor using nvidia-smi CLI."""

    def get_gpu_info(self) -> List[GPUInfo]:
        """Query all GPUs via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                logger.warning(f"nvidia-smi failed: {result.stderr}")
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                try:
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total_mb=int(parts[2]),
                        memory_used_mb=int(parts[3]),
                        memory_free_mb=int(parts[4]),
                        temperature=int(parts[5]),
                        utilization=int(parts[6]),
                    ))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse GPU line: {line}, error: {e}")
            return gpus

        except FileNotFoundError:
            logger.warning("nvidia-smi not found, no GPU info available")
            return []
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out")
            return []
        except Exception as e:
            logger.error(f"GPU monitor error: {e}")
            return []
