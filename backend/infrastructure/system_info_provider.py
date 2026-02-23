# -*- coding: utf-8 -*-
"""
System Info Provider - Infrastructure Implementation

Collects OS, Python, Torch, CUDA version information.
"""

import platform
import sys
import logging

from ..domain.system.repositories import ISystemInfoProvider
from ..domain.system.entities import SystemInfo

logger = logging.getLogger(__name__)


class LocalSystemInfoProvider(ISystemInfoProvider):
    """Collects system info from the local environment."""

    def get_system_info(self) -> SystemInfo:
        torch_version = ""
        cuda_version = ""
        try:
            import torch
            torch_version = torch.__version__
            cuda_version = torch.version.cuda or ""
        except ImportError:
            pass

        return SystemInfo(
            os=f"{platform.system()} {platform.release()}",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=torch_version,
            cuda_version=cuda_version,
        )
